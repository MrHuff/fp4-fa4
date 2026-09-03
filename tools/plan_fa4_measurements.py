#!/usr/bin/env python3
"""Validate and print the paper measurement command graph without running it.

This is deliberately a planner, not a launcher.  It never imports CUDA, starts
training, downloads data, or invokes a scheduler.  A command is printed as
executable only when every source, compiled artifact, and external asset that
the command can authenticate is present.  Historical recipes whose immutable
inputs or exact invocation were not preserved remain visible as blocked nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "release" / "EXPERIMENT_MATRIX.md"
ARTIFACT_SCHEMA = "fa4_artifact_manifest_v3"
ASSET_SCHEMA = "fa4_external_assets_v1"
LAUNCHER_SCHEMA = "fa4_torchrun_launcher_v1"
EXPECTED_CUTLASS_DSL = "4.5.2"

SHA256 = re.compile(r"[0-9a-f]{64}")
IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
GPU_SELECTOR = re.compile(r"[0-9]+")
ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SECRET_NAME = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|ACCESS_KEY|PRIVATE_KEY|API_KEY)",
    re.IGNORECASE,
)
SCRUB_ENVIRONMENT = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
)

FAMILIES: dict[str, str] = {
    "noncausal-forward": "Non-causal HAO grid and FP8 controls",
    "downstream": "ViT, BERT masked-LM, and SST-2 replays",
    "vit-mae": "ViT-MAE reconstruction",
    "wan": "Wan 2.1 video replay",
    "b300-aggregate": "B300 aggregate reconstruction",
    "causal-backward": "Isolated causal backward",
    "projection-fwd-bwd": "Projection-inclusive forward and backward",
    "llama8b-e2e": "Single-GPU 8B B1/B2/B4 full-update timing",
    "distributed-training": (
        "Matched distributed BF16 and E4M3/NVFP4 projection by " "FP8/MXFP4-PV training"
    ),
    "mxfp4-divergence": "MXFP4-P/V divergence diagnostic",
    "llama1p2b-d64": (
        "D64 isolated boundaries, saturated 1.2B, real-data numerics, and "
        "DDP16 save/resume"
    ),
}

EXPECTED_ASSETS: dict[str, tuple[str, str | None]] = {
    "vit_cifar10_model": (
        "huggingface_snapshot",
        "nateraw/vit-base-patch16-224-cifar10",
    ),
    "cifar10_dataset": ("dataset_snapshot", "uoft-cs/cifar10"),
    "bert_mlm_model": (
        "huggingface_snapshot",
        "google-bert/bert-base-uncased",
    ),
    "wikitext_dataset": ("dataset_snapshot", "Salesforce/wikitext"),
    "bert_sst2_model": (
        "huggingface_snapshot",
        "textattack/bert-base-uncased-SST-2",
    ),
    "sst2_dataset": ("dataset_snapshot", "glue"),
    "vit_mae_model": ("huggingface_snapshot", "facebook/vit-mae-base"),
    "coco_val_100": ("image_subset", None),
    "wan_1_3b_model": (
        "huggingface_snapshot",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ),
    "wan_14b_model": (
        "huggingface_snapshot",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    ),
    "llama3_8b_assets": (
        "huggingface_snapshot",
        "meta-llama/Llama-3.1-8B",
    ),
    "slimpajama_dataset": (
        "dataset_snapshot",
        "cerebras/SlimPajama-627B",
    ),
}

PRIMARY_HAO_SHARDS = (
    ("1,256,16,128", "1,1024,16,128", "4,4096,16,128"),
    ("1,32768,16,128", "4,4096,32,128"),
    ("1,4096,12,128", "1,32768,12,128"),
    ("1,4096,24,128", "1,32768,24,128", "1,32768,24,64"),
)
UNIFIED_HAO_SHAPES = (
    "1,256,16,128",
    "1,1024,24,128",
    "1,2048,24,128",
    "1,4096,24,128",
    "1,4096,64,128",
    "1,8192,64,128",
)
DOWNSTREAM_TASKS = (
    "vit-s256",
    "vit-s1024",
    "vit-s4096",
    "bert-mlm-s256",
    "bert-mlm-s512",
    "bert-sst2-s256",
)
DOWNSTREAM_ASSETS = {
    "vit-s256": ("vit_cifar10_model", "cifar10_dataset"),
    "vit-s1024": ("vit_cifar10_model", "cifar10_dataset"),
    "vit-s4096": ("vit_cifar10_model", "cifar10_dataset"),
    "bert-mlm-s256": ("bert_mlm_model", "wikitext_dataset"),
    "bert-mlm-s512": ("bert_mlm_model", "wikitext_dataset"),
    "bert-sst2-s256": ("bert_sst2_model", "sst2_dataset"),
}
ROUTES = (
    "bf16_fa4",
    "nvfp4_qk_fp8_pv",
    "nvfp4_qk_mxfp4_pv",
    "e4m3_proj_nvfp4_qk_fp8_pv",
    "e4m3_proj_nvfp4_qk_mxfp4_pv",
)
LOWP_ROUTES = ROUTES[1:]
BATCHES = (1, 2, 4)
D64_ROUTES = (
    "bf16_fa4",
    "e4m3_proj_nvfp4_qk_fp8_pv",
    "e4m3_proj_nvfp4_qk_mxfp4_pv",
)
D64_BATCH = 16

OPTIMIZER_ENVIRONMENT = {
    "FA4_DATALOADER_PIN_MEMORY": "0",
    "FA4_DATALOADER_PREFETCH_FACTOR": "8",
    "FA4_TRAIN_DATALOADER_NUM_WORKERS": "8",
    "FA4_VALIDATION_DATALOADER_NUM_WORKERS": "1",
    "LBT_DCP_SYNC_CPU_PROCESS_GROUP": "1",
    "TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC": "1",
    "LBT_ADAMW_BF16_SR_PROVIDER": "lbt_fused_stateless_adamw_bf16_sr",
    "LBT_ADAMW_BF16_SR_PROVIDER_VERSION": "1",
    "LBT_ADAMW_BF16_SR_SOURCE_SHA256": (
        "05e9133ac24ac286e059ebaaef4311921c5566f0b57e07367af30ac2f48f4dbd"
    ),
    "LBT_ADAMW_BF16_SR_SEED": "0",
    "LBT_ADAMW_BF16_SR_CHECKPOINT_SCHEMA": "v2-fused-stateless",
}
RESERVED_LAUNCH_ENVIRONMENT = {
    *OPTIMIZER_ENVIRONMENT,
    "FA4_EXTERNAL_ASSETS_MANIFEST",
    "FA4_LAUNCHER_SOURCE_REVISION",
}


class PlanError(RuntimeError):
    """A planner input violates the fail-closed reproduction contract."""


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    bytes: int
    sha256: str

    def authenticate(self, label: str) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {self.path}")
        if self.path.stat().st_size != self.bytes:
            raise PlanError(
                f"{label} byte identity mismatch: "
                f"{self.path.stat().st_size} != {self.bytes}"
            )
        observed = _sha256(self.path)
        if observed != self.sha256:
            raise PlanError(f"{label} SHA256 mismatch: {observed} != {self.sha256}")


@dataclass(frozen=True)
class ExternalAsset:
    name: str
    kind: str
    identifier: str
    revision: str
    root: Path
    files: tuple[FileIdentity, ...]
    tree_sha256: str


@dataclass(frozen=True)
class Launcher:
    executable: FileIdentity
    source_revision: str
    world_size: int
    argv_prefix: tuple[str, ...]
    environment: Mapping[str, str]


@dataclass(frozen=True)
class Step:
    name: str
    families: tuple[str, ...]
    kind: str
    description: str
    cwd: Path
    command: tuple[str, ...] | None
    environment: Mapping[str, str]
    dependencies: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    @property
    def runnable(self) -> bool:
        return self.command is not None and not self.blockers

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["cwd"] = str(self.cwd)
        value["command"] = list(self.command) if self.command is not None else None
        value["environment"] = dict(sorted(self.environment.items()))
        value["outputs"] = [str(path) for path in self.outputs]
        value["runnable"] = self.runnable
        return value


@dataclass(frozen=True)
class Context:
    python: Path
    output_root: Path
    noncausal_build_root: Path
    cuda_home: Path | None
    cutlass_dsl_root: Path | None
    gpu: str
    artifacts: Mapping[tuple[str, int], Any]
    artifact_errors: tuple[str, ...]
    assets: Mapping[str, ExternalAsset]
    asset_manifest_path: Path | None
    launcher: Launcher | None
    launcher_error: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(
    value: str | Path,
    label: str,
    *,
    must_exist: bool = False,
    preserve_final_symlink: bool = False,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PlanError(f"{label} must be an absolute path: {path}")
    path = Path(os.path.abspath(path))
    if not preserve_final_symlink:
        path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _strict_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise PlanError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _identity_from_json(
    value: object, label: str, *, relative_to: Path | None
) -> FileIdentity:
    raw = _strict_mapping(value, label)
    _exact_keys(raw, {"path", "bytes", "sha256"}, label)
    if not isinstance(raw["path"], str) or not raw["path"]:
        raise PlanError(f"{label}.path must be a non-empty string")
    path = Path(raw["path"])
    if relative_to is None:
        path = _absolute(path, f"{label}.path")
    else:
        if path.is_absolute() or ".." in path.parts:
            raise PlanError(f"{label}.path must be a safe relative path")
        path = (relative_to / path).resolve()
        try:
            path.relative_to(relative_to)
        except ValueError as error:
            raise PlanError(f"{label}.path escapes its asset root") from error
    size = raw["bytes"]
    digest = raw["sha256"]
    if type(size) is not int or size <= 0:
        raise PlanError(f"{label}.bytes must be a positive integer")
    if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        raise PlanError(f"{label}.sha256 must be lowercase SHA256")
    identity = FileIdentity(path=path, bytes=size, sha256=digest)
    identity.authenticate(label)
    return identity


def _asset_tree_digest(root: Path, files: Sequence[FileIdentity]) -> str:
    digest = hashlib.sha256()
    for identity in sorted(
        files, key=lambda item: item.path.relative_to(root).as_posix()
    ):
        relative = identity.path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(identity.bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(identity.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_external_assets(
    path: Path | None,
) -> tuple[dict[str, ExternalAsset], Path | None]:
    if path is None:
        return {}, None
    manifest_path = _absolute(path, "external asset manifest", must_exist=True)
    raw = _strict_mapping(json.loads(manifest_path.read_text()), "asset manifest")
    _exact_keys(raw, {"schema", "assets"}, "asset manifest")
    if raw["schema"] != ASSET_SCHEMA:
        raise PlanError(f"unsupported asset schema: {raw['schema']!r}")
    assets_raw = _strict_mapping(raw["assets"], "assets")
    result: dict[str, ExternalAsset] = {}
    for name, value in assets_raw.items():
        if not isinstance(name, str) or not name:
            raise PlanError("asset names must be non-empty strings")
        item = _strict_mapping(value, f"assets.{name}")
        _exact_keys(
            item,
            {"kind", "identifier", "revision", "root", "files", "tree_sha256"},
            f"assets.{name}",
        )
        revision = item["revision"]
        tree_sha256 = item["tree_sha256"]
        if (
            not isinstance(revision, str)
            or IMMUTABLE_REVISION.fullmatch(revision) is None
        ):
            raise PlanError(
                f"assets.{name}.revision must be an immutable 40- or 64-hex id"
            )
        if not isinstance(tree_sha256, str) or SHA256.fullmatch(tree_sha256) is None:
            raise PlanError(f"assets.{name}.tree_sha256 must be lowercase SHA256")
        root = _absolute(item["root"], f"assets.{name}.root", must_exist=True)
        if not root.is_dir():
            raise PlanError(f"assets.{name}.root must be a directory: {root}")
        files_raw = item["files"]
        if not isinstance(files_raw, list) or not files_raw:
            raise PlanError(f"assets.{name}.files must be a non-empty list")
        files = tuple(
            _identity_from_json(
                record, f"assets.{name}.files[{index}]", relative_to=root
            )
            for index, record in enumerate(files_raw)
        )
        relative_names = [file.path.relative_to(root).as_posix() for file in files]
        if len(set(relative_names)) != len(relative_names):
            raise PlanError(f"assets.{name}.files contains duplicate paths")
        declared_names = {
            Path(_strict_mapping(record, "asset file")["path"]).as_posix()
            for record in files_raw
        }
        observed_names: set[str] = set()
        symlink_directories: list[str] = []
        for directory, directories, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            directories.sort()
            filenames.sort()
            for child in directories:
                child_path = directory_path / child
                if child_path.is_symlink():
                    symlink_directories.append(child_path.relative_to(root).as_posix())
            observed_names.update(
                (directory_path / filename).relative_to(root).as_posix()
                for filename in filenames
            )
        if symlink_directories:
            raise PlanError(
                f"assets.{name}.root contains unauthenticated symlink directories: "
                f"{symlink_directories}"
            )
        if observed_names != declared_names:
            raise PlanError(
                f"assets.{name}.files is not exhaustive: "
                f"missing={sorted(observed_names - declared_names)}, "
                f"absent={sorted(declared_names - observed_names)}"
            )
        observed_tree = _asset_tree_digest(root, files)
        if observed_tree != tree_sha256:
            raise PlanError(
                f"assets.{name}.tree_sha256 mismatch: {observed_tree} != {tree_sha256}"
            )
        kind = item["kind"]
        identifier = item["identifier"]
        if not isinstance(kind, str) or not kind:
            raise PlanError(f"assets.{name}.kind must be non-empty")
        if not isinstance(identifier, str) or not identifier:
            raise PlanError(f"assets.{name}.identifier must be non-empty")
        result[name] = ExternalAsset(
            name=name,
            kind=kind,
            identifier=identifier,
            revision=revision,
            root=root,
            files=files,
            tree_sha256=tree_sha256,
        )
    return result, manifest_path


def load_launcher(path: Path | None) -> tuple[Launcher | None, str | None]:
    if path is None:
        return None, None
    try:
        manifest_path = _absolute(path, "launcher manifest", must_exist=True)
        raw = _strict_mapping(
            json.loads(manifest_path.read_text()), "launcher manifest"
        )
        _exact_keys(
            raw,
            {
                "schema",
                "source_revision",
                "world_size",
                "executable",
                "argv_prefix",
                "environment",
            },
            "launcher manifest",
        )
        if raw["schema"] != LAUNCHER_SCHEMA:
            raise PlanError(f"unsupported launcher schema: {raw['schema']!r}")
        revision = raw["source_revision"]
        if (
            not isinstance(revision, str)
            or IMMUTABLE_REVISION.fullmatch(revision) is None
        ):
            raise PlanError("launcher source_revision must be immutable 40- or 64-hex")
        world_size = raw["world_size"]
        if type(world_size) is not int or world_size not in (16, 64):
            raise PlanError(
                "paper distributed launcher must declare world_size 16 or 64"
            )
        executable = _identity_from_json(
            raw["executable"], "launcher executable", relative_to=None
        )
        if not os.access(executable.path, os.X_OK):
            raise PlanError(f"launcher executable is not executable: {executable.path}")
        argv = raw["argv_prefix"]
        if not isinstance(argv, list) or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise PlanError("launcher argv_prefix must be a list of non-empty strings")
        forbidden = {
            "-m",
            "torchtitan.train",
            "torchtitan.experiments.fa4.train",
            "--job.config-file",
        }
        if forbidden.intersection(argv):
            raise PlanError(
                "launcher argv_prefix must stop before the training module; "
                "the planner appends the audited FA4 trainer and config"
            )

        def integer_option(*names: str) -> int:
            values: list[str] = []
            for index, argument in enumerate(argv):
                for name in names:
                    if argument == name:
                        if index + 1 >= len(argv):
                            raise PlanError(f"launcher {name} has no value")
                        values.append(argv[index + 1])
                    elif argument.startswith(name + "="):
                        values.append(argument.split("=", 1)[1])
            if len(values) != 1 or not values[0].isdigit():
                raise PlanError(
                    f"launcher must set exactly one integer {'/'.join(names)}"
                )
            return int(values[0])

        nodes = integer_option("--nnodes")
        processes_per_node = integer_option("--nproc-per-node", "--nproc_per_node")
        if nodes * processes_per_node != world_size:
            raise PlanError(
                "launcher topology does not match world_size: "
                f"{nodes} * {processes_per_node} != {world_size}"
            )
        environment = _strict_mapping(raw["environment"], "launcher environment")
        checked_environment: dict[str, str] = {}
        for name, value in environment.items():
            if (
                not isinstance(name, str)
                or ENVIRONMENT_NAME.fullmatch(name) is None
                or SECRET_NAME.search(name)
            ):
                raise PlanError(f"unsafe launcher environment name: {name!r}")
            if name in RESERVED_LAUNCH_ENVIRONMENT:
                raise PlanError(
                    f"launcher environment may not override audited setting {name!r}"
                )
            if not isinstance(value, str) or "\0" in value:
                raise PlanError(f"launcher environment value for {name!r} must be text")
            checked_environment[name] = value
        return (
            Launcher(
                executable=executable,
                source_revision=revision,
                world_size=world_size,
                argv_prefix=tuple(argv),
                environment=checked_environment,
            ),
            None,
        )
    except Exception as error:  # retain the reason in every launch node
        return None, str(error)


def _artifact_schema_module() -> ModuleType:
    path = ROOT / "torchtitan" / "experiments" / "fa4" / "artifacts.py"
    specification = importlib.util.spec_from_file_location(
        "_fa4_measurement_artifact_schema", path
    )
    if specification is None or specification.loader is None:
        raise PlanError(f"cannot load artifact schema: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_artifacts(
    paths: Sequence[Path],
) -> tuple[dict[tuple[str, int], Any], tuple[str, ...]]:
    if not paths:
        return {}, ()
    schema = _artifact_schema_module()
    result: dict[tuple[str, int], Any] = {}
    errors: list[str] = []
    for raw_path in paths:
        try:
            path = _absolute(raw_path, "artifact manifest", must_exist=True)
            manifest = schema.load_artifact_manifest(path, require_training=False)
            key = (manifest.route, manifest.batch)
            if key in result:
                raise PlanError(f"duplicate artifact manifest for {key}")
            result[key] = manifest
        except Exception as error:
            errors.append(f"{raw_path}: {error}")
    return result, tuple(errors)


def _source_blockers(paths: Iterable[Path]) -> tuple[str, ...]:
    return tuple(
        f"source input is missing: {path}" for path in paths if not path.is_file()
    )


def _forward_build_sources(directory: Path, source_root: Path) -> tuple[Path, ...]:
    """Direct scripts, Makefiles, and quoted includes used by both forward builds."""
    names = (
        "Makefile.hao_direct",
        "Makefile.hao_direct_fp4pv",
        "hao_direct_benchmark.py",
        "hao_direct_fp4pv_benchmark.py",
        "hao_native_reference_benchmark.py",
        "hao_direct_candidate.cu",
        "hao_direct_fp4pv_candidate.cu",
        "upstream_mxfp4_fp8pv_bf16_baseline.inc",
        "fwd_configs.inc",
        "fwd_device_helpers.inc",
        "fwd_qstage2_two_n64.inc",
        "stage2_ex2_alu_helpers.cuh",
        "depth1_upstream_mxfp4_fp8pv_kernel.inc",
        "depth1_upstream_mxfp4_fp8pv_quarter_reader.inc",
        "hao_direct_config.inc",
        "hao_direct_kernel.inc",
        "hao_direct_softmax_reader.inc",
        "shared_host_helpers.inc",
        "hao_direct_host.inc",
        "hao_direct_fp4pv_config.inc",
        "hao_direct_fp4pv_kernel.inc",
        "hao_direct_fp4pv_softmax_reader.inc",
        "hao_direct_fp4pv_host.inc",
    )
    return (
        *(directory / name for name in names),
        source_root / "ThunderKittens/kernels/common.mk",
        source_root / "ThunderKittens/include/kittens.cuh",
    )


def _artifact(
    context: Context, route: str, batch: int, *, training: bool
) -> tuple[Any | None, tuple[str, ...]]:
    manifest = context.artifacts.get((route, batch))
    blockers = list(context.artifact_errors)
    if manifest is None:
        blockers.append(
            f"missing explicit {ARTIFACT_SCHEMA} for route={route}, batch={batch}"
        )
        return None, tuple(blockers)
    if training and manifest.purpose != "training":
        blockers.append(
            f"route={route}, batch={batch} manifest is operator_only, not training"
        )
    if training and (
        manifest.runtime_source is None
        or manifest.flash_interface is None
        or manifest.cutlass_dsl is None
    ):
        blockers.append(
            f"route={route}, batch={batch} training manifest omits authenticated runtime sources"
        )
    return manifest, tuple(blockers)


def _asset(context: Context, name: str) -> tuple[ExternalAsset | None, tuple[str, ...]]:
    asset = context.assets.get(name)
    if asset is None:
        return None, (f"missing authenticated external asset {name!r}",)
    expected_kind, expected_identifier = EXPECTED_ASSETS[name]
    blockers: list[str] = []
    if asset.kind != expected_kind:
        blockers.append(f"{name}.kind={asset.kind!r} does not match {expected_kind!r}")
    if expected_identifier is not None and asset.identifier != expected_identifier:
        blockers.append(
            f"{name}.identifier={asset.identifier!r} does not match "
            f"{expected_identifier!r}"
        )
    return asset, tuple(blockers)


def _runtime_environment_and_blockers(
    manifest: Any,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Bind a benchmark process to the runtime sources authenticated at build."""
    runtime = manifest.runtime_source
    flash = manifest.flash_interface
    cutlass = manifest.cutlass_dsl
    if runtime is None or flash is None or cutlass is None:
        return {}, (
            "artifact manifest omits runtime_source, flash_interface, or CUTLASS DSL; "
            "rebuild without --operator-only for this CuTe-backed benchmark",
        )
    blockers: list[str] = []
    current_runtime = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
    current_flash = ROOT / "flash-attention/flash_attn/cute/interface.py"
    for label, current, authenticated in (
        ("benchmark runtime", current_runtime, runtime),
        ("FlashAttention interface", current_flash, flash),
    ):
        if not current.is_file():
            blockers.append(f"{label} source is missing: {current}")
        elif (
            current.stat().st_size != authenticated.bytes
            or _sha256(current) != authenticated.sha256
        ):
            blockers.append(
                f"current {label} does not match the artifact manifest identity"
            )
    return {
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(cutlass.root))),
    }, tuple(blockers)


def _base_environment(context: Context) -> dict[str, str]:
    environment = {
        "CUDA_VISIBLE_DEVICES": context.gpu,
        "PATH": os.pathsep.join(
            (str(context.python.parent), os.environ.get("PATH", ""))
        ),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if context.asset_manifest_path is not None:
        environment["FA4_EXTERNAL_ASSETS_MANIFEST"] = str(context.asset_manifest_path)
    if context.cuda_home is not None:
        environment["CUDA_HOME"] = str(context.cuda_home)
        environment["CUDACXX"] = str(context.cuda_home / "bin/nvcc")
    if context.cutlass_dsl_root is not None:
        environment["PYTHONPATH"] = str(context.cutlass_dsl_root)
    return environment


def _noncausal_steps(context: Context) -> list[Step]:
    script = (
        ROOT
        / "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd"
        / "hao_comprehensive_suite.py"
    )
    environment = {
        **_base_environment(context),
        "HAO_FLASH_ATTN_ROOT": str(ROOT / "third_party/hao_flash_attention_fp4"),
    }
    source_directory = script.parent
    build_sources = _forward_build_sources(
        source_directory,
        ROOT / "reproduction/snapshots/forward_cfc06dad",
    )
    steps: list[Step] = []
    for variant, route_name in (
        ("nvmx-fast", "fast"),
        ("nvmx-accurate", "accurate"),
    ):
        for shard, shapes in enumerate(PRIMARY_HAO_SHARDS):
            command = [str(context.python), "-B", str(script)]
            for shape in shapes:
                command.extend(("--shape", shape))
            output_dir = (
                context.output_root / "noncausal" / f"{route_name}_shard{shard}"
            )
            build_root = context.noncausal_build_root / f"{route_name}_shard{shard}"
            command.extend(
                (
                    "--variant",
                    variant,
                    "--warmup-ms",
                    "300",
                    "--rep-ms",
                    "3000",
                    "--cooldown-seconds",
                    "0.8",
                    "--seed",
                    "20260814",
                    "--gpu",
                    context.gpu,
                    "--output-dir",
                    str(output_dir),
                    "--build-root",
                    str(build_root),
                )
            )
            steps.append(
                Step(
                    name=f"noncausal.hao-grid.{route_name}.shard{shard}",
                    families=("noncausal-forward",),
                    kind="gpu-measurement",
                    description=f"Matched primary HAO grid for {variant}, shard {shard}",
                    cwd=ROOT,
                    command=tuple(command),
                    environment=environment,
                    outputs=(output_dir,),
                    blockers=_source_blockers(
                        (
                            script,
                            *build_sources,
                            ROOT
                            / "third_party/hao_flash_attention_fp4/flash_attn/cute/flash_fwd_sm100_fp4.py",
                        )
                    ),
                )
            )
    command = [str(context.python), "-B", str(script)]
    for shape in UNIFIED_HAO_SHAPES:
        command.extend(("--shape", shape))
    command.extend(
        (
            "--variant",
            "all",
            "--warmup-ms",
            "300",
            "--rep-ms",
            "3000",
            "--cooldown-seconds",
            "0.8",
            "--seed",
            "20260814",
            "--gpu",
            context.gpu,
            "--output-dir",
            str(context.output_root / "noncausal" / "unified"),
            "--build-root",
            str(context.noncausal_build_root / "unified"),
        )
    )
    steps.append(
        Step(
            name="noncausal.unified",
            families=("noncausal-forward",),
            kind="gpu-measurement",
            description="Six-shape FP4/FP8 saturation-control suite",
            cwd=ROOT,
            command=tuple(command),
            environment=environment,
            outputs=(context.output_root / "noncausal" / "unified",),
            blockers=_source_blockers((script, *build_sources)),
        )
    )
    return steps


def _downstream_steps(context: Context) -> list[Step]:
    script = ROOT / "tk_fa4/fp4_fa4_fwd/downstream_provider_suite.py"
    downstream_sources = (
        script,
        ROOT / "tk_fa4/fp4_fa4_fwd/eval_regular_attention.py",
        ROOT / "tk_fa4/fp4_fa4_fwd/eval_bert_mlm_attention.py",
        ROOT / "tk_fa4/fp4_fa4_fwd/eval_bert_sequence_classification.py",
    )
    steps = []
    for task in DOWNSTREAM_TASKS:
        model_name, dataset_name = DOWNSTREAM_ASSETS[task]
        model, model_blockers = _asset(context, model_name)
        dataset, dataset_blockers = _asset(context, dataset_name)
        command = (
            str(context.python),
            "-B",
            str(script),
            "--task",
            task,
            "--gpu",
            context.gpu,
            "--output-dir",
            str(context.output_root / "downstream"),
            "--extension-root",
            str(context.noncausal_build_root / "unified"),
            "--model-root",
            str(model.root) if model else f"<missing-{model_name}>",
            "--dataset-root",
            str(dataset.root) if dataset else f"<missing-{dataset_name}>",
            "--asset-manifest",
            (
                str(context.asset_manifest_path)
                if context.asset_manifest_path is not None
                else "<missing-external-assets-manifest>"
            ),
            "--model-asset",
            model_name,
            "--dataset-asset",
            dataset_name,
        )
        steps.append(
            Step(
                name=f"downstream.{task}",
                families=("downstream",),
                kind="gpu-model-replay",
                description=f"Four-provider, fully pinned replacement replay for {task}",
                cwd=ROOT,
                command=command,
                environment={
                    **_base_environment(context),
                    "HF_DATASETS_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                dependencies=("noncausal.unified",),
                outputs=(context.output_root / "downstream",),
                blockers=(
                    *model_blockers,
                    *dataset_blockers,
                    *_source_blockers(downstream_sources),
                ),
                note=(
                    "This produces new authenticated evidence. The historical paper "
                    "receipt remains non-exact because its original asset revisions "
                    "were not recorded."
                ),
            )
        )
    return steps


def _mae_steps(context: Context) -> list[Step]:
    script = ROOT / "tk_fa4/fp4_fa4_fwd/eval_vit_mae_reconstruction.py"
    evaluation_sources = (
        script,
        ROOT / "tk_fa4/fp4_fa4_fwd/eval_regular_attention.py",
    )
    model, model_blockers = _asset(context, "vit_mae_model")
    images, image_blockers = _asset(context, "coco_val_100")
    asset_blockers = (*model_blockers, *image_blockers)
    if images is not None and len(images.files) != 100:
        asset_blockers = (
            *asset_blockers,
            "coco_val_100 must authenticate exactly 100 files",
        )
    expected_images_receipt = (
        ROOT / "results/fp4_fa4_reconstruction_20260805/nvmx_fast_100.json"
    )
    if images is not None and expected_images_receipt.is_file():
        expected_image_names = tuple(
            json.loads(expected_images_receipt.read_text())["images"]
        )
        observed_image_names = tuple(sorted(file.path.name for file in images.files))
        if (
            any(file.path.parent != images.root for file in images.files)
            or observed_image_names != expected_image_names
        ):
            asset_blockers = (
                *asset_blockers,
                "coco_val_100 does not match the exact 100 image names in the committed reconstruction receipt",
            )
    elif not expected_images_receipt.is_file():
        asset_blockers = (
            *asset_blockers,
            f"expected ViT-MAE image-list receipt is missing: {expected_images_receipt}",
        )
    specifications = (
        ("nvmx-fast", "tk", False, "nvmx_fast_100.json"),
        ("nvmx-accurate", "tk", True, "nvmx_accurate_100.json"),
        ("nv-nv", "hao-native", False, "hao_nvnv_100.json"),
        ("nv-nv", "hao-fp8", False, "hao_nvfp8_100.json"),
    )
    steps: list[Step] = []
    for variant, backend, anchor, filename in specifications:
        extension = (
            context.noncausal_build_root / "unified" / f"b1_s256_h16_d128_{variant}.so"
        )
        command = [
            str(context.python),
            "-B",
            str(script),
            "--model",
            str(model.root) if model else "<missing-vit-mae-model>",
            "--image-dir",
            str(images.root) if images else "<missing-coco-images>",
            "--asset-manifest",
            (
                str(context.asset_manifest_path)
                if context.asset_manifest_path is not None
                else "<missing-external-assets-manifest>"
            ),
            "--model-asset",
            "vit_mae_model",
            "--image-asset",
            "coco_val_100",
            "--samples",
            "100",
            "--seed",
            "20260805",
            "--mask-value",
            "10",
            "--extension",
            str(extension),
            "--extension-module",
            "_C_tk_hao_direct_fp4pv",
            "--attention-backend",
            backend,
            "--hao-root",
            str(ROOT / "third_party/hao_flash_attention_fp4"),
            "--output",
            str(context.output_root / "vit-mae" / filename),
        ]
        if anchor:
            command.extend(("--global-anchor-kv", "--global-anchor-samples", "32"))
        blockers = [*asset_blockers, *_source_blockers(evaluation_sources)]
        steps.append(
            Step(
                name=f"vit-mae.{backend}.{variant}",
                families=("vit-mae",),
                kind="gpu-model-replay",
                description=f"100-image ViT-MAE replay ({backend}, {variant})",
                cwd=ROOT,
                command=tuple(command),
                environment={**_base_environment(context), "TRANSFORMERS_OFFLINE": "1"},
                dependencies=("noncausal.unified",),
                outputs=(context.output_root / "vit-mae" / filename,),
                blockers=tuple(blockers),
            )
        )
    return steps


def _wan_steps(context: Context) -> list[Step]:
    build_script = ROOT / "tk_fa4/fp4_fa4_fwd/build_wan_nv_mx_bundle.py"
    eval_script = ROOT / "tk_fa4/fp4_fa4_fwd/eval_wan_video.py"
    eval_sources = (
        eval_script,
        ROOT / "tk_fa4/fp4_fa4_fwd/eval_regular_attention.py",
    )
    hao_root = ROOT / "third_party/hao_flash_attention_fp4"
    environment = {**_base_environment(context), "TRANSFORMERS_OFFLINE": "1"}
    build_sources = _forward_build_sources(build_script.parent, ROOT)
    steps: list[Step] = []
    for model_tag, asset_name, model_identifier in (
        (
            "1.3b",
            "wan_1_3b_model",
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ),
        ("14b", "wan_14b_model", "Wan-AI/Wan2.1-T2V-14B-Diffusers"),
    ):
        bundle = context.output_root / "wan" / f"bundle-{model_tag}"
        build_name = f"wan.build.{model_tag}"
        steps.append(
            Step(
                name=build_name,
                families=("wan",),
                kind="gpu-kernel-build",
                description=f"Build the calibrated {model_tag} NV/MX policy bundle",
                cwd=ROOT,
                command=(
                    str(context.python),
                    "-B",
                    str(build_script),
                    "--model",
                    model_tag,
                    "--output-dir",
                    str(bundle),
                    "--policies",
                    "fast,accurate",
                    "--gpu",
                    "B200",
                    "--num-sm",
                    "152",
                ),
                environment=environment,
                outputs=(bundle / "manifest.json",),
                blockers=_source_blockers((build_script, *build_sources)),
            )
        )
        model, model_blockers = _asset(context, asset_name)
        replay_blockers = [*model_blockers]
        if model is not None and model.identifier != model_identifier:
            replay_blockers.append(
                f"{asset_name}.identifier={model.identifier!r} does not match "
                f"{model_identifier!r}"
            )
        for policy in ("fast", "accurate"):
            for diffusion_steps in (1, 4, 20):
                output_type = (
                    "np" if model_tag == "1.3b" and diffusion_steps == 20 else "latent"
                )
                output = (
                    context.output_root
                    / "wan"
                    / f"wan-{model_tag}-{policy}-step{diffusion_steps}.json"
                )
                command = (
                    str(context.python),
                    "-B",
                    str(eval_script),
                    "--model",
                    model_identifier,
                    "--model-path",
                    str(model.root) if model else f"<missing-{asset_name}>",
                    "--asset-manifest",
                    (
                        str(context.asset_manifest_path)
                        if context.asset_manifest_path is not None
                        else "<missing-external-assets-manifest>"
                    ),
                    "--model-asset",
                    asset_name,
                    "--providers",
                    "hao-bf16,tk",
                    "--hao-root",
                    str(hao_root),
                    "--policy-manifest",
                    str(bundle / "manifest.json"),
                    "--policy",
                    policy,
                    "--height",
                    "512",
                    "--width",
                    "768",
                    "--num-frames",
                    "17",
                    "--steps",
                    str(diffusion_steps),
                    "--guidance-scale",
                    "5.0",
                    "--seed",
                    "20260805",
                    "--output-type",
                    output_type,
                    "--local-files-only",
                    "--device",
                    "cuda:0",
                    "--output",
                    str(output),
                )
                steps.append(
                    Step(
                        name=f"wan.replay.{model_tag}.{policy}.step{diffusion_steps}",
                        families=("wan",),
                        kind="gpu-model-replay",
                        description=f"Wan {model_tag} {policy} paired HAO-BF16/TK replay",
                        cwd=ROOT,
                        command=command,
                        environment=environment,
                        dependencies=(build_name,),
                        outputs=(output,),
                        blockers=(
                            *replay_blockers,
                            *_source_blockers(eval_sources),
                        ),
                        note=(
                            "The logical model identifier is checked by the policy "
                            "manifest; model bytes come only from the separately "
                            "authenticated local snapshot."
                        ),
                    )
                )
        steps.append(
            Step(
                name=f"wan.hao-controls.{model_tag}",
                families=("wan",),
                kind="blocked-historical-measurement",
                description=f"Wan {model_tag} HAO NV/NV and NV/FP8 controls",
                cwd=ROOT,
                command=None,
                environment=environment,
                blockers=(
                    "the exact Wan-shape NV/NV extension build invocation and authenticated binary identity were not preserved",
                ),
                note="Do not substitute the NV/MX policy bundle for HAO's required NV/NV topology extension.",
            )
        )
    steps.append(
        Step(
            name="wan.affine-route-table",
            families=("wan",),
            kind="blocked-historical-measurement",
            description="Wan affine calibration and held-out route table",
            cwd=ROOT,
            command=None,
            environment=environment,
            blockers=(
                "some original Wan table inputs and the immutable calibration split are absent",
            ),
        )
    )
    return steps


def _b300_steps(context: Context) -> list[Step]:
    script = ROOT / "results/fp4_fa4_b300_tuning_20260802/build_summary.py"
    return [
        Step(
            name="b300.aggregate.from-summary",
            families=("b300-aggregate",),
            kind="offline-render",
            description="Regenerate B300 tables from the committed validated aggregate",
            cwd=ROOT,
            command=(str(context.python), "-B", str(script), "--from-summary"),
            environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            outputs=(ROOT / "results/fp4_fa4_b300_tuning_20260802/tables",),
            blockers=_source_blockers(
                (script, ROOT / "results/fp4_fa4_b300_tuning_20260802/summary.json")
            ),
            note="This is aggregate reconstruction, not a new B300 measurement.",
        ),
        Step(
            name="b300.aggregate.from-raw",
            families=("b300-aggregate",),
            kind="blocked-historical-measurement",
            description="Recompute the B300 aggregate from raw cluster captures",
            cwd=ROOT,
            command=None,
            environment={},
            blockers=("the complete checksummed B300 raw-capture archive is absent",),
        ),
    ]


def _boundary_steps(context: Context) -> list[Step]:
    script = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_v509_report_boundaries.py"
    manifest, blockers = _artifact(context, "nvfp4_qk_fp8_pv", 1, training=False)
    command: tuple[str, ...] | None = None
    environment = _base_environment(context)
    if manifest is not None and manifest.is_low_precision:
        forward = manifest.forward
        publisher = manifest.projection_publisher
        backward = manifest.v509_backward
        assert forward is not None and publisher is not None and backward is not None
        runtime_environment, runtime_blockers = _runtime_environment_and_blockers(
            manifest
        )
        blockers = (*blockers, *runtime_blockers)
        environment.update(runtime_environment)
        environment["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = str(publisher.path)
        command = (
            str(context.python),
            "-B",
            "-m",
            "tk_fa4.lowp_fa4_bwd.benchmark_v509_report_boundaries",
            "--projection-extension",
            str(publisher.path),
            "--projection-sha256",
            publisher.sha256,
            "--projection-bytes",
            str(publisher.bytes),
            "--forward-extension",
            str(forward.path),
            "--forward-module",
            forward.module,
            "--forward-sha256",
            forward.sha256,
            "--forward-bytes",
            str(forward.bytes),
            "--backward-extension",
            str(backward.path),
            "--backward-module",
            backward.module,
            "--backward-sha256",
            backward.sha256,
            "--backward-bytes",
            str(backward.bytes),
            "--warmups",
            "20",
            "--samples",
            "101",
            "--seed",
            "20260901",
            "--output",
            str(context.output_root / "causal" / "v509-report-boundaries.json"),
        )
    return [
        Step(
            name="causal.v509-report-boundaries",
            families=("causal-backward", "projection-fwd-bwd"),
            kind="gpu-measurement",
            description="One capture containing isolated backward and projection-inclusive F+B boundaries",
            cwd=ROOT,
            command=command,
            environment=environment,
            outputs=(context.output_root / "causal" / "v509-report-boundaries.json",),
            blockers=(*blockers, *_source_blockers((script,))),
            note="Both paper boundaries share this capture; do not run it twice.",
        )
    ]


def _e2e_command(
    context: Context, manifest: Any, output: Path
) -> tuple[tuple[str, ...], dict[str, str]]:
    forward = manifest.forward
    publisher = manifest.projection_publisher
    backward = manifest.v509_backward
    assert forward is not None and publisher is not None and backward is not None
    runtime_environment, _ = _runtime_environment_and_blockers(manifest)
    environment = {
        **_base_environment(context),
        **runtime_environment,
        "TK_FA4_LOWP_BWD_EXTENSION_SOURCE": str(publisher.path),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    projection_format = manifest.learned_projection_format
    if projection_format not in ("e4m3", "nvfp4"):
        raise PlanError(
            "low-precision E2E manifest requires E4M3 or NVFP4 learned projections"
        )
    projection_arguments = (
        "--qkv-projection-format",
        projection_format,
        "--output-projection-format",
        projection_format,
    )
    if projection_format == "nvfp4":
        projection_arguments += ("--experimental-native-nvfp4-projection-out",)
    command = (
        str(context.python),
        "-B",
        "-m",
        "tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e",
        "--model-preset",
        "llama3.1-8b",
        "--batch",
        str(manifest.batch),
        "--warmups",
        "10",
        "--samples",
        "21",
        "--bf16-attention-control",
        "packed_qkv_single_linear",
        "--seed",
        "20260903",
        "--learning-rate",
        "0.0001",
        "--compile-loss",
        *projection_arguments,
        "--per-block-qk-scales",
        "--projection-weight-scaling",
        "2d",
        "--v-mxfp4-scaling",
        "1d",
        "--projection-dgrad",
        "nvfp4",
        "--native-tk-d128-backward-extension",
        str(backward.path),
        "--native-tk-d128-backward-module",
        backward.module,
        "--native-tk-d128-backward-sha256",
        backward.sha256,
        "--native-tk-d128-backward-bytes",
        str(backward.bytes),
        "--native-tk-d128-native-score-backward",
        "--native-tk-d128-v509-e5m2-dout-backward",
        "--forward-extension",
        str(forward.path),
        "--forward-module",
        forward.module,
        "--output",
        str(output),
    )
    return command, environment


def _e2e_steps(context: Context) -> list[Step]:
    source = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
    steps: list[Step] = []
    for route in LOWP_ROUTES:
        for batch in BATCHES:
            manifest, blockers = _artifact(context, route, batch, training=False)
            blockers = (
                *blockers,
                "the published B1/B2/B4 timing protocol binds GPU0 CPU and memory to NUMA0, but no authenticated portable NUMA launcher is recorded",
            )
            output = context.output_root / "llama8b-e2e" / f"{route}-b{batch}.json"
            command = None
            environment = _base_environment(context)
            if manifest is not None:
                _, runtime_blockers = _runtime_environment_and_blockers(manifest)
                blockers = (*blockers, *runtime_blockers)
                command, environment = _e2e_command(context, manifest, output)
            steps.append(
                Step(
                    name=f"llama8b-e2e.{route}.b{batch}",
                    families=("llama8b-e2e",),
                    kind="gpu-measurement",
                    description=f"Matched BF16/low-precision 8B full-update timing, {route}, B{batch}",
                    cwd=ROOT,
                    command=command,
                    environment=environment,
                    outputs=(output,),
                    blockers=(*blockers, *_source_blockers((source,))),
                    note="Synthetic-token performance/correctness bracket; not convergence evidence.",
                )
            )
    return steps


def _d64_steps(context: Context) -> list[Step]:
    """Describe the retained and replacement D64/1.2B evidence graph."""
    fp8, fp8_blockers = _artifact(
        context,
        "e4m3_proj_nvfp4_qk_fp8_pv",
        D64_BATCH,
        training=True,
    )
    mx, mx_blockers = _artifact(
        context,
        "e4m3_proj_nvfp4_qk_mxfp4_pv",
        D64_BATCH,
        training=True,
    )
    bf16, bf16_blockers = _artifact(
        context,
        "bf16_fa4",
        D64_BATCH,
        training=True,
    )

    expected_shape = (16, 4096, 32, 8, 64)

    def profile_blockers(manifest: Any | None) -> tuple[str, ...]:
        if manifest is None:
            return ()
        blockers: list[str] = []
        if manifest.profile != "llama1p2b-d64-b16":
            blockers.append(
                f"D64 artifact profile is {manifest.profile!r}, not "
                "'llama1p2b-d64-b16'"
            )
        observed_shape = (
            manifest.batch,
            manifest.sequence,
            manifest.q_heads,
            manifest.kv_heads,
            manifest.head_dim,
        )
        if observed_shape != expected_shape:
            blockers.append(
                f"D64 artifact shape is {observed_shape!r}, not {expected_shape!r}"
            )
        return tuple(blockers)

    fp8_blockers = (*fp8_blockers, *profile_blockers(fp8))
    mx_blockers = (*mx_blockers, *profile_blockers(mx))
    bf16_blockers = (*bf16_blockers, *profile_blockers(bf16))
    if fp8 is not None and mx is not None:
        shared_blockers: list[str] = []
        for name in ("projection_publisher", "v416_backward"):
            fp8_identity = getattr(fp8, name)
            mx_identity = getattr(mx, name)
            if fp8_identity is None or mx_identity is None:
                shared_blockers.append(f"D64 low-precision manifests omit {name}")
                continue
            if (fp8_identity.sha256, fp8_identity.bytes) != (
                mx_identity.sha256,
                mx_identity.bytes,
            ):
                shared_blockers.append(
                    f"D64 FP8 and MX manifests do not share one {name} image"
                )
        fp8_blockers = (*fp8_blockers, *shared_blockers)
        mx_blockers = (*mx_blockers, *shared_blockers)
    steps: list[Step] = []

    forward_script = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_b16_forward_factorial.py"
    forward_command: tuple[str, ...] | None = None
    forward_environment = _base_environment(context)
    forward_blockers = [*fp8_blockers, *mx_blockers]
    if fp8 is not None and mx is not None:
        if fp8.projection_publisher is None or mx.projection_publisher is None:
            forward_blockers.append(
                "D64 forward manifests omit the projection publisher"
            )
        elif (
            fp8.projection_publisher.sha256,
            fp8.projection_publisher.bytes,
        ) != (
            mx.projection_publisher.sha256,
            mx.projection_publisher.bytes,
        ):
            forward_blockers.append(
                "D64 FP8 and MX manifests do not share one projection publisher"
            )
        if fp8.forward is None or mx.forward is None:
            forward_blockers.append("D64 low-precision manifest omits a forward image")
        if not forward_blockers:
            assert fp8.forward is not None
            assert mx.forward is not None
            assert fp8.projection_publisher is not None
            runtime_environment, runtime_blockers = _runtime_environment_and_blockers(
                fp8
            )
            forward_environment.update(runtime_environment)
            forward_blockers.extend(runtime_blockers)
            forward_environment["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = str(
                fp8.projection_publisher.path
            )
            forward_command = (
                str(context.python),
                "-B",
                "-m",
                "tk_fa4.lowp_fa4_bwd.benchmark_b16_forward_factorial",
                "--mx-extension",
                str(mx.forward.path),
                "--mx-module",
                mx.forward.module or "",
                "--fp8-extension",
                str(fp8.forward.path),
                "--fp8-module",
                fp8.forward.module or "",
                "--projection-extension",
                str(fp8.projection_publisher.path),
                "--projection-module",
                fp8.projection_publisher.module or "",
                "--seed",
                "20260826",
                "--warmups",
                "4",
                "--samples",
                "40",
                "--minimum-bf16-output-cosine",
                "0.95",
                "--maximum-bf16-output-relative-l2",
                "0.35",
                "--output",
                str(context.output_root / "d64/isolated-forward.json"),
            )
    steps.append(
        Step(
            name="d64.isolated-forward",
            families=("llama1p2b-d64",),
            kind="gpu-measurement",
            description=(
                "Matched B16/S4096/Hq32/Hkv8/D64 causal BF16, FP8-P/V, "
                "and MXFP4-P/V forward boundary"
            ),
            cwd=ROOT,
            command=forward_command,
            environment=forward_environment,
            outputs=(context.output_root / "d64/isolated-forward.json",),
            blockers=(*forward_blockers, *_source_blockers((forward_script,))),
            note="New measurement; it does not recreate an absent historical raw capture.",
        )
    )

    v416_source = ROOT / "tk_fa4/lowp_fa4_bwd/native_tk_d64_backward.py"
    v416_artifact_blockers = [*fp8_blockers]
    if fp8 is not None and fp8.v416_backward is None:
        v416_artifact_blockers.append("D64 manifest omits native v416 backward")
    steps.append(
        Step(
            name="d64.isolated-v416-backward",
            families=("llama1p2b-d64",),
            kind="blocked-historical-measurement",
            description="Reacquire the isolated native-v416 versus BF16 backward bracket",
            cwd=ROOT,
            command=None,
            environment=_base_environment(context),
            blockers=(
                *v416_artifact_blockers,
                "the exact isolated v416 acquisition driver was not retained",
                "the 220876-byte CuTe cd57 control source is identified by SHA256 but absent",
                *_source_blockers((v416_source,)),
            ),
            note=(
                "The committed v416 receipt remains valid historical evidence; do not "
                "substitute the current FlashAttention control and call it an exact replay."
            ),
        )
    )
    steps.append(
        Step(
            name="d64.combined-forward-backward",
            families=("llama1p2b-d64",),
            kind="blocked-historical-measurement",
            description="Reacquire one isolated attention forward-plus-backward boundary",
            cwd=ROOT,
            command=None,
            environment=_base_environment(context),
            blockers=(
                *v416_artifact_blockers,
                "no retained driver measures the D64 attention-only F+B boundary with v416",
            ),
            note=(
                "The saturated model harness below records model forward and backward "
                "phases, but those are not an attention-only F+B measurement."
            ),
        )
    )

    saturated_script = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_saturated.py"
    initial_checkpoint = context.output_root / "d64/saturated/initial.pt"
    reference_samples = context.output_root / "d64/saturated/bf16_fa4.samples.pt"
    route_labels = {
        "bf16_fa4": "bf16_packed",
        "e4m3_proj_nvfp4_qk_fp8_pv": "fp8",
        "e4m3_proj_nvfp4_qk_mxfp4_pv": "mx",
    }
    manifests = {
        "bf16_fa4": (bf16, bf16_blockers),
        "e4m3_proj_nvfp4_qk_fp8_pv": (fp8, fp8_blockers),
        "e4m3_proj_nvfp4_qk_mxfp4_pv": (mx, mx_blockers),
    }
    for route in D64_ROUTES:
        manifest, blockers = manifests[route]
        command = None
        environment = _base_environment(context)
        output = context.output_root / "d64/saturated" / f"{route}.json"
        samples = context.output_root / "d64/saturated" / f"{route}.samples.pt"
        if manifest is not None:
            arguments = [
                str(context.python),
                "-B",
                "-m",
                "tk_fa4.lowp_fa4_bwd.benchmark_llama12b_saturated",
                "--model-preset",
                "llama3.2-1b",
                "--route",
                route_labels[route],
                "--batch",
                "16",
                "--warmups",
                "3",
                "--updates",
                "20",
                "--seed",
                "20260825",
                "--learning-rate",
                "0.00048828125",
                "--max-grad-norm",
                "1.0",
                "--tokens",
                "synthetic",
                "--trial-label",
                f"release-d64-{route}",
            ]
            if route == "bf16_fa4":
                arguments.extend(("--save-initial-checkpoint", str(initial_checkpoint)))
            else:
                assert manifest.forward is not None
                assert manifest.projection_publisher is not None
                assert manifest.v416_backward is not None
                runtime_environment, runtime_blockers = (
                    _runtime_environment_and_blockers(manifest)
                )
                blockers = (*blockers, *runtime_blockers)
                environment.update(runtime_environment)
                environment["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = str(
                    manifest.projection_publisher.path
                )
                arguments.extend(
                    (
                        "--initial-checkpoint",
                        str(initial_checkpoint),
                        "--reference-samples",
                        str(reference_samples),
                        "--forward-extension",
                        str(manifest.forward.path),
                        "--forward-module",
                        manifest.forward.module or "",
                        "--forward-sha256",
                        manifest.forward.sha256,
                        "--forward-bytes",
                        str(manifest.forward.bytes),
                        "--projection-extension",
                        str(manifest.projection_publisher.path),
                        "--projection-sha256",
                        manifest.projection_publisher.sha256,
                        "--projection-bytes",
                        str(manifest.projection_publisher.bytes),
                        "--native-tk-d64-backward-extension",
                        str(manifest.v416_backward.path),
                        "--native-tk-d64-backward-module",
                        manifest.v416_backward.module or "",
                        "--native-tk-d64-backward-sha256",
                        manifest.v416_backward.sha256,
                        "--native-tk-d64-backward-bytes",
                        str(manifest.v416_backward.bytes),
                    )
                )
            arguments.extend(
                ("--samples-output", str(samples), "--output", str(output))
            )
            command = tuple(arguments)
        steps.append(
            Step(
                name=f"d64.saturated.{route}",
                families=("llama1p2b-d64",),
                kind="gpu-measurement",
                description=f"Saturated single-GB200 1.235B full-update bracket for {route}",
                cwd=ROOT,
                command=command,
                environment=environment,
                dependencies=(
                    () if route == "bf16_fa4" else ("d64.saturated.bf16_fa4",)
                ),
                outputs=(output, samples),
                blockers=(*blockers, *_source_blockers((saturated_script,))),
                note=(
                    "Twenty synthetic updates test performance and short numerical "
                    "behavior; they are not convergence evidence."
                ),
            )
        )

    steps.append(
        Step(
            name="d64.short-real-data-numerics",
            families=("llama1p2b-d64",),
            kind="blocked-historical-measurement",
            description="Exact 256-update Dolma3 Longmino numerical replay",
            cwd=ROOT,
            command=None,
            environment=_base_environment(context),
            blockers=(
                "the exact 512-row Dolma3 JSONL input is not redistributed",
                "the exact Llama tokenizer bytes are not redistributed",
                "the 220876-byte CuTe cd57 control source is identified by SHA256 but absent",
            ),
            note=(
                "The committed receipt and cd59dda source snapshot remain historical "
                "evidence. Supplying different public text creates a new experiment."
            ),
        )
    )

    renderer = ROOT / "tools/render_fa4_training_config.py"
    verifier = ROOT / "tools/verify_fa4_training_config.py"
    dataset_tool = ROOT / "tools/fa4_dataset_manifest.py"
    tokenizer, tokenizer_blockers = _asset(context, "llama3_8b_assets")
    dataset, dataset_blockers = _asset(context, "slimpajama_dataset")
    data_manifest = context.output_root / "d64/slimpajama.fa4-dataset-manifest.json"
    data_command = None
    if dataset is not None:
        data_command = (
            str(context.python),
            "-B",
            str(dataset_tool),
            "create",
            "--root",
            str(dataset.root),
            "--output",
            str(data_manifest),
        )
    steps.append(
        Step(
            name="d64.dataset-manifest",
            families=("llama1p2b-d64",),
            kind="data-manifest",
            description="Authenticate the complete public SlimPajama replacement snapshot",
            cwd=ROOT,
            command=data_command,
            environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            outputs=(data_manifest,),
            blockers=(*dataset_blockers, *_source_blockers((dataset_tool,))),
            note="The historical SlimPajama MDS shard order is unavailable.",
        )
    )

    asset_blockers = (*tokenizer_blockers, *dataset_blockers)
    for route in D64_ROUTES:
        manifest, manifest_blockers = manifests[route]
        config = context.output_root / "d64/distributed" / f"{route}.toml"
        render_name = f"d64.distributed.render.{route}"
        render_command = None
        if manifest is not None:
            render_command = (
                str(context.python),
                "-B",
                str(renderer),
                "--profile",
                "llama1p2b-d64-50b",
                "--artifact-manifest",
                str(manifest.path),
                "--output",
                str(config),
                "--dump-folder",
                str(context.output_root / "d64/distributed" / route),
                "--hf-assets-path",
                str(tokenizer.root) if tokenizer else "<missing-llama-assets>",
                "--dataset-path",
                str(dataset.root) if dataset else "<missing-slimpajama>",
                "--dataset-manifest",
                str(data_manifest),
                "--world-size",
                "16",
                "--global-batch-size",
                "256",
                "--steps",
                "47684",
                "--warmup-steps",
                "954",
                "--checkpoint-interval",
                "954",
                "--validation-frequency",
                "262",
                "--validation-steps",
                "16",
                "--seed",
                "42",
            )
        steps.append(
            Step(
                name=render_name,
                families=("llama1p2b-d64",),
                kind="config-render",
                description=f"Render the manifest-bound 50B-token D64 recipe for {route}",
                cwd=ROOT,
                command=render_command,
                environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                dependencies=("d64.dataset-manifest",),
                outputs=(config, Path(str(config) + ".receipt.json")),
                blockers=(
                    *manifest_blockers,
                    *asset_blockers,
                    *_source_blockers((renderer,)),
                ),
                note=(
                    "This new portable recipe uses BF16-SR AdamW. The historical "
                    "1.2B templates used ordinary AdamW."
                ),
            )
        )
        verify_name = f"d64.distributed.verify.{route}"
        steps.append(
            Step(
                name=verify_name,
                families=("llama1p2b-d64",),
                kind="config-verify",
                description=f"Rehash the complete DDP16 launch boundary for {route}",
                cwd=ROOT,
                command=(
                    str(context.python),
                    "-B",
                    str(verifier),
                    "--config",
                    str(config),
                    "--world-size",
                    "16",
                ),
                environment={
                    **OPTIMIZER_ENVIRONMENT,
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                dependencies=(render_name,),
                blockers=(
                    *manifest_blockers,
                    *asset_blockers,
                    *_source_blockers((verifier,)),
                ),
            )
        )
        launch_blockers: list[str] = []
        launch_command = None
        launch_environment = {**OPTIMIZER_ENVIRONMENT}
        if context.asset_manifest_path is not None:
            launch_environment["FA4_EXTERNAL_ASSETS_MANIFEST"] = str(
                context.asset_manifest_path
            )
        if context.launcher_error:
            launch_blockers.append(context.launcher_error)
        if context.launcher is None:
            launch_blockers.append(
                f"missing explicit {LAUNCHER_SCHEMA}; multi-node rendezvous flags are site-owned"
            )
        elif context.launcher.world_size != 16:
            launch_blockers.append(
                "the supplied launcher declares world_size "
                f"{context.launcher.world_size}, but the D64 recipe requires 16"
            )
        else:
            launch_environment.update(context.launcher.environment)
            launch_environment["FA4_LAUNCHER_SOURCE_REVISION"] = (
                context.launcher.source_revision
            )
            launch_command = (
                str(context.launcher.executable.path),
                *context.launcher.argv_prefix,
                "-m",
                "torchtitan.experiments.fa4.train",
                "--job.config-file",
                str(config),
            )
        steps.append(
            Step(
                name=f"d64.distributed.launch.{route}",
                families=("llama1p2b-d64",),
                kind="distributed-gpu-training",
                description=f"Launch the matched DDP16 50B-token arm for {route}",
                cwd=ROOT,
                command=launch_command,
                environment=launch_environment,
                dependencies=(verify_name,),
                outputs=(context.output_root / "d64/distributed" / route,),
                blockers=tuple(launch_blockers),
                note="The planner prints but never submits this site-owned launch.",
            )
        )

    smoke_manifest = fp8
    for phase, training_steps, dependency in (
        ("save", 1, "d64.dataset-manifest"),
        ("fresh-resume", 2, "d64.ddp16-smoke.launch.save"),
    ):
        config = context.output_root / "d64/ddp16-smoke" / f"{phase}.toml"
        render_name = f"d64.ddp16-smoke.render.{phase}"
        render_command = None
        if smoke_manifest is not None:
            render_command = (
                str(context.python),
                "-B",
                str(renderer),
                "--profile",
                "llama1p2b-d64-50b",
                "--artifact-manifest",
                str(smoke_manifest.path),
                "--output",
                str(config),
                "--dump-folder",
                str(context.output_root / "d64/ddp16-smoke/state"),
                "--hf-assets-path",
                str(tokenizer.root) if tokenizer else "<missing-llama-assets>",
                "--dataset-path",
                str(dataset.root) if dataset else "<missing-slimpajama>",
                "--dataset-manifest",
                str(data_manifest),
                "--world-size",
                "16",
                "--global-batch-size",
                "256",
                "--steps",
                str(training_steps),
                "--warmup-steps",
                "1",
                "--checkpoint-interval",
                "1",
                "--validation-frequency",
                "1",
                "--no-validation",
                "--seed",
                "42",
            )
        steps.append(
            Step(
                name=render_name,
                families=("llama1p2b-d64",),
                kind="config-render",
                description=f"Render DDP16 checkpoint {phase} smoke config",
                cwd=ROOT,
                command=render_command,
                environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                dependencies=(dependency,),
                outputs=(config, Path(str(config) + ".receipt.json")),
                blockers=(
                    *fp8_blockers,
                    *asset_blockers,
                    *_source_blockers((renderer,)),
                ),
            )
        )
        verify_name = f"d64.ddp16-smoke.verify.{phase}"
        steps.append(
            Step(
                name=verify_name,
                families=("llama1p2b-d64",),
                kind="config-verify",
                description=f"Verify DDP16 checkpoint {phase} smoke config",
                cwd=ROOT,
                command=(
                    str(context.python),
                    "-B",
                    str(verifier),
                    "--config",
                    str(config),
                    "--world-size",
                    "16",
                ),
                environment={**OPTIMIZER_ENVIRONMENT},
                dependencies=(render_name,),
                blockers=(
                    *fp8_blockers,
                    *asset_blockers,
                    *_source_blockers((verifier,)),
                ),
            )
        )
        smoke_launch_blockers: list[str] = []
        smoke_launch = None
        smoke_environment = {**OPTIMIZER_ENVIRONMENT}
        if context.asset_manifest_path is not None:
            smoke_environment["FA4_EXTERNAL_ASSETS_MANIFEST"] = str(
                context.asset_manifest_path
            )
        if context.launcher_error:
            smoke_launch_blockers.append(context.launcher_error)
        if context.launcher is None:
            smoke_launch_blockers.append(
                f"missing explicit {LAUNCHER_SCHEMA}; multi-node rendezvous flags are site-owned"
            )
        elif context.launcher.world_size != 16:
            smoke_launch_blockers.append(
                "DDP16 smoke requires a world-size-16 launcher"
            )
        else:
            smoke_environment.update(context.launcher.environment)
            smoke_environment["FA4_LAUNCHER_SOURCE_REVISION"] = (
                context.launcher.source_revision
            )
            smoke_launch = (
                str(context.launcher.executable.path),
                *context.launcher.argv_prefix,
                "-m",
                "torchtitan.experiments.fa4.train",
                "--job.config-file",
                str(config),
            )
        steps.append(
            Step(
                name=f"d64.ddp16-smoke.launch.{phase}",
                families=("llama1p2b-d64",),
                kind="distributed-gpu-training",
                description=f"Run DDP16 checkpoint {phase} in a fresh process",
                cwd=ROOT,
                command=smoke_launch,
                environment=smoke_environment,
                dependencies=(verify_name,),
                outputs=(context.output_root / "d64/ddp16-smoke/state",),
                blockers=tuple(smoke_launch_blockers),
                note=(
                    "The second process must load the update-1 distributed checkpoint "
                    "and execute update 2 without changing artifact or data receipts."
                ),
            )
        )
    return steps


def _training_steps(context: Context) -> list[Step]:
    renderer = ROOT / "tools/render_fa4_training_config.py"
    config_verifier = ROOT / "tools/verify_fa4_training_config.py"
    dataset_manifest_tool = ROOT / "tools/fa4_dataset_manifest.py"
    tokenizer, tokenizer_blockers = _asset(context, "llama3_8b_assets")
    dataset, dataset_blockers = _asset(context, "slimpajama_dataset")
    if tokenizer is not None:
        tokenizer_names = {
            file.path.relative_to(tokenizer.root).as_posix() for file in tokenizer.files
        }
        if "tokenizer_config.json" not in tokenizer_names:
            tokenizer_blockers = (
                *tokenizer_blockers,
                "llama3_8b_assets omits tokenizer_config.json",
            )
        has_tokenizer_json = "tokenizer.json" in tokenizer_names
        vocabulary_names = tokenizer_names.intersection({"vocab.json", "vocab.txt"})
        if not has_tokenizer_json and len(vocabulary_names) != 1:
            tokenizer_blockers = (
                *tokenizer_blockers,
                "llama3_8b_assets requires tokenizer.json or exactly one of vocab.json/vocab.txt",
            )
    asset_blockers = (*tokenizer_blockers, *dataset_blockers)
    dataset_manifest_name = "distributed.dataset-manifest"
    dataset_manifest = (
        context.output_root / "distributed/slimpajama.fa4-dataset-manifest.json"
    )
    dataset_manifest_command = None
    if dataset is not None:
        dataset_manifest_command = (
            str(context.python),
            "-B",
            str(dataset_manifest_tool),
            "create",
            "--root",
            str(dataset.root),
            "--output",
            str(dataset_manifest),
        )
    steps: list[Step] = [
        Step(
            name=dataset_manifest_name,
            families=("distributed-training", "mxfp4-divergence"),
            kind="data-manifest",
            description="Authenticate the complete local SlimPajama snapshot",
            cwd=ROOT,
            command=dataset_manifest_command,
            environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            outputs=(dataset_manifest,),
            blockers=(
                *dataset_blockers,
                *_source_blockers((dataset_manifest_tool,)),
            ),
        )
    ]
    for route in ROUTES:
        manifest, manifest_blockers = _artifact(context, route, 4, training=True)
        config = context.output_root / "distributed" / f"{route}-b4.toml"
        render_name = f"distributed.render.{route}"
        command = None
        if manifest is not None:
            command = (
                str(context.python),
                "-B",
                str(renderer),
                "--artifact-manifest",
                str(manifest.path),
                "--output",
                str(config),
                "--dump-folder",
                str(context.output_root / "distributed" / route),
                "--hf-assets-path",
                str(tokenizer.root) if tokenizer else "<missing-llama-assets>",
                "--dataset-path",
                str(dataset.root) if dataset else "<missing-slimpajama>",
                "--dataset-manifest",
                str(dataset_manifest),
                "--world-size",
                "64",
                "--global-batch-size",
                "1024",
                "--steps",
                "23842",
                "--warmup-steps",
                "2000",
                "--checkpoint-interval",
                "239",
                "--validation-frequency",
                "298",
                "--validation-steps",
                "16",
                "--seed",
                "42",
            )
        steps.append(
            Step(
                name=render_name,
                families=("distributed-training",)
                + (("mxfp4-divergence",) if route.endswith("mxfp4_pv") else ()),
                kind="config-render",
                description=f"Render authenticated B4/W64 TorchTitan config for {route}",
                cwd=ROOT,
                command=command,
                environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                dependencies=(dataset_manifest_name,),
                outputs=(config, Path(str(config) + ".receipt.json")),
                blockers=(
                    *manifest_blockers,
                    *asset_blockers,
                    *_source_blockers((renderer,)),
                ),
            )
        )
        verify_name = f"distributed.verify.{route}"
        steps.append(
            Step(
                name=verify_name,
                families=("distributed-training",)
                + (("mxfp4-divergence",) if route.endswith("mxfp4_pv") else ()),
                kind="config-verify",
                description=f"Rehash the complete launch boundary for {route}",
                cwd=ROOT,
                command=(
                    str(context.python),
                    "-B",
                    str(config_verifier),
                    "--config",
                    str(config),
                    "--world-size",
                    "64",
                ),
                environment={
                    **OPTIMIZER_ENVIRONMENT,
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                dependencies=(render_name,),
                blockers=(
                    *manifest_blockers,
                    *asset_blockers,
                    *_source_blockers((config_verifier,)),
                ),
            )
        )
        launch_blockers: list[str] = []
        launch_command = None
        launch_environment = {**OPTIMIZER_ENVIRONMENT}
        if context.asset_manifest_path is not None:
            launch_environment["FA4_EXTERNAL_ASSETS_MANIFEST"] = str(
                context.asset_manifest_path
            )
        if context.launcher_error:
            launch_blockers.append(context.launcher_error)
        if context.launcher is None:
            launch_blockers.append(
                f"missing explicit {LAUNCHER_SCHEMA}; multi-node rendezvous flags are site-owned"
            )
        elif context.launcher.world_size != 64:
            launch_blockers.append(
                "the supplied launcher declares world_size "
                f"{context.launcher.world_size}, but the 8B recipe requires 64"
            )
        else:
            launch_environment.update(context.launcher.environment)
            launch_environment["FA4_LAUNCHER_SOURCE_REVISION"] = (
                context.launcher.source_revision
            )
            launch_command = (
                str(context.launcher.executable.path),
                *context.launcher.argv_prefix,
                "-m",
                "torchtitan.experiments.fa4.train",
                "--job.config-file",
                str(config),
            )
        steps.append(
            Step(
                name=f"distributed.launch.{route}",
                families=("distributed-training",)
                + (("mxfp4-divergence",) if route.endswith("mxfp4_pv") else ()),
                kind="distributed-gpu-training",
                description=f"Launch the matched 64-GPU training arm for {route}",
                cwd=ROOT,
                command=launch_command,
                environment=launch_environment,
                dependencies=(verify_name,),
                outputs=(context.output_root / "distributed" / route,),
                blockers=tuple(launch_blockers),
                note="The planner never submits this command; inspect it and use the site's authorized launch path.",
            )
        )
    return steps


def build_steps(context: Context) -> list[Step]:
    return [
        *_noncausal_steps(context),
        *_downstream_steps(context),
        *_mae_steps(context),
        *_wan_steps(context),
        *_b300_steps(context),
        *_boundary_steps(context),
        *_e2e_steps(context),
        *_d64_steps(context),
        *_training_steps(context),
    ]


def _selected_steps(steps: Sequence[Step], families: Sequence[str]) -> list[Step]:
    selected = set(families)
    initial = [step for step in steps if selected.intersection(step.families)]
    by_name = {step.name: step for step in steps}
    names = {step.name for step in initial}
    pending = list(initial)
    while pending:
        step = pending.pop()
        for dependency in step.dependencies:
            if dependency not in names and dependency in by_name:
                names.add(dependency)
                pending.append(by_name[dependency])
    selected_steps = [step for step in steps if step.name in names]
    selected_by_name = {step.name: step for step in selected_steps}
    resolved: dict[str, Step] = {}

    def resolve(step: Step, visiting: tuple[str, ...] = ()) -> Step:
        if step.name in resolved:
            return resolved[step.name]
        if step.name in visiting:
            raise PlanError(f"dependency cycle: {' -> '.join((*visiting, step.name))}")
        dependency_blockers: list[str] = []
        for dependency_name in step.dependencies:
            dependency = selected_by_name.get(dependency_name)
            if dependency is None:
                dependency_blockers.append(f"missing dependency node {dependency_name}")
                continue
            checked = resolve(dependency, (*visiting, step.name))
            if not checked.runnable:
                dependency_blockers.append(f"dependency {dependency_name} is blocked")
        result = replace(
            step,
            blockers=tuple(dict.fromkeys((*step.blockers, *dependency_blockers))),
        )
        resolved[step.name] = result
        return result

    return [resolve(step) for step in selected_steps]


def _render_shell(steps: Sequence[Step]) -> str:
    lines = [
        "# Generated command graph only: review before running.",
        "# The planner itself executes none of these commands.",
        "set -euo pipefail",
    ]
    for step in steps:
        lines.extend(("", f"# {step.name} [{step.kind}]", f"# {step.description}"))
        if step.dependencies:
            lines.append(f"# depends on: {', '.join(step.dependencies)}")
        if step.note:
            lines.append(f"# note: {step.note}")
        if step.blockers:
            for blocker in step.blockers:
                lines.append(f"# BLOCKED: {blocker}")
            if step.command is not None:
                preview = _shell_command(step)
                lines.append(f"# blocked command: {preview}")
            continue
        if step.command is None:
            lines.append("# BLOCKED: exact command was not reconstructed")
            continue
        lines.append(_shell_command(step))
    return "\n".join(lines) + "\n"


def _shell_command(step: Step) -> str:
    environment = [
        f"{name}={value}" for name, value in sorted(step.environment.items())
    ]
    scrub = [argument for name in SCRUB_ENVIRONMENT for argument in ("-u", name)]
    command = [*scrub, *environment, *(step.command or ())]
    return f"(cd {shlex.quote(str(step.cwd))} && env {shlex.join(command)})"


def _render_check(steps: Sequence[Step]) -> str:
    lines = []
    for step in steps:
        status = "READY" if step.runnable else "BLOCKED"
        lines.append(f"{status:7} {step.name}")
        for blocker in step.blockers:
            lines.append(f"        - {blocker}")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "check", "print"))
    parser.add_argument("--family", action="append", choices=("all", *FAMILIES))
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    parser.add_argument("--python", help="explicit absolute Python executable")
    parser.add_argument("--output-root", help="explicit absolute fresh-result root")
    parser.add_argument(
        "--noncausal-build-root",
        help="explicit absolute build root for historical non-causal extensions",
    )
    parser.add_argument(
        "--cuda-home",
        help=(
            "explicit absolute CUDA toolkit root; required by source-building "
            "families"
        ),
    )
    parser.add_argument(
        "--cutlass-dsl-root",
        help=(
            "explicit absolute CUTLASS DSL Python-package root; required by "
            "source-building families"
        ),
    )
    parser.add_argument("--gpu", default="0", help="single visible GPU index")
    parser.add_argument(
        "--artifact-manifest",
        action="append",
        default=[],
        help=f"explicit absolute {ARTIFACT_SCHEMA} path; repeat per route/batch",
    )
    parser.add_argument(
        "--external-assets-manifest",
        help=f"explicit absolute {ASSET_SCHEMA} path",
    )
    parser.add_argument(
        "--launcher-manifest",
        help=f"explicit absolute {LAUNCHER_SCHEMA} path",
    )
    return parser


def _families(raw: Sequence[str] | None) -> tuple[str, ...]:
    if not raw or raw == ["all"]:
        return tuple(FAMILIES)
    if "all" in raw:
        raise PlanError("--family all cannot be combined with another family")
    return tuple(dict.fromkeys(raw))


def _require_fresh_directory(path: Path, label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise PlanError(f"{label} exists and is not a directory: {path}")
    try:
        first_entry = next(path.iterdir())
    except StopIteration:
        return
    raise PlanError(
        f"{label} must be absent or empty for a fresh measurement; "
        f"found {first_entry}"
    )


def _context(args: argparse.Namespace, selected_families: Sequence[str]) -> Context:
    source_building = bool(
        {"noncausal-forward", "downstream", "vit-mae", "wan"}.intersection(
            selected_families
        )
    )
    missing = [
        option
        for option, value in (
            ("--python", args.python),
            ("--output-root", args.output_root),
            (
                "--noncausal-build-root",
                args.noncausal_build_root if source_building else "not-needed",
            ),
            ("--cuda-home", args.cuda_home if source_building else "not-needed"),
            (
                "--cutlass-dsl-root",
                args.cutlass_dsl_root if source_building else "not-needed",
            ),
        )
        if value is None
    ]
    if missing:
        raise PlanError(f"{', '.join(missing)} required for {args.action}")
    python = _absolute(
        args.python,
        "Python executable",
        must_exist=True,
        preserve_final_symlink=True,
    )
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PlanError(f"Python executable is not an executable file: {python}")
    if source_building:
        build_python = python.parent / "python3"
        if (
            not build_python.is_file()
            or not os.access(build_python, os.X_OK)
            or not os.path.samefile(python, build_python)
        ):
            raise PlanError(
                "source-building families invoke python3 from PATH; the selected "
                f"interpreter must have an equivalent executable at {build_python}"
            )
    output_root = _absolute(args.output_root, "output root")
    noncausal_build_root = (
        _absolute(args.noncausal_build_root, "non-causal build root")
        if args.noncausal_build_root is not None
        else output_root / ".unused-noncausal-build"
    )
    cuda_home = (
        _absolute(args.cuda_home, "CUDA root", must_exist=True)
        if args.cuda_home is not None
        else None
    )
    cutlass_dsl_root = (
        _absolute(args.cutlass_dsl_root, "CUTLASS DSL root", must_exist=True)
        if args.cutlass_dsl_root is not None
        else None
    )
    _require_fresh_directory(output_root, "output root")
    if source_building:
        _require_fresh_directory(noncausal_build_root, "non-causal build root")
        if (
            output_root == noncausal_build_root
            or output_root in noncausal_build_root.parents
            or noncausal_build_root in output_root.parents
        ):
            raise PlanError("output root and non-causal build root must not overlap")
    if cuda_home is not None and not (cuda_home / "bin/nvcc").is_file():
        raise PlanError(f"CUDA root does not contain bin/nvcc: {cuda_home}")
    if cutlass_dsl_root is not None:
        cutlass_init = cutlass_dsl_root / "cutlass/__init__.py"
        if not cutlass_init.is_file():
            raise PlanError(
                "CUTLASS DSL root does not contain cutlass/__init__.py: "
                f"{cutlass_dsl_root}"
            )
        version_match = re.search(
            r'^__version__\s*=\s*["\']([^"\']+)["\']',
            cutlass_init.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        observed_version = version_match.group(1) if version_match else None
        if observed_version != EXPECTED_CUTLASS_DSL:
            raise PlanError(
                f"CUTLASS DSL {observed_version!r} != {EXPECTED_CUTLASS_DSL!r}"
            )
    if GPU_SELECTOR.fullmatch(args.gpu) is None:
        raise PlanError("--gpu must be one non-negative integer index")
    artifacts, artifact_errors = load_artifacts(
        tuple(Path(value) for value in args.artifact_manifest)
    )
    assets, asset_manifest_path = load_external_assets(
        Path(args.external_assets_manifest) if args.external_assets_manifest else None
    )
    launcher, launcher_error = load_launcher(
        Path(args.launcher_manifest) if args.launcher_manifest else None
    )
    return Context(
        python=python,
        output_root=output_root,
        noncausal_build_root=noncausal_build_root,
        cuda_home=cuda_home,
        cutlass_dsl_root=cutlass_dsl_root,
        gpu=args.gpu,
        artifacts=artifacts,
        artifact_errors=artifact_errors,
        assets=assets,
        asset_manifest_path=asset_manifest_path,
        launcher=launcher,
        launcher_error=launcher_error,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        selected_families = _families(args.family)
        if args.action == "list":
            if args.format == "json":
                print(json.dumps(FAMILIES, indent=2, sort_keys=True))
            else:
                for name, description in FAMILIES.items():
                    print(f"{name:24} {description}")
            return 0
        context = _context(args, selected_families)
        steps = _selected_steps(build_steps(context), selected_families)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "schema": "fa4_measurement_command_graph_v1",
                        "matrix": str(MATRIX),
                        "families": list(selected_families),
                        "steps": [step.record() for step in steps],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.action == "check":
            print(_render_check(steps), end="")
        else:
            print(_render_shell(steps), end="")
        return 2 if any(not step.runnable for step in steps) else 0
    except (PlanError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
