#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
"""Schema and authentication for compiled FA4 training artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


# Version 2 replaces historical repository-commit labels with the exact live
# release checkout and a content-authenticated source closure.
SCHEMA = "fa4_artifact_manifest_v3"
SOURCE_IDENTITY_SCHEMA = "fa4_release_source_identity_v1"
SOURCE_CLOSURE_ALGORITHM = "sha256-path-size-content-v1"
SOURCE_CLOSURE_TREES = (
    "TK_quantisation",
    "tk_fa4",
    "torchtitan",
    "ThunderKittens/include",
    "ThunderKittens/prototype",
    "flash-attention/flash_attn/cute",
)
SOURCE_CLOSURE_FILES = (
    "scripts/fa4/run_torchrun.sh",
    "tools/build_fa4.py",
    "tools/fa4_dataset_manifest.py",
    "tools/plan_fa4_measurements.py",
    "tools/render_fa4_training_config.py",
    "tools/verify_fa4_training_config.py",
    "ThunderKittens/kernels/common.mk",
    "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_quantize.cuh",
)
_SOURCE_EXCLUDED_DIRECTORIES = {"__pycache__", "results"}
_SOURCE_EXCLUDED_SUFFIXES = {
    ".a",
    ".cubin",
    ".ncu-rep",
    ".nsys-rep",
    ".o",
    ".ptx",
    ".pyc",
    ".sass",
    ".so",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_LOWP_ROUTES = {
    "nvfp4_qk_fp8_pv": ("e4m3_fp8", "nvfp4"),
    "nvfp4_qk_mxfp4_pv": ("mxfp4_e8m0_block32", "nvfp4"),
    "e4m3_proj_nvfp4_qk_fp8_pv": ("e4m3_fp8", "e4m3"),
    "e4m3_proj_nvfp4_qk_mxfp4_pv": (
        "mxfp4_e8m0_block32",
        "e4m3",
    ),
}
_ROUTES = {"bf16_fa4", *_LOWP_ROUTES}
_PROFILE_SPECS = {
    "llama1p2b-d64-b16": {
        "model_preset": "llama3.2-1b",
        "runtime_contract": "d64-b16-native-v416",
        "fp8_pv_forward": "exact",
        "mxfp4_pv_forward": "d4q01-anchored",
        "projection_publisher": "b300-lowp-bwd",
        "native_backward": "v416",
        "shape": (16, 4096, 32, 8, 64),
        "routes": {
            "bf16_fa4",
            "e4m3_proj_nvfp4_qk_fp8_pv",
            "e4m3_proj_nvfp4_qk_mxfp4_pv",
        },
    },
    **{
        f"llama8b-d128-b{batch}": {
            "model_preset": "llama3.1-8b",
            "runtime_contract": f"d128-b{batch}-native-v509",
            "fp8_pv_forward": "exact",
            "mxfp4_pv_forward": "maxsafe-anchor32-represented",
            "projection_publisher": "b300-lowp-bwd",
            "native_backward": "v509",
            "shape": (batch, 4096, 32, 8, 128),
            "routes": _ROUTES,
        }
        for batch in (1, 2, 4)
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name!r}")
        result[name] = value
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _plain_int(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _plain_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class GitIdentity:
    head: str
    head_tree: str
    dirty: bool

    @classmethod
    def parse(cls, value: object, label: str) -> "GitIdentity":
        raw = _mapping(value, label)
        _exact_keys(raw, {"head", "head_tree", "dirty"}, label)
        head = _plain_string(raw["head"], f"{label}.head")
        head_tree = _plain_string(raw["head_tree"], f"{label}.head_tree")
        if _COMMIT.fullmatch(head) is None:
            raise ValueError(f"{label}.head must be a lowercase Git object ID")
        if _COMMIT.fullmatch(head_tree) is None:
            raise ValueError(f"{label}.head_tree must be a lowercase Git object ID")
        dirty = raw["dirty"]
        if type(dirty) is not bool:
            raise TypeError(f"{label}.dirty must be exactly bool")
        return cls(head=head, head_tree=head_tree, dirty=dirty)


def _git_output(root: Path, label: str, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cannot authenticate {label} Git identity: {detail}")
    return completed.stdout.strip()


def _live_git_identity(root: Path, label: str) -> GitIdentity:
    return GitIdentity(
        head=_git_output(root, label, "rev-parse", "HEAD"),
        head_tree=_git_output(root, label, "rev-parse", "HEAD^{tree}"),
        dirty=bool(
            _git_output(
                root,
                label,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
        ),
    )


def _release_gitlink(root: Path, name: str) -> str:
    row = _git_output(root, "release source root", "ls-tree", "HEAD", "--", name)
    fields = row.split(maxsplit=3)
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        raise RuntimeError(f"release HEAD does not contain submodule gitlink {name!r}")
    return fields[2]


@dataclass(frozen=True)
class SourceFileIdentity:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def parse(cls, value: object, label: str) -> "SourceFileIdentity":
        raw = _mapping(value, label)
        _exact_keys(raw, {"path", "bytes", "sha256"}, label)
        relative = _plain_string(raw["path"], f"{label}.path")
        candidate = Path(relative)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"{label}.path must be a safe relative path")
        size = raw["bytes"]
        if type(size) is not int or size < 0:
            raise TypeError(f"{label}.bytes must be a non-negative integer")
        sha256 = _plain_string(raw["sha256"], f"{label}.sha256")
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError(f"{label}.sha256 must be lowercase SHA256")
        return cls(path=candidate.as_posix(), bytes=size, sha256=sha256)

    def authenticate(self, root: Path, label: str) -> None:
        path = root / self.path
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{label} does not exist: {path}") from error
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} escapes release source root: {path}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} is not a file: {resolved}")
        observed_bytes = resolved.stat().st_size
        if observed_bytes != self.bytes:
            raise RuntimeError(
                f"{label} byte identity mismatch: {observed_bytes} != {self.bytes}"
            )
        observed_sha256 = _sha256(resolved)
        if observed_sha256 != self.sha256:
            raise RuntimeError(
                f"{label} SHA256 mismatch: {observed_sha256} != {self.sha256}"
            )


def _source_closure_digest(files: tuple[SourceFileIdentity, ...]) -> str:
    digest = hashlib.sha256()
    for record in files:
        digest.update(record.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_file_is_in_scope(path: Path, *, tree: Path) -> bool:
    relative = path.relative_to(tree)
    if any(
        part in _SOURCE_EXCLUDED_DIRECTORIES or part.startswith(".causal_")
        for part in relative.parts[:-1]
    ):
        return False
    return not any(path.name.endswith(suffix) for suffix in _SOURCE_EXCLUDED_SUFFIXES)


def _release_source_paths(root: Path) -> tuple[str, ...]:
    paths: set[Path] = set()
    for relative_tree in SOURCE_CLOSURE_TREES:
        tree = root / relative_tree
        if not tree.is_dir():
            raise FileNotFoundError(f"release source-closure tree is missing: {tree}")
        paths.update(
            path
            for path in tree.rglob("*")
            if path.is_file() and _source_file_is_in_scope(path, tree=tree)
        )
    for relative_file in SOURCE_CLOSURE_FILES:
        path = root / relative_file
        if not path.is_file():
            raise FileNotFoundError(f"release source-closure file is missing: {path}")
        paths.add(path)
    return tuple(sorted(path.relative_to(root).as_posix() for path in paths))


@dataclass(frozen=True)
class ReleaseSourceIdentity:
    root: Path
    git: GitIdentity
    submodules: Mapping[str, GitIdentity]
    closure_sha256: str
    files: tuple[SourceFileIdentity, ...]

    @classmethod
    def parse(cls, value: object) -> "ReleaseSourceIdentity":
        raw = _mapping(value, "sources.release")
        _exact_keys(
            raw,
            {"schema", "root", "git", "submodules", "closure"},
            "sources.release",
        )
        if raw["schema"] != SOURCE_IDENTITY_SCHEMA:
            raise ValueError(f"unsupported release source schema {raw['schema']!r}")
        root = Path(_plain_string(raw["root"], "sources.release.root"))
        if not root.is_absolute():
            raise ValueError("sources.release.root must be absolute")
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"release source root does not exist: {root}")
        git = GitIdentity.parse(raw["git"], "sources.release.git")

        raw_submodules = _mapping(raw["submodules"], "sources.release.submodules")
        required_submodules = {
            "ThunderKittens",
            "cutlass",
            "flash-attention",
            "qutlass",
        }
        _exact_keys(raw_submodules, required_submodules, "sources.release.submodules")
        submodules = {
            name: GitIdentity.parse(
                raw_submodules[name], f"sources.release.submodules.{name}"
            )
            for name in sorted(raw_submodules)
        }

        closure = _mapping(raw["closure"], "sources.release.closure")
        _exact_keys(
            closure,
            {"algorithm", "scope", "sha256", "file_count", "files"},
            "sources.release.closure",
        )
        if closure["algorithm"] != SOURCE_CLOSURE_ALGORITHM:
            raise ValueError(
                "unsupported source closure algorithm " f"{closure['algorithm']!r}"
            )
        scope = _mapping(closure["scope"], "sources.release.closure.scope")
        _exact_keys(
            scope,
            {"trees", "files", "excluded_directories", "excluded_suffixes"},
            "sources.release.closure.scope",
        )
        expected_scope = {
            "trees": list(SOURCE_CLOSURE_TREES),
            "files": list(SOURCE_CLOSURE_FILES),
            "excluded_directories": sorted(_SOURCE_EXCLUDED_DIRECTORIES),
            "excluded_suffixes": sorted(_SOURCE_EXCLUDED_SUFFIXES),
        }
        if dict(scope) != expected_scope:
            raise ValueError("release source closure uses a noncanonical scope")
        closure_sha256 = _plain_string(
            closure["sha256"], "sources.release.closure.sha256"
        )
        if _SHA256.fullmatch(closure_sha256) is None:
            raise ValueError("sources.release.closure.sha256 must be lowercase SHA256")
        file_count = _plain_int(
            closure["file_count"],
            "sources.release.closure.file_count",
            positive=True,
        )
        raw_files = closure["files"]
        if not isinstance(raw_files, list):
            raise TypeError("sources.release.closure.files must be an array")
        files = tuple(
            SourceFileIdentity.parse(item, f"sources.release.closure.files[{index}]")
            for index, item in enumerate(raw_files)
        )
        if len(files) != file_count:
            raise ValueError(
                "sources.release.closure.file_count does not match its file array"
            )
        paths = tuple(record.path for record in files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError(
                "sources.release.closure.files must have unique sorted paths"
            )
        observed_paths = _release_source_paths(root)
        if paths != observed_paths:
            raise RuntimeError(
                "release source inventory differs from the build manifest"
            )
        required_paths = {
            "scripts/fa4/run_torchrun.sh",
            "tools/build_fa4.py",
            "tools/fa4_dataset_manifest.py",
            "tools/render_fa4_training_config.py",
            "tools/verify_fa4_training_config.py",
            "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py",
            "tk_fa4/lowp_fa4_bwd/native_tk_d64_backward.py",
            "torchtitan/experiments/fa4/artifacts.py",
            "torchtitan/experiments/fa4/exact_lowp_attention.py",
            "ThunderKittens/kernels/common.mk",
            "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_quantize.cuh",
            "flash-attention/flash_attn/cute/interface.py",
        }
        missing = required_paths - set(paths)
        if missing:
            raise ValueError(
                "release source closure omits required sources: " f"{sorted(missing)}"
            )
        observed_closure_sha256 = _source_closure_digest(files)
        if observed_closure_sha256 != closure_sha256:
            raise RuntimeError(
                "release source closure record mismatch: "
                f"{observed_closure_sha256} != {closure_sha256}"
            )
        for index, record in enumerate(files):
            record.authenticate(
                root,
                f"sources.release.closure.files[{index}] ({record.path})",
            )
        live_git = _live_git_identity(root, "release source root")
        if live_git != git:
            raise RuntimeError(
                "release source Git identity mismatch: "
                f"observed={live_git}, manifest={git}"
            )
        for name, expected in submodules.items():
            if expected.dirty:
                raise ValueError(f"submodule {name} must be clean in a build manifest")
            gitlink = _release_gitlink(root, name)
            if gitlink != expected.head:
                raise RuntimeError(
                    f"submodule {name} manifest head {expected.head} does not "
                    f"match release gitlink {gitlink}"
                )
            observed = _live_git_identity(root / name, f"submodule {name}")
            if observed != expected:
                raise RuntimeError(
                    f"submodule {name} Git identity mismatch: "
                    f"observed={observed}, manifest={expected}"
                )
        return cls(
            root=root,
            git=git,
            submodules=submodules,
            closure_sha256=closure_sha256,
            files=files,
        )

    def file(self, relative_path: str) -> SourceFileIdentity:
        for record in self.files:
            if record.path == relative_path:
                return record
        raise KeyError(relative_path)


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    bytes: int
    sha256: str
    module: str | None = None

    @classmethod
    def parse(
        cls,
        value: object,
        label: str,
        *,
        requires_module: bool,
    ) -> "FileIdentity":
        raw = _mapping(value, label)
        expected = {"path", "bytes", "sha256"}
        if requires_module:
            expected.add("module")
        _exact_keys(raw, expected, label)
        path = Path(_plain_string(raw["path"], f"{label}.path"))
        if not path.is_absolute():
            raise ValueError(f"{label}.path must be absolute: {path}")
        path = path.resolve()
        size = _plain_int(raw["bytes"], f"{label}.bytes", positive=True)
        sha256 = _plain_string(raw["sha256"], f"{label}.sha256")
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError(f"{label}.sha256 must be lowercase SHA256")
        module = None
        if requires_module:
            module = _plain_string(raw["module"], f"{label}.module")
            if not module.isidentifier():
                raise ValueError(f"{label}.module must be a Python identifier")
        identity = cls(path=path, bytes=size, sha256=sha256, module=module)
        identity.authenticate(label)
        return identity

    def authenticate(self, label: str) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {self.path}")
        observed_bytes = self.path.stat().st_size
        if observed_bytes != self.bytes:
            raise RuntimeError(
                f"{label} byte identity mismatch: {observed_bytes} != {self.bytes}"
            )
        observed_sha256 = _sha256(self.path)
        if observed_sha256 != self.sha256:
            raise RuntimeError(
                f"{label} SHA256 mismatch: {observed_sha256} != {self.sha256}"
            )


@dataclass(frozen=True)
class CutlassDSLIdentity:
    root: Path
    version: str
    native: FileIdentity
    closure_sha256: str
    files: tuple[SourceFileIdentity, ...]

    @classmethod
    def parse(cls, value: object) -> "CutlassDSLIdentity":
        raw = _mapping(value, "sources.cutlass_dsl")
        _exact_keys(
            raw,
            {"root", "version", "native", "closure"},
            "sources.cutlass_dsl",
        )
        root = Path(_plain_string(raw["root"], "sources.cutlass_dsl.root"))
        if not root.is_absolute():
            raise ValueError("sources.cutlass_dsl.root must be absolute")
        root = root.resolve()
        if not (root / "cutlass" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"CUTLASS DSL Python package is incomplete below {root}"
            )
        version = _plain_string(raw["version"], "sources.cutlass_dsl.version")
        if version != "4.5.2":
            raise ValueError("the measured FA4 runtime requires CUTLASS DSL 4.5.2")
        native = FileIdentity.parse(
            raw["native"], "sources.cutlass_dsl.native", requires_module=False
        )
        try:
            native.path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "CUTLASS DSL native library is outside its root"
            ) from error
        native_candidates = tuple(
            candidate.resolve()
            for candidate in (root / "cutlass" / "_mlir" / "_mlir_libs").glob(
                "_cutlass_ir*.so"
            )
        )
        if native_candidates != (native.path,):
            raise ValueError(
                "CUTLASS DSL root must contain exactly the authenticated "
                "_cutlass_ir native library"
            )
        closure = _mapping(raw["closure"], "sources.cutlass_dsl.closure")
        _exact_keys(
            closure,
            {"algorithm", "sha256", "file_count", "files"},
            "sources.cutlass_dsl.closure",
        )
        if closure["algorithm"] != SOURCE_CLOSURE_ALGORITHM:
            raise ValueError("unsupported CUTLASS DSL closure algorithm")
        closure_sha256 = _plain_string(
            closure["sha256"], "sources.cutlass_dsl.closure.sha256"
        )
        if _SHA256.fullmatch(closure_sha256) is None:
            raise ValueError(
                "sources.cutlass_dsl.closure.sha256 must be lowercase SHA256"
            )
        file_count = _plain_int(
            closure["file_count"],
            "sources.cutlass_dsl.closure.file_count",
            positive=True,
        )
        raw_files = closure["files"]
        if not isinstance(raw_files, list):
            raise TypeError("sources.cutlass_dsl.closure.files must be an array")
        files = tuple(
            SourceFileIdentity.parse(
                value, f"sources.cutlass_dsl.closure.files[{index}]"
            )
            for index, value in enumerate(raw_files)
        )
        paths = tuple(record.path for record in files)
        if len(files) != file_count:
            raise ValueError(
                "sources.cutlass_dsl.closure.file_count does not match its file array"
            )
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError(
                "sources.cutlass_dsl.closure.files must have unique sorted paths"
            )
        if not all(path.startswith("cutlass/") for path in paths):
            raise ValueError("CUTLASS DSL closure paths must remain below cutlass/")
        observed_paths = tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in (root / "cutlass").rglob("*")
                if path.is_file()
                and "__pycache__" not in path.relative_to(root / "cutlass").parts
                and path.suffix != ".pyc"
            )
        )
        if paths != observed_paths:
            raise RuntimeError(
                "CUTLASS DSL source inventory differs from the build manifest"
            )
        observed_closure_sha256 = _source_closure_digest(files)
        if observed_closure_sha256 != closure_sha256:
            raise RuntimeError(
                "CUTLASS DSL source closure record mismatch: "
                f"{observed_closure_sha256} != {closure_sha256}"
            )
        for index, record in enumerate(files):
            record.authenticate(
                root,
                f"sources.cutlass_dsl.closure.files[{index}] ({record.path})",
            )
        native_relative = native.path.relative_to(root).as_posix()
        native_record = next(
            (record for record in files if record.path == native_relative),
            None,
        )
        if native_record is None or (
            native_record.bytes,
            native_record.sha256,
        ) != (native.bytes, native.sha256):
            raise ValueError("CUTLASS DSL native identity is absent from its closure")
        return cls(
            root=root,
            version=version,
            native=native,
            closure_sha256=closure_sha256,
            files=files,
        )


@dataclass(frozen=True)
class ArtifactManifest:
    path: Path
    purpose: str
    route: str
    pv_format: str | None
    learned_projection_format: str | None
    batch: int
    sequence: int
    q_heads: int
    kv_heads: int
    head_dim: int
    gpu: str
    compute_capability: tuple[int, int]
    cuda_arch: str
    forward: FileIdentity | None
    projection_publisher: FileIdentity | None
    v509_backward: FileIdentity | None
    release_source: ReleaseSourceIdentity
    runtime_source: FileIdentity | None
    flash_interface: FileIdentity | None
    cutlass_dsl: CutlassDSLIdentity | None
    profile: str = "llama8b-d128-b1"
    v416_backward: FileIdentity | None = None

    @property
    def is_low_precision(self) -> bool:
        return self.route in _LOWP_ROUTES

    @property
    def native_backward(self) -> FileIdentity | None:
        """Return the profile-specific native consumer without aliasing ABIs."""
        if self.profile == "llama1p2b-d64-b16":
            return self.v416_backward
        return self.v509_backward


def load_artifact_manifest(
    path: str | Path,
    *,
    require_training: bool = True,
) -> ArtifactManifest:
    """Load, structurally validate, and rehash one manifest and every file."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest does not exist: {manifest_path}")
    raw = _mapping(
        json.loads(
            manifest_path.read_text(),
            object_pairs_hook=_object_without_duplicate_keys,
        ),
        "manifest",
    )
    _exact_keys(
        raw,
        {
            "schema",
            "purpose",
            "profile",
            "route",
            "shape",
            "architecture",
            "artifacts",
            "sources",
        },
        "manifest",
    )
    if raw["schema"] != SCHEMA:
        raise ValueError(f"unsupported artifact schema {raw['schema']!r}")
    purpose = _plain_string(raw["purpose"], "purpose")
    if purpose not in {"training", "operator_only"}:
        raise ValueError("purpose must be 'training' or 'operator_only'")
    if require_training and purpose != "training":
        raise ValueError("training refuses an operator-only artifact manifest")

    profile_raw = _mapping(raw["profile"], "profile")
    _exact_keys(
        profile_raw,
        {
            "name",
            "model_preset",
            "runtime_contract",
            "fp8_pv_forward",
            "mxfp4_pv_forward",
            "projection_publisher",
            "native_backward",
        },
        "profile",
    )
    profile = _plain_string(profile_raw["name"], "profile.name")
    profile_spec = _PROFILE_SPECS.get(profile)
    if profile_spec is None:
        raise ValueError(f"unsupported FA4 build profile {profile!r}")
    for field in (
        "model_preset",
        "runtime_contract",
        "fp8_pv_forward",
        "mxfp4_pv_forward",
        "projection_publisher",
        "native_backward",
    ):
        if profile_raw[field] != profile_spec[field]:
            raise ValueError(
                f"profile {profile!r} requires {field}=" f"{profile_spec[field]!r}"
            )

    route_raw = _mapping(raw["route"], "route")
    _exact_keys(
        route_raw,
        {"name", "pv_format", "learned_projection_format"},
        "route",
    )
    route = _plain_string(route_raw["name"], "route.name")
    if route not in _ROUTES:
        raise ValueError(f"unsupported FA4 route {route!r}")
    if route not in profile_spec["routes"]:
        raise ValueError(f"route {route!r} is not valid for profile {profile!r}")
    pv_format = route_raw["pv_format"]
    projection_format = route_raw["learned_projection_format"]
    if route == "bf16_fa4":
        if pv_format is not None or projection_format is not None:
            raise ValueError("BF16 route must use null PV and projection formats")
    else:
        required_pv_format, required_projection_format = _LOWP_ROUTES[route]
        if pv_format != required_pv_format:
            raise ValueError(f"route {route} requires pv_format={required_pv_format!r}")
        if projection_format != required_projection_format:
            raise ValueError(
                f"route {route} requires learned_projection_format="
                f"{required_projection_format!r}"
            )

    shape = _mapping(raw["shape"], "shape")
    _exact_keys(
        shape, {"batch", "sequence", "q_heads", "kv_heads", "head_dim"}, "shape"
    )
    batch = _plain_int(shape["batch"], "shape.batch", positive=True)
    sequence = _plain_int(shape["sequence"], "shape.sequence", positive=True)
    q_heads = _plain_int(shape["q_heads"], "shape.q_heads", positive=True)
    kv_heads = _plain_int(shape["kv_heads"], "shape.kv_heads", positive=True)
    head_dim = _plain_int(shape["head_dim"], "shape.head_dim", positive=True)
    observed_shape = (batch, sequence, q_heads, kv_heads, head_dim)
    if observed_shape != profile_spec["shape"]:
        raise ValueError(
            f"profile {profile!r} requires shape {profile_spec['shape']!r}; "
            f"observed {observed_shape!r}"
        )
    if (
        require_training
        and profile.startswith("llama8b-d128-")
        and route in _LOWP_ROUTES
        and batch not in (1, 4)
    ):
        raise ValueError(
            "native-score E5M2-dO training is authenticated only for B1/B4"
        )

    architecture = _mapping(raw["architecture"], "architecture")
    _exact_keys(
        architecture, {"gpu", "compute_capability", "cuda_arch"}, "architecture"
    )
    gpu = _plain_string(architecture["gpu"], "architecture.gpu")
    capability_raw = architecture["compute_capability"]
    if not isinstance(capability_raw, list) or len(capability_raw) != 2:
        raise TypeError("architecture.compute_capability must be [major, minor]")
    capability = tuple(
        _plain_int(item, "architecture.compute_capability") for item in capability_raw
    )
    cuda_arch = _plain_string(architecture["cuda_arch"], "architecture.cuda_arch")
    if capability != (10, 0) or cuda_arch not in {"sm_100", "sm_100a"}:
        raise ValueError("FA4 training artifacts require Blackwell SM100")
    if "b200" not in gpu.lower() and "gb200" not in gpu.lower():
        raise ValueError("architecture.gpu must identify the measured B200/GB200 GPU")

    artifacts = _mapping(raw["artifacts"], "artifacts")
    _exact_keys(
        artifacts,
        {"forward", "projection_publisher", "native_backward"},
        "artifacts",
    )
    parsed_artifacts: dict[str, FileIdentity | None] = {}
    for name in ("forward", "projection_publisher", "native_backward"):
        value = artifacts[name]
        parsed_artifacts[name] = (
            None
            if value is None
            else FileIdentity.parse(value, f"artifacts.{name}", requires_module=True)
        )
    if route == "bf16_fa4":
        if any(parsed_artifacts.values()):
            raise ValueError("BF16 manifest must not claim low-precision binaries")
    elif any(value is None for value in parsed_artifacts.values()):
        raise ValueError("low-precision manifest requires all three artifacts")
    if parsed_artifacts["projection_publisher"] is not None and (
        parsed_artifacts["projection_publisher"].module != "_C_b300_lowp_bwd"
    ):
        raise ValueError("projection publisher module must be _C_b300_lowp_bwd")
    if profile == "llama1p2b-d64-b16":
        expected_forward = (
            "_C_tk_causal_gqa_nvfp4_fp8pv_exact_b16s4096h32kv8d64"
            if pv_format == "e4m3_fp8"
            else "_C_cfwd_mx_d4q01_b16s4096h32kv8d64"
        )
    else:
        expected_forward = (
            "_C_tk_causal_gqa_nvfp4_fp8pv_exact_" f"b{batch}s4096h32kv8d128"
            if pv_format == "e4m3_fp8"
            else (
                "_C_d128_mx_maxsafe_anchor32_represented_"
                f"b{batch}s4096h32kv8d128_b200_sm152"
            )
        )
    if (
        parsed_artifacts["forward"] is not None
        and parsed_artifacts["forward"].module != expected_forward
    ):
        raise ValueError(
            f"forward module does not match profile {profile!r} and route "
            f"{route!r}: {parsed_artifacts['forward'].module!r} != "
            f"{expected_forward!r}"
        )
    expected_backward = (
        "_C_sm100_gqa_tk_v416_d64_e4m3_production_bshd_dq_first"
        if profile == "llama1p2b-d64-b16"
        else (
            "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_"
            f"e5m2_dout_b{batch}_s4096"
        )
    )
    if parsed_artifacts["native_backward"] is not None and (
        parsed_artifacts["native_backward"].module != expected_backward
    ):
        raise ValueError(
            f"{profile_raw['native_backward']} backward module does not match "
            f"profile {profile!r}: "
            f"{parsed_artifacts['native_backward'].module!r} != "
            f"{expected_backward!r}"
        )

    sources = _mapping(raw["sources"], "sources")
    _exact_keys(
        sources,
        {
            "release",
            "runtime_source",
            "flash_interface",
            "cutlass_dsl",
        },
        "sources",
    )
    release_source = ReleaseSourceIdentity.parse(sources["release"])
    runtime_values = tuple(
        sources[name] for name in ("runtime_source", "flash_interface", "cutlass_dsl")
    )
    if any(value is None for value in runtime_values) and not all(
        value is None for value in runtime_values
    ):
        raise ValueError(
            "runtime_source, flash_interface, and cutlass_dsl must be either "
            "all present or all null"
        )
    if require_training and all(value is None for value in runtime_values):
        raise ValueError(
            "training manifest requires runtime_source, flash_interface, "
            "and cutlass_dsl identities"
        )
    if runtime_values[0] is None:
        runtime_source = None
        flash_interface = None
        cutlass_dsl = None
    else:
        runtime_source = FileIdentity.parse(
            sources["runtime_source"],
            "sources.runtime_source",
            requires_module=False,
        )
        if runtime_source.path.name != "benchmark_llama12b_e2e.py":
            raise ValueError("runtime_source is not benchmark_llama12b_e2e.py")
        expected_runtime_path = (
            release_source.root
            / "tk_fa4"
            / "lowp_fa4_bwd"
            / "benchmark_llama12b_e2e.py"
        )
        if runtime_source.path != expected_runtime_path:
            raise ValueError(
                "runtime_source is outside the authenticated release source root"
            )
        runtime_record = release_source.file(
            "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
        )
        if (runtime_record.bytes, runtime_record.sha256) != (
            runtime_source.bytes,
            runtime_source.sha256,
        ):
            raise ValueError("runtime_source disagrees with the release closure")
        flash_interface = FileIdentity.parse(
            sources["flash_interface"],
            "sources.flash_interface",
            requires_module=False,
        )
        if flash_interface.path.name != "interface.py":
            raise ValueError("flash_interface is not interface.py")
        expected_flash_path = (
            release_source.root
            / "flash-attention"
            / "flash_attn"
            / "cute"
            / "interface.py"
        )
        if flash_interface.path != expected_flash_path:
            raise ValueError(
                "flash_interface is outside the authenticated release source root"
            )
        flash_record = release_source.file(
            "flash-attention/flash_attn/cute/interface.py"
        )
        if (flash_record.bytes, flash_record.sha256) != (
            flash_interface.bytes,
            flash_interface.sha256,
        ):
            raise ValueError("flash_interface disagrees with the release closure")
        cutlass_dsl = CutlassDSLIdentity.parse(sources["cutlass_dsl"])

    return ArtifactManifest(
        path=manifest_path,
        purpose=purpose,
        route=route,
        pv_format=pv_format,
        learned_projection_format=projection_format,
        batch=batch,
        sequence=sequence,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        gpu=gpu,
        compute_capability=capability,
        cuda_arch=cuda_arch,
        forward=parsed_artifacts["forward"],
        projection_publisher=parsed_artifacts["projection_publisher"],
        v509_backward=(
            parsed_artifacts["native_backward"]
            if profile.startswith("llama8b-d128-")
            else None
        ),
        release_source=release_source,
        runtime_source=runtime_source,
        flash_interface=flash_interface,
        cutlass_dsl=cutlass_dsl,
        profile=profile,
        v416_backward=(
            parsed_artifacts["native_backward"]
            if profile == "llama1p2b-d64-b16"
            else None
        ),
    )


__all__ = [
    "ArtifactManifest",
    "FileIdentity",
    "GitIdentity",
    "ReleaseSourceIdentity",
    "SCHEMA",
    "SOURCE_CLOSURE_ALGORITHM",
    "SOURCE_CLOSURE_FILES",
    "SOURCE_CLOSURE_TREES",
    "SOURCE_IDENTITY_SCHEMA",
    "SourceFileIdentity",
    "load_artifact_manifest",
]
