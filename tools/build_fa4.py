#!/usr/bin/env python3
"""Plan and build the pinned SM100 FP4 FlashAttention extensions.

The CLI deliberately has no build-directory or CUDA-toolkit defaults.  Every
artifact is written below the absolute ``--build-root`` supplied by the user.
Use ``plan`` for a CPU-only dry run, ``verify`` to check the complete measured
environment, and ``build`` to compile and write authenticated artifact
manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]

FLASH_ATTENTION_OVERLAY_COMMIT = "b531f67557b8213db339492cd1629e721776f758"

ARTIFACT_MANIFEST_SCHEMA = "fa4_artifact_manifest_v3"
SOURCE_IDENTITY_SCHEMA = "fa4_release_source_identity_v1"
SOURCE_CLOSURE_ALGORITHM = "sha256-path-size-content-v1"

# This is deliberately broader than the compiler's direct include graph.  It
# covers every in-repository source that can participate in the supported FA4
# build or be imported by the training adapter.  Results and generated code are
# excluded below.  A broad content closure is cheap enough for this research
# workspace and avoids silently missing a dynamically selected Python helper or
# C++ header.
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

EXPECTED_SUBMODULES = {
    "ThunderKittens": "9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b",
    "cutlass": "acb45938e9cb3e4db8c1d75155b63d31791e0e5d",
    "flash-attention": FLASH_ATTENTION_OVERLAY_COMMIT,
    "qutlass": "406e86fb2d7df436e94f825bcda8e59b1a7250a6",
}

EXPECTED_PYTHON = (3, 12)
EXPECTED_DRIVER = "580.126.09"
EXPECTED_CUDA_RELEASE = "13.0"
EXPECTED_DISTRIBUTIONS = {
    "torch": "2.9.0a0+145a3a7bda.nv25.10",
    "triton": "3.5.1",
    "flashinfer-python": "0.6.15.post1",
    "pybind11": "3.0.1",
}
EXPECTED_CUTLASS_DSL = "4.5.2"

BATCHES = (1, 2, 4)
D64_BATCHES = (16,)
SEQUENCE = 4096
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
NUM_SM = 152

D128_BUILD_PROFILE = "llama8b-d128"
D64_BUILD_PROFILE = "llama1p2b-d64-b16"
BUILD_PROFILES = (D128_BUILD_PROFILE, D64_BUILD_PROFILE)

D128_TARGET_GROUPS = (
    "mxfp4-quantizer",
    "projection-publisher",
    "fp8-forward",
    "mx-forward",
    "v509-backward",
)
D64_TARGET_GROUPS = (
    "projection-publisher",
    "fp8-forward",
    "mx-forward",
    "v416-backward",
)
TARGET_GROUPS = tuple(dict.fromkeys((*D128_TARGET_GROUPS, *D64_TARGET_GROUPS)))

ROUTES = {
    "bf16_fa4": {
        "pv_format": None,
        "learned_projection_format": None,
    },
    "nvfp4_qk_fp8_pv": {
        "pv_format": "e4m3_fp8",
        "learned_projection_format": "nvfp4",
    },
    "nvfp4_qk_mxfp4_pv": {
        "pv_format": "mxfp4_e8m0_block32",
        "learned_projection_format": "nvfp4",
    },
    "e4m3_proj_nvfp4_qk_fp8_pv": {
        "pv_format": "e4m3_fp8",
        "learned_projection_format": "e4m3",
    },
    "e4m3_proj_nvfp4_qk_mxfp4_pv": {
        "pv_format": "mxfp4_e8m0_block32",
        "learned_projection_format": "e4m3",
    },
}


class BuildError(RuntimeError):
    """A release build invariant was not satisfied."""


@dataclass(frozen=True)
class BuildSpec:
    """One deterministic compiler invocation and its expected output."""

    name: str
    group: str
    batch: int | None
    cwd: Path
    command: tuple[str, ...]
    output: Path
    module: str


@dataclass(frozen=True)
class BuildLayout:
    """Names and paths shared by the compiler plan and manifest writer."""

    build_root: Path
    extension_suffix: str
    profile: str = D128_BUILD_PROFILE

    @property
    def artifact_root(self) -> Path:
        return self.build_root / "artifacts"

    @property
    def manifest_root(self) -> Path:
        return self.build_root / "manifests"

    @property
    def quantizer_module(self) -> str:
        return "mxfp4_quant_v3"

    @property
    def quantizer(self) -> Path:
        return (
            self.artifact_root
            / "mxfp4_quantizer"
            / f"{self.quantizer_module}{self.extension_suffix}"
        )

    @property
    def publisher_module(self) -> str:
        return "_C_b300_lowp_bwd"

    @property
    def publisher(self) -> Path:
        return (
            self.artifact_root
            / "projection_publisher"
            / f"{self.publisher_module}{self.extension_suffix}"
        )

    def fp8_forward_module(self, batch: int) -> str:
        head_dim = 64 if self.profile == D64_BUILD_PROFILE else HEAD_DIM
        return (
            "_C_tk_causal_gqa_nvfp4_fp8pv_exact_"
            f"b{batch}s{SEQUENCE}h{Q_HEADS}kv{KV_HEADS}d{head_dim}"
        )

    def mx_forward_module(self, batch: int) -> str:
        if self.profile == D64_BUILD_PROFILE:
            return "_C_cfwd_mx_d4q01_" f"b{batch}s{SEQUENCE}h{Q_HEADS}kv{KV_HEADS}d64"
        return (
            "_C_d128_mx_maxsafe_anchor32_represented_"
            f"b{batch}s{SEQUENCE}h{Q_HEADS}kv{KV_HEADS}d{HEAD_DIM}_"
            f"b200_sm{NUM_SM}"
        )

    def forward(self, route: str, batch: int) -> tuple[str, Path]:
        route_formats = ROUTES.get(route)
        if route_formats is None or route_formats["pv_format"] is None:
            raise ValueError(f"route has no low-precision forward: {route}")
        if route_formats["pv_format"] == "e4m3_fp8":
            module = self.fp8_forward_module(batch)
        elif route_formats["pv_format"] == "mxfp4_e8m0_block32":
            module = self.mx_forward_module(batch)
        else:
            raise ValueError(
                f"route has unsupported PV format: {route_formats['pv_format']}"
            )
        return (
            module,
            self.artifact_root / "forward" / f"{module}{self.extension_suffix}",
        )

    def backward_module(self, batch: int) -> str:
        return (
            "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_"
            f"e5m2_dout_b{batch}_s{SEQUENCE}"
        )

    def backward(self, batch: int) -> tuple[str, Path]:
        if self.profile != D128_BUILD_PROFILE:
            raise ValueError("v509 backward is available only in the D128 profile")
        module = self.backward_module(batch)
        return (
            module,
            self.artifact_root / "backward" / f"{module}{self.extension_suffix}",
        )

    @property
    def v416_backward_module(self) -> str:
        return "_C_sm100_gqa_tk_v416_d64_e4m3_production_bshd_dq_first"

    @property
    def v416_backward(self) -> Path:
        return (
            self.artifact_root
            / "backward"
            / f"{self.v416_backward_module}{self.extension_suffix}"
        )

    def native_backward(self, batch: int) -> tuple[str, Path]:
        if self.profile == D64_BUILD_PROFILE:
            if batch != 16:
                raise ValueError("the D64 v416 profile is fixed to local batch 16")
            return self.v416_backward_module, self.v416_backward
        return self.backward(batch)


def _absolute_path(value: str, option: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BuildError(f"{option} must be an absolute path: {path}")
    return path.resolve()


def _extension_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise BuildError("the active Python has no extension suffix")
    return suffix


def make_layout(
    build_root: Path,
    extension_suffix: str | None = None,
    *,
    profile: str = D128_BUILD_PROFILE,
) -> BuildLayout:
    if not build_root.is_absolute():
        raise BuildError(f"build root must be absolute: {build_root}")
    if profile not in BUILD_PROFILES:
        raise BuildError(f"unknown build profile: {profile!r}")
    return BuildLayout(
        build_root=build_root.resolve(),
        extension_suffix=extension_suffix or _extension_suffix(),
        profile=profile,
    )


def build_plan(
    *,
    build_root: Path,
    cuda_home: Path,
    python: Path,
    batches: Sequence[int] | None = None,
    targets: Sequence[str] | None = None,
    profile: str = D128_BUILD_PROFILE,
    jobs: int = 1,
    nvcc_threads: int = 1,
    nvcc_split_compile: int = 1,
    extension_suffix: str | None = None,
) -> tuple[BuildLayout, tuple[BuildSpec, ...]]:
    """Construct compiler commands without probing CUDA or importing Torch."""
    if not cuda_home.is_absolute():
        raise BuildError(f"CUDA root must be absolute: {cuda_home}")
    if not python.is_absolute():
        raise BuildError(f"Python executable must be absolute: {python}")
    if profile not in BUILD_PROFILES:
        raise BuildError(f"unknown build profile: {profile!r}")
    profile_batches = D64_BATCHES if profile == D64_BUILD_PROFILE else BATCHES
    profile_targets = (
        D64_TARGET_GROUPS if profile == D64_BUILD_PROFILE else D128_TARGET_GROUPS
    )
    batches = tuple(profile_batches if batches is None else batches)
    targets = tuple(profile_targets if targets is None else targets)
    if not batches or any(batch not in profile_batches for batch in batches):
        allowed = ", ".join(str(batch) for batch in profile_batches)
        raise BuildError(
            f"{profile} batches must be a non-empty subset of {{{allowed}}}"
        )
    unknown = set(targets) - set(TARGET_GROUPS)
    if unknown:
        raise BuildError(f"unknown build target groups: {sorted(unknown)}")
    disallowed = set(targets) - set(profile_targets)
    if disallowed:
        raise BuildError(
            f"{profile} does not provide target groups: {sorted(disallowed)}"
        )
    if min(jobs, nvcc_threads, nvcc_split_compile) <= 0:
        raise BuildError("jobs and NVCC parallelism must be positive")

    layout = make_layout(build_root, extension_suffix, profile=profile)
    cuda_home = cuda_home.resolve()
    # Keep the selected virtual-environment entry point. ``resolve()`` follows
    # ``venv/bin/python`` to the host interpreter, which can make a Makefile
    # import a different Torch/pybind environment from the one authenticated
    # by ``verify_environment``.
    python = Path(os.path.abspath(os.fspath(python)))
    nvcc = cuda_home / "bin" / "nvcc"
    selected = set(targets)
    specs: list[BuildSpec] = []

    if "mxfp4-quantizer" in selected:
        specs.append(
            BuildSpec(
                name="mxfp4-quantizer",
                group="mxfp4-quantizer",
                batch=None,
                cwd=ROOT / "TK_quantisation" / "mxfp4_v3",
                command=(
                    "make",
                    "-B",
                    "-f",
                    "Makefile",
                    f"-j{jobs}",
                    "GPU=B200",
                    f"CUDA_HOME={cuda_home}",
                    f"NVCC={nvcc}",
                    f"PYTHON={python}",
                    f"OUT={layout.quantizer}",
                ),
                output=layout.quantizer,
                module=layout.quantizer_module,
            )
        )

    if "projection-publisher" in selected:
        specs.append(
            BuildSpec(
                name="projection-publisher",
                group="projection-publisher",
                batch=None,
                cwd=ROOT / "tk_fa4" / "lowp_fa4_bwd",
                command=(
                    "make",
                    "-B",
                    "-f",
                    "Makefile",
                    f"-j{jobs}",
                    "GPU=B200",
                    f"CUDA_HOME={cuda_home}",
                    f"NVCC={nvcc}",
                    f"OUT={layout.publisher}",
                ),
                output=layout.publisher,
                module=layout.publisher_module,
            )
        )

    forward_builder = ROOT / "tk_fa4" / "lowp_fa4_bwd"
    head_dim = 64 if profile == D64_BUILD_PROFILE else HEAD_DIM
    for batch in batches:
        if "fp8-forward" in selected:
            module, output = layout.forward("nvfp4_qk_fp8_pv", batch)
            specs.append(
                BuildSpec(
                    name=f"fp8-forward-b{batch}",
                    group="fp8-forward",
                    batch=batch,
                    cwd=ROOT,
                    command=(
                        str(python),
                        str(forward_builder / "build_causal_gqa_fp8pv_forward.py"),
                        "--batch",
                        str(batch),
                        "--sequence",
                        str(SEQUENCE),
                        "--q-heads",
                        str(Q_HEADS),
                        "--kv-heads",
                        str(KV_HEADS),
                        "--head-dim",
                        str(head_dim),
                        "--gpu",
                        "B200",
                        "--probability-policy",
                        "exact",
                        "--jobs",
                        str(jobs),
                        "--nvcc-threads",
                        str(nvcc_threads),
                        "--nvcc-split-compile",
                        str(nvcc_split_compile),
                        "--module",
                        module,
                        "--output",
                        str(output),
                    ),
                    output=output,
                    module=module,
                )
            )

        if "mx-forward" in selected:
            module, output = layout.forward("nvfp4_qk_mxfp4_pv", batch)
            if profile == D64_BUILD_PROFILE:
                builder = forward_builder / "build_causal_gqa_mxfp4pv_forward.py"
                mx_options = (
                    "--head-dim",
                    "64",
                    "--mx-policy",
                    "d4q01",
                    "--variant",
                    "anchored",
                )
            else:
                builder = forward_builder / "build_causal_gqa_d128_mxfp4pv_forward.py"
                mx_options = (
                    "--anchor-variant",
                    "anchor32",
                    "--saved-lse-denom",
                    "represented",
                )
            specs.append(
                BuildSpec(
                    name=f"mx-forward-b{batch}",
                    group="mx-forward",
                    batch=batch,
                    cwd=ROOT,
                    command=(
                        str(python),
                        str(builder),
                        "--batch",
                        str(batch),
                        "--sequence",
                        str(SEQUENCE),
                        "--q-heads",
                        str(Q_HEADS),
                        "--kv-heads",
                        str(KV_HEADS),
                        "--gpu",
                        "B200",
                        "--num-sm",
                        str(NUM_SM),
                        *mx_options,
                        "--jobs",
                        str(jobs),
                        "--nvcc-threads",
                        str(nvcc_threads),
                        "--nvcc-split-compile",
                        str(nvcc_split_compile),
                        "--module",
                        module,
                        "--output",
                        str(output),
                    ),
                    output=output,
                    module=module,
                )
            )

        if "v509-backward" in selected:
            module, output = layout.backward(batch)
            makefile = "Makefile.v509" if batch == 1 else f"Makefile.v509_b{batch}"
            specs.append(
                BuildSpec(
                    name=f"v509-backward-b{batch}",
                    group="v509-backward",
                    batch=batch,
                    cwd=ROOT / "tk_fa4" / "native_gqa_tk_bwd",
                    command=(
                        "make",
                        "-B",
                        "-f",
                        makefile,
                        f"-j{jobs}",
                        "GPU=B200",
                        f"CUDA_HOME={cuda_home}",
                        f"NVCC={nvcc}",
                        f"BUILD_DIR={output.parent}",
                    ),
                    output=output,
                    module=module,
                )
            )

    if "v416-backward" in selected:
        module, output = layout.native_backward(16)
        specs.append(
            BuildSpec(
                name="v416-backward-b16",
                group="v416-backward",
                batch=16,
                cwd=ROOT / "tk_fa4" / "native_gqa_tk_bwd",
                command=(
                    "make",
                    "-B",
                    "-f",
                    "Makefile.v416",
                    f"-j{jobs}",
                    "GPU=B200",
                    f"CUDA_HOME={cuda_home}",
                    f"NVCC={nvcc}",
                    f"BUILD_DIR={output.parent}",
                ),
                output=output,
                module=module,
            )
        )

    return layout, tuple(specs)


def _run_output(command: Sequence[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BuildError(f"command failed ({shlex.join(command)}): {detail}")
    return result.stdout.strip()


def verify_submodules(root: Path = ROOT) -> None:
    """Require both the recorded gitlink and checkout to match each pin."""
    problems: list[str] = []
    for relative, expected in EXPECTED_SUBMODULES.items():
        stage = _run_output(["git", "ls-files", "--stage", "--", relative], cwd=root)
        fields = stage.split()
        if len(fields) < 4 or fields[0] != "160000":
            problems.append(f"{relative}: missing gitlink")
            continue
        if fields[1] != expected:
            problems.append(f"{relative}: gitlink {fields[1]} != {expected}")
        path = root / relative
        try:
            head = _run_output(["git", "rev-parse", "HEAD"], cwd=path)
        except (BuildError, FileNotFoundError) as exc:
            problems.append(f"{relative}: uninitialized ({exc})")
            continue
        if head != expected:
            problems.append(f"{relative}: checkout {head} != {expected}")
        status = _run_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=path
        )
        if status:
            problems.append(f"{relative}: checkout is dirty")
    if problems:
        raise BuildError("submodule verification failed:\n  " + "\n  ".join(problems))


def _cutlass_dsl_native(root: Path) -> Path:
    if not root.is_absolute():
        raise BuildError(f"CUTLASS DSL root must be absolute: {root}")
    package_init = root / "cutlass" / "__init__.py"
    if not package_init.is_file():
        raise BuildError(
            "CUTLASS DSL root must contain cutlass/__init__.py: " f"{root}"
        )
    version_match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        package_init.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    observed_version = version_match.group(1) if version_match else None
    if observed_version != EXPECTED_CUTLASS_DSL:
        raise BuildError(
            f"CUTLASS DSL {observed_version!r} != {EXPECTED_CUTLASS_DSL!r}"
        )
    native_directory = root / "cutlass" / "_mlir" / "_mlir_libs"
    native = tuple(native_directory.glob("_cutlass_ir*.so"))
    if len(native) != 1:
        raise BuildError(
            "CUTLASS DSL root must contain exactly one "
            "cutlass/_mlir/_mlir_libs/_cutlass_ir*.so; "
            f"found {len(native)} below {native_directory}"
        )
    return native[0].resolve()


def verify_environment(
    cuda_home: Path,
    cutlass_dsl_root: Path | None,
    *,
    operator_only: bool,
) -> None:
    """Fail closed unless the measured GB200 software stack is active."""
    problems: list[str] = []
    if sys.version_info[:2] != EXPECTED_PYTHON:
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} != 3.12"
        )
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"missing Python distribution {distribution}=={expected}")
            continue
        if observed != expected:
            problems.append(f"{distribution} {observed} != {expected}")

    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        problems.append(f"CUDA compiler does not exist: {nvcc}")
    else:
        try:
            nvcc_version = _run_output([str(nvcc), "--version"])
        except BuildError as exc:
            problems.append(str(exc))
        else:
            release = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc_version)
            observed = release.group(1) if release else None
            if observed != EXPECTED_CUDA_RELEASE:
                problems.append(
                    f"CUDA compiler release {observed!r} != {EXPECTED_CUDA_RELEASE}"
                )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        problems.append("nvidia-smi is unavailable")
    else:
        try:
            rows = _run_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ]
            ).splitlines()
        except BuildError as exc:
            problems.append(str(exc))
        else:
            if not rows:
                problems.append("nvidia-smi reported no GPUs")
            for index, row in enumerate(rows):
                fields = [field.strip() for field in row.split(",")]
                if len(fields) != 3:
                    problems.append(f"GPU {index}: malformed nvidia-smi row {row!r}")
                    continue
                name, driver, capability = fields
                if "GB200" not in name:
                    problems.append(f"GPU {index}: {name!r} is not NVIDIA GB200")
                if driver != EXPECTED_DRIVER:
                    problems.append(
                        f"GPU {index}: driver {driver} != {EXPECTED_DRIVER}"
                    )
                if capability != "10.0":
                    problems.append(
                        f"GPU {index}: compute capability {capability} != 10.0"
                    )

    if cutlass_dsl_root is None:
        if not operator_only:
            problems.append("training environment requires --cutlass-dsl-root")
    else:
        try:
            _cutlass_dsl_native(cutlass_dsl_root)
        except BuildError as exc:
            problems.append(str(exc))
    for executable in ("make", "patch"):
        if shutil.which(executable) is None:
            problems.append(f"required executable is unavailable: {executable}")
    if problems:
        raise BuildError("environment verification failed:\n  " + "\n  ".join(problems))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_is_in_scope(path: Path, *, tree: Path) -> bool:
    relative = path.relative_to(tree)
    if any(
        part in _SOURCE_EXCLUDED_DIRECTORIES or part.startswith(".causal_")
        for part in relative.parts[:-1]
    ):
        return False
    return not any(path.name.endswith(suffix) for suffix in _SOURCE_EXCLUDED_SUFFIXES)


def _resolve_source_path(path: Path, *, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BuildError(f"source closure path cannot be resolved: {path}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BuildError(f"source closure path escapes its root: {path}") from error
    return resolved


def _source_closure_paths(root: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for relative_tree in SOURCE_CLOSURE_TREES:
        tree = root / relative_tree
        tree_target = _resolve_source_path(tree, root=root)
        if not stat.S_ISDIR(tree_target.stat().st_mode):
            raise BuildError(f"source-closure tree is missing: {tree}")
        for path in tree.rglob("*"):
            target = _resolve_source_path(path, root=root)
            target_mode = target.stat().st_mode
            if stat.S_ISDIR(target_mode) or not _source_file_is_in_scope(
                path, tree=tree
            ):
                continue
            if not stat.S_ISREG(target_mode):
                raise BuildError(f"source closure requires a regular file: {path}")
            paths.add(path)
    for relative_file in SOURCE_CLOSURE_FILES:
        path = root / relative_file
        target = _resolve_source_path(path, root=root)
        if not stat.S_ISREG(target.stat().st_mode):
            raise BuildError(f"source-closure file is missing: {path}")
        paths.add(path)
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def _source_closure_digest(files: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in files:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cutlass_dsl_source_closure(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    package_root = root / "cutlass"
    paths = tuple(
        sorted(
            (
                path
                for path in package_root.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.relative_to(package_root).parts
                and path.suffix != ".pyc"
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not paths:
        raise BuildError(f"CUTLASS DSL package has no source files: {package_root}")
    symlinks = tuple(path for path in package_root.rglob("*") if path.is_symlink())
    if symlinks:
        raise BuildError(f"CUTLASS DSL closure refuses symlink: {symlinks[0]}")
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    return {
        "algorithm": SOURCE_CLOSURE_ALGORITHM,
        "sha256": _source_closure_digest(records),
        "file_count": len(records),
        "files": records,
    }


def _git_identity(root: Path) -> dict[str, object]:
    status = _run_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
    )
    return {
        "head": _run_output(["git", "rev-parse", "HEAD"], cwd=root),
        "head_tree": _run_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root),
        "dirty": bool(status),
    }


def capture_release_source_identity(root: Path = ROOT) -> dict[str, object]:
    """Capture the exact materialized source used by one artifact build.

    Git metadata is provenance, while the content closure is the executable
    identity.  This permits a dirty development checkout without pretending
    that ``HEAD`` contains local source edits.
    """
    root = root.resolve(strict=True)
    file_records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _source_closure_paths(root)
    ]
    submodules: dict[str, object] = {}
    for relative in sorted(EXPECTED_SUBMODULES):
        submodule_root = root / relative
        submodules[relative] = _git_identity(submodule_root)
    return {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "root": str(root),
        "git": _git_identity(root),
        "submodules": submodules,
        "closure": {
            "algorithm": SOURCE_CLOSURE_ALGORITHM,
            "scope": {
                "trees": list(SOURCE_CLOSURE_TREES),
                "files": list(SOURCE_CLOSURE_FILES),
                "excluded_directories": sorted(_SOURCE_EXCLUDED_DIRECTORIES),
                "excluded_suffixes": sorted(_SOURCE_EXCLUDED_SUFFIXES),
            },
            "sha256": _source_closure_digest(file_records),
            "file_count": len(file_records),
            "files": file_records,
        },
    }


def _file_identity(path: Path, module: str | None = None) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise BuildError(f"artifact is missing or empty: {path}")
    identity: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if module is not None:
        if not module.isidentifier():
            raise BuildError(f"extension module is not an identifier: {module}")
        identity["module"] = module
    return identity


def _runtime_sources(
    cutlass_dsl_root: Path | None,
    *,
    operator_only: bool,
) -> tuple[object, object, object]:
    if operator_only:
        return None, None, None
    if cutlass_dsl_root is None:
        raise BuildError("training manifests require --cutlass-dsl-root")
    runtime = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
    interface = ROOT / "flash-attention" / "flash_attn" / "cute" / "interface.py"
    native = _cutlass_dsl_native(cutlass_dsl_root)
    cutlass_record = {
        "root": str(cutlass_dsl_root.resolve()),
        "version": EXPECTED_CUTLASS_DSL,
        "native": _file_identity(native),
        "closure": _cutlass_dsl_source_closure(cutlass_dsl_root),
    }
    return _file_identity(runtime), _file_identity(interface), cutlass_record


def _manifest_sources(
    runtime_source: object,
    flash_interface: object,
    cutlass_dsl: object,
    release_source: object,
) -> dict[str, object]:
    return {
        "release": release_source,
        "runtime_source": runtime_source,
        "flash_interface": flash_interface,
        "cutlass_dsl": cutlass_dsl,
    }


def _manifest_profile(layout: BuildLayout, batch: int) -> dict[str, str]:
    if layout.profile == D64_BUILD_PROFILE:
        if batch != 16:
            raise BuildError("llama1p2b-d64-b16 manifests require batch 16")
        return {
            "name": D64_BUILD_PROFILE,
            "model_preset": "llama3.2-1b",
            "runtime_contract": "d64-b16-native-v416",
            "fp8_pv_forward": "exact",
            "mxfp4_pv_forward": "d4q01-anchored",
            "projection_publisher": "b300-lowp-bwd",
            "native_backward": "v416",
        }
    if batch not in BATCHES:
        raise BuildError(f"llama8b-d128 manifests do not support batch {batch}")
    return {
        "name": f"llama8b-d128-b{batch}",
        "model_preset": "llama3.1-8b",
        "runtime_contract": f"d128-b{batch}-native-v509",
        "fp8_pv_forward": "exact",
        "mxfp4_pv_forward": "maxsafe-anchor32-represented",
        "projection_publisher": "b300-lowp-bwd",
        "native_backward": "v509",
    }


def write_manifests(
    layout: BuildLayout,
    *,
    batches: Iterable[int],
    cutlass_dsl_root: Path | None,
    operator_only: bool,
    release_source: object | None = None,
) -> tuple[Path, ...]:
    """Write canonical manifests for BF16 and every complete low-P route."""
    runtime, interface, cutlass_dsl = _runtime_sources(
        cutlass_dsl_root, operator_only=operator_only
    )
    if release_source is None:
        release_source = capture_release_source_identity()
    sources = _manifest_sources(runtime, interface, cutlass_dsl, release_source)
    layout.manifest_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest_routes = (
        (
            "bf16_fa4",
            "e4m3_proj_nvfp4_qk_fp8_pv",
            "e4m3_proj_nvfp4_qk_mxfp4_pv",
        )
        if layout.profile == D64_BUILD_PROFILE
        else tuple(ROUTES)
    )
    for batch in batches:
        backward_module, backward_path = layout.native_backward(batch)
        profile = _manifest_profile(layout, batch)
        for route in manifest_routes:
            route_formats = ROUTES[route]
            if route == "bf16_fa4":
                artifacts = {
                    "forward": None,
                    "projection_publisher": None,
                    "native_backward": None,
                }
            else:
                forward_module, forward_path = layout.forward(route, batch)
                required = (forward_path, layout.publisher, backward_path)
                if not all(
                    path.is_file() and path.stat().st_size > 0 for path in required
                ):
                    continue
                artifacts = {
                    "forward": _file_identity(forward_path, forward_module),
                    "projection_publisher": _file_identity(
                        layout.publisher, layout.publisher_module
                    ),
                    "native_backward": _file_identity(backward_path, backward_module),
                }
            purpose = "operator_only" if operator_only else "training"
            if route != "bf16_fa4" and batch == 2:
                purpose = "operator_only"
            manifest = {
                "schema": ARTIFACT_MANIFEST_SCHEMA,
                "purpose": purpose,
                "profile": profile,
                "route": {"name": route, **route_formats},
                "shape": {
                    "batch": batch,
                    "sequence": SEQUENCE,
                    "q_heads": Q_HEADS,
                    "kv_heads": KV_HEADS,
                    "head_dim": 64 if layout.profile == D64_BUILD_PROFILE else HEAD_DIM,
                },
                "architecture": {
                    "gpu": "B200",
                    "compute_capability": [10, 0],
                    "cuda_arch": "sm_100a",
                },
                "artifacts": artifacts,
                "sources": sources,
            }
            destination = layout.manifest_root / f"{route}_b{batch}_s4096_sm100.json"
            destination.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(destination)
    return tuple(written)


def execute_plan(specs: Sequence[BuildSpec], *, cuda_home: Path) -> None:
    """Run a plan without overwriting an existing compiled artifact."""
    existing = [spec.output for spec in specs if spec.output.exists()]
    if existing:
        raise BuildError(
            "clean build refused because outputs already exist; choose a new "
            "--build-root:\n  " + "\n  ".join(str(path) for path in existing)
        )
    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(cuda_home.resolve())
    environment["PYTHONNOUSERSITE"] = "1"
    interpreter_bin = str(Path(os.path.abspath(sys.executable)).parent)
    environment["PATH"] = interpreter_bin + os.pathsep + environment.get("PATH", "")
    for spec in specs:
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"BUILD {spec.name}", flush=True)
        print(f"  cwd: {spec.cwd}", flush=True)
        print(f"  cmd: {shlex.join(spec.command)}", flush=True)
        subprocess.run(spec.command, cwd=spec.cwd, env=environment, check=True)
        if not spec.output.is_file() or spec.output.stat().st_size <= 0:
            raise BuildError(f"build did not create {spec.output}")


def require_clean_build_root(build_root: Path) -> None:
    """Reject stale files so manifests cannot combine unrelated binaries."""
    if not build_root.exists():
        return
    if not build_root.is_dir():
        raise BuildError(f"build root is not a directory: {build_root}")
    if next(build_root.iterdir(), None) is not None:
        raise BuildError(
            "clean build requires a new or empty --build-root: " f"{build_root}"
        )


def _select_targets(raw_targets: Sequence[str]) -> tuple[str, ...]:
    if "all" in raw_targets:
        if len(raw_targets) != 1:
            raise BuildError("target 'all' cannot be combined with another target")
        return TARGET_GROUPS
    return tuple(dict.fromkeys(raw_targets))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "verify", "build"))
    parser.add_argument(
        "--profile",
        choices=BUILD_PROFILES,
        default=D128_BUILD_PROFILE,
        help="fixed model/shape/native-backward build contract",
    )
    parser.add_argument(
        "--build-root",
        required=True,
        help="absolute, user-owned directory for every generated artifact",
    )
    parser.add_argument(
        "--cuda-home",
        required=True,
        help="absolute CUDA 13.0 toolkit root containing bin/nvcc",
    )
    parser.add_argument(
        "--cutlass-dsl-root",
        help=(
            "absolute CUTLASS DSL Python-package root containing "
            "cutlass/__init__.py; required for training manifests"
        ),
    )
    parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        choices=("all", *TARGET_GROUPS),
        default=None,
    )
    parser.add_argument(
        "--batch",
        dest="batches",
        action="append",
        type=int,
        choices=tuple(dict.fromkeys((*BATCHES, *D64_BATCHES))),
        default=None,
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--nvcc-threads", type=int, default=1)
    parser.add_argument("--nvcc-split-compile", type=int, default=1)
    parser.add_argument(
        "--operator-only",
        action="store_true",
        help="emit manifests without training runtime identities",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_root = _absolute_path(args.build_root, "--build-root")
        cuda_home = _absolute_path(args.cuda_home, "--cuda-home")
        cutlass_dsl_root = (
            None
            if args.cutlass_dsl_root is None
            else _absolute_path(args.cutlass_dsl_root, "--cutlass-dsl-root")
        )
        if not args.operator_only and cutlass_dsl_root is None:
            raise BuildError(
                "--cutlass-dsl-root is required unless --operator-only is used"
            )
        profile_targets = (
            D64_TARGET_GROUPS
            if args.profile == D64_BUILD_PROFILE
            else D128_TARGET_GROUPS
        )
        profile_batches = D64_BATCHES if args.profile == D64_BUILD_PROFILE else BATCHES
        if args.targets is None or tuple(args.targets) == ("all",):
            targets = profile_targets
        else:
            targets = _select_targets(args.targets)
        batches = tuple(dict.fromkeys(args.batches or profile_batches))
        layout, specs = build_plan(
            build_root=build_root,
            cuda_home=cuda_home,
            python=Path(os.path.abspath(sys.executable)),
            batches=batches,
            targets=targets,
            profile=args.profile,
            jobs=args.jobs,
            nvcc_threads=args.nvcc_threads,
            nvcc_split_compile=args.nvcc_split_compile,
        )
        verify_submodules()
        if args.action in {"verify", "build"}:
            verify_environment(
                cuda_home,
                cutlass_dsl_root,
                operator_only=args.operator_only,
            )
        if args.action == "plan":
            for spec in specs:
                print(f"PLAN {spec.name}")
                print(f"  cwd: {spec.cwd}")
                print(f"  cmd: {shlex.join(spec.command)}")
                print(f"  output: {spec.output}")
            return 0
        if args.action == "verify":
            print("PASS: pinned submodules and measured GB200 environment verified")
            return 0

        require_clean_build_root(build_root)
        source_before = capture_release_source_identity()
        execute_plan(specs, cuda_home=cuda_home)
        source_after = capture_release_source_identity()
        if source_after["closure"] != source_before["closure"]:
            raise BuildError(
                "release source closure changed during compilation; discard "
                "this build root and rebuild from one stable source state"
            )
        manifests = write_manifests(
            layout,
            batches=batches,
            cutlass_dsl_root=cutlass_dsl_root,
            operator_only=args.operator_only,
            release_source=source_after,
        )
        print("MANIFESTS")
        for path in manifests:
            print(path)
        return 0
    except (BuildError, OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
