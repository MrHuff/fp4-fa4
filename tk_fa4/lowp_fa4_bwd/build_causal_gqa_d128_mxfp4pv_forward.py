#!/usr/bin/env python3
"""Build an allowlisted causal D128 NVFP4-QK/MXFP4-PV forward route."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FORWARD_SOURCE = REPO_ROOT / "tk_fa4" / "fp4_fa4_fwd"

ANCHOR_VARIANTS = {
    "anchor32": {
        "mx_global_anchor32": True,
        "mx_global_anchor128": False,
        "mx_global_anchor_margin_log2": 64,
    },
    "anchor128-m0": {
        "mx_global_anchor32": False,
        "mx_global_anchor128": True,
        "mx_global_anchor_margin_log2": 0,
    },
    "anchor128-m64": {
        "mx_global_anchor32": False,
        "mx_global_anchor128": True,
        "mx_global_anchor_margin_log2": 64,
    },
}

SAVED_LSE_DENOM_POLICIES = {
    "represented": {
        "mx_full_approx_denom": False,
        "mx_full_approx_denom_mode": 0,
        "mx_dual_lse_denom": False,
        "mx_stable_lse_logsum": False,
        "mx_alternate_lse_stat": False,
        "mx_causal_q3_progressive_reuse": True,
    },
    "full-approx-mode1": {
        "mx_full_approx_denom": True,
        "mx_full_approx_denom_mode": 1,
        "mx_dual_lse_denom": True,
        "mx_stable_lse_logsum": False,
        "mx_alternate_lse_stat": True,
        "mx_causal_q3_progressive_reuse": False,
    },
    "stable-represented-logsum": {
        "mx_full_approx_denom": False,
        "mx_full_approx_denom_mode": 0,
        "mx_dual_lse_denom": False,
        "mx_stable_lse_logsum": True,
        "mx_alternate_lse_stat": True,
        "mx_causal_q3_progressive_reuse": True,
    },
    "stable-full-approx-mode1": {
        "mx_full_approx_denom": True,
        "mx_full_approx_denom_mode": 1,
        "mx_dual_lse_denom": True,
        "mx_stable_lse_logsum": True,
        "mx_alternate_lse_stat": True,
        "mx_causal_q3_progressive_reuse": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_identities(
    forward_source: Path = FORWARD_SOURCE,
) -> dict[str, dict[str, int | str]]:
    sources = [Path(__file__).resolve()]
    sources.extend(
        path
        for path in forward_source.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".causal_")
            for part in path.relative_to(forward_source).parts
        )
        and (
            path.name.startswith("Makefile")
            or path.suffix in {".cu", ".cuh", ".h", ".inc", ".py"}
        )
    )
    canonical_forward_root = FORWARD_SOURCE.relative_to(REPO_ROOT)
    return {
        str(
            path.relative_to(REPO_ROOT)
            if path == Path(__file__).resolve()
            else canonical_forward_root / path.relative_to(forward_source)
        ): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(set(sources))
    }


def _load_topology(path: Path, module_name: str) -> dict[str, object]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import built extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.read_hao_direct_topology())


def _require_requested_topology(
    topology: dict[str, object],
    *,
    args: argparse.Namespace,
) -> None:
    anchor = ANCHOR_VARIANTS[args.anchor_variant]
    saved_lse = SAVED_LSE_DENOM_POLICIES[args.saved_lse_denom]
    expected = {
        "batch": args.batch,
        "seqlen": args.sequence,
        "heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "dqk": 128,
        "dvo": 128,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "pv_format": "mxfp4_e8m0_block32",
        "route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
        "fixed_route_fastpath": True,
        "causal_interleaved_kv": False,
        "mx_scale_select": 4,
        "mx_stored_scale_shift_log2": 32,
        "mx_anchor_affine_hoist": False,
        **anchor,
        **saved_lse,
    }
    for key, value in expected.items():
        actual = topology.get(key)
        if actual != value or type(actual) is not type(value):
            raise RuntimeError(
                f"built topology {key}={actual!r}, expected {value!r}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--gpu", choices=("B200", "B300"), default="B200")
    parser.add_argument(
        "--anchor-variant",
        choices=tuple(ANCHOR_VARIANTS),
        default="anchor32",
        help=(
            "allowlisted row-anchor policy; anchor128 variants are "
            "diagnostic candidates until separately authenticated"
        ),
    )
    parser.add_argument(
        "--saved-lse-denom",
        choices=tuple(SAVED_LSE_DENOM_POLICIES),
        default="represented",
        help=(
            "saved-LSE denominator policy; alternate policies retain the "
            "represented MX denominator for O and publish either the "
            "pre-quantization statistic or a stable log-domain sum for "
            "backward"
        ),
    )
    parser.add_argument("--num-sm", type=int, default=152)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--nvcc-threads", type=int, default=1)
    parser.add_argument("--nvcc-split-compile", type=int, default=1)
    parser.add_argument("--module")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    if args.sequence <= 0 or args.sequence % 256:
        parser.error("--sequence must be positive and divisible by 256")
    if args.kv_heads <= 0:
        parser.error("--kv-heads must be positive")
    if args.q_heads <= 0 or args.q_heads % args.kv_heads:
        parser.error(
            "--q-heads must be positive and divisible by --kv-heads"
        )
    if args.num_sm <= 0:
        parser.error("--num-sm must be positive")
    for option, value in (
        ("--jobs", args.jobs),
        ("--nvcc-threads", args.nvcc_threads),
        ("--nvcc-split-compile", args.nvcc_split_compile),
    ):
        if value <= 0:
            parser.error(f"{option} must be positive")
    if args.batch in (2, 4) and (
        args.sequence,
        args.q_heads,
        args.kv_heads,
    ) != (4096, 32, 8):
        parser.error(
            "B2/B4 are restricted to --sequence 4096 --q-heads 32 "
            "--kv-heads 8"
        )
    if (
        args.saved_lse_denom in (
            "full-approx-mode1",
            "stable-full-approx-mode1",
        )
        and args.anchor_variant != "anchor128-m64"
    ):
        parser.error(
            "full-approx saved-LSE policies are initially allowlisted only with "
            "--anchor-variant anchor128-m64"
        )
    if (
        args.saved_lse_denom == "stable-represented-logsum"
        and args.anchor_variant != "anchor128-m64"
    ):
        parser.error(
            "stable-represented-logsum is initially allowlisted only with "
            "--anchor-variant anchor128-m64"
        )
    if (
        args.saved_lse_denom.startswith("stable-")
        and args.gpu != "B200"
    ):
        parser.error(
            "stable saved-LSE logsum currently requires the B200 "
            "four-carrier deferred-denominator path"
        )
    return args


def main() -> None:
    args = _parse_args()
    live_sources_before = _source_identities()
    shape = (
        f"b{args.batch}s{args.sequence}h{args.q_heads}"
        f"kv{args.kv_heads}d128"
    )
    target = f"{args.gpu.lower()}_sm{args.num_sm}"
    anchor_tag = args.anchor_variant.replace("-", "_")
    lse_tag = args.saved_lse_denom.replace("-", "_")
    module = args.module or (
        f"_C_d128_mx_maxsafe_{anchor_tag}_{lse_tag}_{shape}_{target}"
    )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python extension suffix is unavailable")
    output = (args.output or Path("/tmp") / f"{module}{suffix}").resolve()
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing forward artifact or manifest: "
            f"{output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    symbol_tag = f"d128_mx_maxsafe_{anchor_tag}_{lse_tag}_{shape}"
    workdir = Path(
        tempfile.mkdtemp(
            prefix=".causal_gqa_d128_mxfp4pv_build_",
            dir=FORWARD_SOURCE.parent,
        )
    )
    try:
        shutil.rmtree(workdir)
        shutil.copytree(FORWARD_SOURCE, workdir)
        build_sources = _source_identities(workdir)
        if build_sources != live_sources_before:
            raise RuntimeError(
                "forward sources changed while creating the isolated build "
                "snapshot"
            )
        command = [
            "make",
            "-B",
            "-f",
            "Makefile.hao_direct_fp4pv",
            f"-j{args.jobs}",
            f"GPU={args.gpu}",
            f"HAO_BATCH={args.batch}",
            f"HAO_SEQ_LEN={args.sequence}",
            f"HAO_HEADS={args.q_heads}",
            f"HAO_KV_HEADS={args.kv_heads}",
            "HAO_HEAD_DIM=128",
            f"HAO_NUM_SM={args.num_sm}",
            "HAO_CAUSAL=1",
            "HAO_FIXED_ROUTE_FASTPATH=1",
            "HAO_CAUSAL_INTERLEAVED_KV=0",
            "HAO_FP4PV_MX_POLICY=causal-accurate",
            "HAO_FP4PV_MX_ANCHOR_AFFINE_HOIST_OVERRIDE=0",
            (
                "HAO_FP4PV_MX_GLOBAL_ANCHOR32_OVERRIDE="
                f"{int(ANCHOR_VARIANTS[args.anchor_variant]['mx_global_anchor32'])}"
            ),
            (
                "HAO_FP4PV_MX_GLOBAL_ANCHOR128_OVERRIDE="
                f"{int(ANCHOR_VARIANTS[args.anchor_variant]['mx_global_anchor128'])}"
            ),
            (
                "HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2_OVERRIDE="
                f"{ANCHOR_VARIANTS[args.anchor_variant]['mx_global_anchor_margin_log2']}"
            ),
            (
                "HAO_FP4PV_MX_FULL_APPROX_DENOM_OVERRIDE="
                f"{SAVED_LSE_DENOM_POLICIES[args.saved_lse_denom]['mx_full_approx_denom_mode']}"
            ),
            (
                "HAO_FP4PV_MX_STABLE_LSE_LOGSUM_OVERRIDE="
                f"{int(SAVED_LSE_DENOM_POLICIES[args.saved_lse_denom]['mx_stable_lse_logsum'])}"
            ),
            (
                "HAO_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE_OVERRIDE="
                f"{int(SAVED_LSE_DENOM_POLICIES[args.saved_lse_denom]['mx_causal_q3_progressive_reuse'])}"
            ),
            "HAO_EXTENSION_SYMBOLIC_BIND=1",
            f"HAO_KERNEL_SYMBOL_TAG={symbol_tag}",
            f"MODULE={module}",
            f"OUT={output}",
            f"NVCC_THREADS={args.nvcc_threads}",
            f"NVCC_SPLIT_COMPILE={args.nvcc_split_compile}",
        ]
        subprocess.run(command, cwd=workdir, check=True)
        if _source_identities(workdir) != build_sources:
            raise RuntimeError("isolated forward sources changed during build")
        if _source_identities() != live_sources_before:
            raise RuntimeError(
                "repository forward sources changed during build"
            )
        output_identity = {
            "path": str(output),
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
        }
        git_diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        topology = _load_topology(output, module)
        _require_requested_topology(topology, args=args)
        manifest = {
            "schema": "causal_gqa_d128_mxfp4pv_forward_build_v2",
            "configuration": {
                "batch": args.batch,
                "sequence": args.sequence,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": 128,
                "gpu_target": args.gpu,
                "num_sm": args.num_sm,
                "anchor_variant": args.anchor_variant,
                "saved_lse_denom": args.saved_lse_denom,
                "jobs": args.jobs,
                "nvcc_threads": args.nvcc_threads,
                "nvcc_split_compile": args.nvcc_split_compile,
                "module": module,
            },
            "command": command,
            "output": output_identity,
            "topology": topology,
            "repository": {
                "head": _command_output(["git", "rev-parse", "HEAD"]),
                "status_porcelain_v1": _command_output(
                    ["git", "status", "--porcelain=v1"]
                ).splitlines(),
                "tracked_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
                "tracked_diff_bytes": len(git_diff),
            },
            "toolchain": {
                "nvcc_version": _command_output(
                    ["/usr/local/cuda/bin/nvcc", "--version"]
                ),
                "make_version": _command_output(["make", "--version"])
                .splitlines()[0],
            },
            "sources": build_sources,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "artifact": output_identity,
                    "manifest": str(manifest_path),
                },
                sort_keys=True,
            )
        )
    finally:
        if args.keep_workdir:
            print(f"kept build directory: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
