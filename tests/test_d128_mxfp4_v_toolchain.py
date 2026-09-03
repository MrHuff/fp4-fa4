from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tk_fa4.lowp_fa4_bwd.cutlass_dsl_toolchain as toolchain


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "tune_d64_gqa_cute.py"
SATURATED = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_saturated.py"
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _synthetic_toolchain(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    site_root = tmp_path / "site-packages"
    origin = (
        site_root
        / "nvidia_cutlass_dsl"
        / "python_packages"
        / "cutlass"
        / "__init__.py"
    )
    native = origin.parent / "_mlir" / "_mlir_libs" / (
        "_cutlass_ir.cpython-312-aarch64-linux-gnu.so"
    )
    runtime = (
        site_root / "nvidia_cutlass_dsl" / "lib" / "libcute_dsl_runtime.so"
    )
    static = runtime.with_name("libcuda_dialect_runtime_static.a")
    pth = site_root / "nvidia_cutlass_dsl.pth"
    _write(origin, b"synthetic cutlass package\n")
    _write(native, b"synthetic llvm21 cutlass ir\n")
    _write(runtime, b"synthetic cute runtime\n")
    _write(static, b"synthetic cuda dialect archive\n")
    _write(pth, b"nvidia_cutlass_dsl/python_packages\n")
    # These files must never affect the authenticated source payload.
    _write(origin.parent / "__pycache__" / "ignored.pyc", b"cache")

    manifest, _ = toolchain.load_d128_mxfp4_v_toolchain_manifest()
    manifest = copy.deepcopy(manifest)
    summary, identities = toolchain.summarize_cutlass_payload(
        site_root,
        manifest["payload"]["root_entries"],
    )
    manifest["payload"].update(summary)
    for artifact in manifest["native_artifacts"].values():
        observed = identities[artifact["relative_path"]]
        artifact["sha256"] = observed["sha256"]
        artifact["bytes"] = observed["bytes"]
    manifest_path = tmp_path / "synthetic-toolchain.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return (
        manifest_path,
        origin,
        runtime,
        {"CUTE_DSL_LIBS": str(runtime.resolve())},
    )


def test_checked_in_manifest_pins_llvm21_payload_and_supply_inputs() -> None:
    manifest, identity = toolchain.load_d128_mxfp4_v_toolchain_manifest()

    assert identity["sha256"] == toolchain.DEFAULT_MANIFEST_SHA256
    assert identity["bytes"] == toolchain.DEFAULT_MANIFEST_BYTES
    assert manifest["payload"] == {
        "hash_schema": "sha256sum_site_relative_path_lines_v1",
        "root_entries": ["nvidia_cutlass_dsl", "nvidia_cutlass_dsl.pth"],
        "excluded": ["__pycache__", "*.pyc", "*.dist-info"],
        "sha256": (
            "7e128c968b10657225baa544e1e6099a9d69cf4cc95a624a76c505c58d4ca519"
        ),
        "files": 195,
        "bytes": 193994979,
    }
    supply = manifest["wheel_supply_chain"]
    assert supply["install_order"] == ["libs_base", "libs_cu13", "meta"]
    assert [wheel["sha256"] for wheel in supply["wheels"]] == [
        "d2a3c412287e356fbe48fe9f845d6d33cd35dea5e20d7e4f628c20957967cacd",
        "3032405dff28892340f96b467e744a822079cae454dce534fc17b77e85190e42",
        "68ed1b63ca74aae87955012da9dfd7fdaae471329d0028b229b841c7192ccf52",
    ]
    assert (
        manifest["native_artifacts"]["cutlass_ir"]["llvm_commit"]
        == "e57c3673ac82461cd3c8a2e5cf6f8a890705c882"
    )


def test_effective_payload_verification_is_deterministic_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, origin, runtime, environ = _synthetic_toolchain(tmp_path)
    monkeypatch.delitem(sys.modules, "cutlass", raising=False)

    receipt = toolchain.verify_d128_mxfp4_v_toolchain(
        manifest_path=manifest_path,
        cutlass_origin=origin,
        environ=environ,
    )
    assert receipt["schema"] == toolchain.TOOLCHAIN_RECEIPT_SCHEMA
    assert receipt["wheel_provenance_is_runtime_inferred"] is False
    assert receipt["runtime_selection"] == "explicit_authenticated_path"
    assert receipt["payload"]["files"] == 5
    assert receipt["native_artifacts"]["cute_dsl_runtime"]["path"] == str(
        runtime.resolve()
    )

    runtime.write_bytes(b"tampered runtime\n")
    with pytest.raises(RuntimeError, match="payload identity mismatch"):
        toolchain.verify_d128_mxfp4_v_toolchain(
            manifest_path=manifest_path,
            cutlass_origin=origin,
            environ=environ,
        )


def test_candidate_compile_environment_is_exact_and_does_not_replace_drift(
    tmp_path: Path,
) -> None:
    dump = (tmp_path / "rank-3").resolve()
    dump.mkdir()
    environ: dict[str, str] = {}

    receipt = toolchain.configure_d128_mxfp4_v_compile_environment(
        dump,
        environ=environ,
    )
    assert receipt == {
        "CUTE_DSL_ARCH": "sm_100a",
        "CUTE_DSL_KEEP": "ptx,cubin",
        "CUTE_DSL_NO_CACHE": "1",
        "CUTE_DSL_DUMP_DIR": str(dump),
    }
    environ["CUTE_DSL_ARCH"] = "sm_103a"
    with pytest.raises(RuntimeError, match="refusing to replace incompatible"):
        toolchain.configure_d128_mxfp4_v_compile_environment(
            dump,
            environ=environ,
        )


def test_toolchain_verifier_is_called_only_below_the_mx_v_gate() -> None:
    tree = ast.parse(LOADER.read_text())
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_control"
    )
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(loader):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    calls = [
        node
        for node in ast.walk(loader)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "require_d128_mxfp4_v_compile_environment",
            "verify_d128_mxfp4_v_toolchain",
        }
    ]
    assert len(calls) == 2
    for call in calls:
        ancestors: list[ast.AST] = []
        current: ast.AST | None = call
        while current in parents:
            current = parents[current]
            ancestors.append(current)
        assert any(
            isinstance(ancestor, ast.If)
            and isinstance(ancestor.test, ast.Name)
            and ancestor.test.id == "use_d128_mxfp4_v_dp"
            for ancestor in ancestors
        )

    saturated = SATURATED.read_text()
    candidate_block = saturated.split(
        "if args.experimental_d128_mxfp4_v_backward:", 2
    )[-1].split(
        'selected_projection = os.environ.get("TK_FA4_LOWP_BWD_EXTENSION_SOURCE")',
        1,
    )[0]
    assert "configure_d128_mxfp4_v_compile_environment(" in candidate_block


def test_compile_receipt_hashes_retained_ptx_and_cubin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = tmp_path / "generated_control.py"
    generated.write_text("# generated candidate control\n")
    patch = tmp_path / "candidate.patch"
    patch.write_text("candidate patch\n")
    toolchain_receipt = {
        "schema": toolchain.TOOLCHAIN_RECEIPT_SCHEMA,
        "target_arch": "sm_100a",
    }
    monkeypatch.setattr(
        toolchain,
        "verify_d128_mxfp4_v_toolchain",
        lambda: toolchain_receipt,
    )
    monkeypatch.setattr(
        toolchain,
        "require_d128_mxfp4_v_compile_environment",
        lambda: {
            "CUTE_DSL_ARCH": "sm_100a",
            "CUTE_DSL_KEEP": "ptx,cubin",
            "CUTE_DSL_NO_CACHE": "1",
            "CUTE_DSL_DUMP_DIR": str(tmp_path),
        },
    )
    patch_receipt = {
        "path": str(patch),
        "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "bytes": patch.stat().st_size,
    }
    control = SimpleNamespace(
        __file__=str(generated),
        TK_D128_MXFP4_V_DP=True,
        TK_D128_MXFP4_V_TOOLCHAIN_PROVENANCE=toolchain_receipt,
        TK_D128_MXFP4_V_DP_PATCH_PROVENANCE=patch_receipt,
    )
    ptx = "// generated\n.version 8.8\n.target sm_100a\n.address_size 64\n"
    cubin = b"\x7fELF\x02\x01synthetic-cubin"
    compiled = SimpleNamespace(__ptx__=ptx, __cubin__=cubin)
    kernel = SimpleNamespace(
        num_regs_reduce=136,
        num_regs_compute=136,
        num_regs_mma=96,
        num_regs_load=96,
    )

    receipt = toolchain.d128_mxfp4_v_compilation_receipt(
        control=control,
        compiled=compiled,
        kernel=kernel,
    )
    assert receipt["schema"] == toolchain.COMPILE_RECEIPT_SCHEMA
    assert receipt["ptx"]["sha256"] == hashlib.sha256(ptx.encode()).hexdigest()
    assert receipt["cubin"]["sha256"] == hashlib.sha256(cubin).hexdigest()
    assert receipt["registers"] == {
        "reduce": 136,
        "compute": 136,
        "mma": 96,
        "load": 96,
    }

    compiled.__ptx__ = ptx.replace("sm_100a", "sm_103a")
    with pytest.raises(RuntimeError, match="does not declare"):
        toolchain.d128_mxfp4_v_compilation_receipt(
            control=control,
            compiled=compiled,
            kernel=kernel,
        )

