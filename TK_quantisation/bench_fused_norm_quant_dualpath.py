#!/usr/bin/env python3

import argparse
import gc
import importlib
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
V6_DIR = ROOT / "nvfp4_v6"
LOCALCTA_DIR = ROOT / "nvfp4_CTA_local_v2"
DEFAULT_SHAPES = ("128x4096", "512x4096", "2048x4096", "4096x8192")
CUOBJDUMP = Path("/usr/local/cuda/bin/cuobjdump")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and validate naive vs ILP fused RMSNorm+activation+quant kernels."
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=list(DEFAULT_SHAPES),
        help="Shapes as MxK entries. Default: %(default)s",
    )
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--with-silu",
        nargs="+",
        type=int,
        default=[0, 1],
        choices=[0, 1],
        help="Activation settings to benchmark: 0 or 1",
    )
    parser.add_argument(
        "--return-transpose",
        nargs="+",
        type=int,
        default=[0, 1],
        choices=[0, 1],
        help="Transpose settings to benchmark: 0 or 1",
    )
    parser.add_argument(
        "--skip-sass",
        action="store_true",
        help="Skip cuobjdump SASS dump/summary generation.",
    )
    parser.add_argument(
        "--sass-dir",
        type=Path,
        default=ROOT / "sass_dumps",
        help="Directory for full cuobjdump SASS dumps.",
    )
    return parser.parse_args()


def import_extension(module_dir: Path, module_name: str):
    sys.path.insert(0, str(module_dir))
    return importlib.import_module(module_name)


def parse_shape(spec: str) -> tuple[int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid shape {spec!r}, expected MxK")
    return int(parts[0]), int(parts[1])


def raw_max_delta(lhs: torch.Tensor, rhs: torch.Tensor) -> int:
    if lhs.numel() == 0 and rhs.numel() == 0:
        return 0
    lhs_raw = lhs.contiguous().view(torch.uint8).to(torch.int16)
    rhs_raw = rhs.contiguous().view(torch.uint8).to(torch.int16)
    return int((lhs_raw - rhs_raw).abs().max().item())


def raw_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.device != rhs.device:
        return False
    if lhs.numel() == 0 and rhs.numel() == 0:
        return True
    lhs_raw = lhs.contiguous().view(torch.uint8)
    rhs_raw = rhs.contiguous().view(torch.uint8)
    return bool(torch.eq(lhs_raw, rhs_raw).all().item())


def assert_exact(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if raw_equal(lhs, rhs):
        return
    raise AssertionError(f"{name} mismatch (max raw-byte delta={raw_max_delta(lhs, rhs)})")


def assert_close(name: str, lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float) -> None:
    if torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
        return
    max_abs = float((lhs.float() - rhs.float()).abs().max().item())
    raise AssertionError(f"{name} mismatch (max_abs={max_abs:.6g}, atol={atol}, rtol={rtol})")


def build_transformed_reference(
    input_bf16: torch.Tensor,
    gamma_bf16: torch.Tensor,
    inv_rms: torch.Tensor,
    with_silu: bool,
) -> torch.Tensor:
    transformed = input_bf16.float() * inv_rms.view(-1, 1) * gamma_bf16.float().view(1, -1)
    if with_silu:
        transformed = F.silu(transformed)
    return transformed.to(torch.bfloat16).contiguous()


def error_metrics(decoded: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    if not torch.isfinite(decoded.float()).all():
        raise AssertionError("decoded tensor contains non-finite values")
    diff = decoded.float() - reference.float()
    ref_f = reference.float()
    rmse = float(diff.square().mean().sqrt().item())
    ref_rms = float(ref_f.square().mean().sqrt().item())
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": rmse,
        "rel_rmse": rmse / ref_rms if ref_rms > 0.0 else 0.0,
    }


def benchmark_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        del out
    gc.collect()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        del out
    end.record()
    torch.cuda.synchronize()
    gc.collect()
    return start.elapsed_time(end) / max(iters, 1)


def run_v6_case(mod, m: int, k: int, epsilon: float, with_silu: bool, return_transpose: bool) -> dict[str, dict[str, float]]:
    x = torch.randn((m, k), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.randn((k,), device="cuda", dtype=torch.float32).to(torch.bfloat16)

    naive = mod.tk_fused_norm_quantize_naive(x, gamma, epsilon, with_silu, return_transpose)
    ilp = mod.tk_fused_norm_quantize_ilp(x, gamma, epsilon, with_silu, return_transpose)

    labels = ("row_fp4", "row_sc", "col_fp4", "col_sc", "sg", "inv_rms", "amax")
    for idx, label in enumerate(labels):
        assert_exact(f"v6 naive vs ilp {label}", naive[idx], ilp[idx])

    ref_transformed = build_transformed_reference(x, gamma, naive[5], with_silu)
    ref_quant = mod.tk_quantize_for_gemm(ref_transformed, return_transpose, True)
    ref_amax = ref_transformed.float().abs().max().reshape(1)
    ref_sg = (ref_amax / 2688.0).to(torch.float32)

    assert_exact("v6 row_fp4 vs decomposed", naive[0], ref_quant[0])
    assert_exact("v6 row_sc vs decomposed", naive[1], ref_quant[1])
    if return_transpose:
        assert_exact("v6 col_fp4 vs decomposed", naive[2], ref_quant[2])
        assert_exact("v6 col_sc vs decomposed", naive[3], ref_quant[3])
    assert_close("v6 sg vs decomposed", naive[4], ref_sg, atol=1e-6, rtol=1e-6)
    assert_close("v6 amax vs decomposed", naive[6], ref_amax, atol=1e-6, rtol=1e-6)

    ref_inv_rms = torch.rsqrt(x.float().square().mean(dim=1) + epsilon)
    assert_close("v6 inv_rms vs torch", naive[5], ref_inv_rms, atol=2e-4, rtol=2e-4)

    row_recon = mod.tk_reconstruct_row(naive[0], naive[1], naive[4])
    metrics = {"row": error_metrics(row_recon, ref_transformed)}
    if return_transpose:
        col_recon = mod.tk_reconstruct_col(naive[2], naive[3], naive[4])
        metrics["col"] = error_metrics(col_recon, ref_transformed.t().contiguous())
    return metrics


def run_localcta_case(mod, m: int, k: int, epsilon: float, with_silu: bool, return_transpose: bool) -> dict[str, dict[str, float]]:
    x = torch.randn((m, k), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.randn((k,), device="cuda", dtype=torch.float32).to(torch.bfloat16)

    naive = mod.tk_localcta_fused_norm_quantize_naive(x, gamma, epsilon, with_silu, return_transpose)
    ilp = mod.tk_localcta_fused_norm_quantize_ilp(x, gamma, epsilon, with_silu, return_transpose)

    labels = ("row_fp4", "row_sc", "col_fp4", "col_sc", "row_sg", "col_sg", "inv_rms")
    for idx, label in enumerate(labels):
        assert_exact(f"localCTA naive vs ilp {label}", naive[idx], ilp[idx])

    ref_transformed = build_transformed_reference(x, gamma, naive[6], with_silu)
    ref_quant = mod.tk_localcta_quantize_for_gemm(ref_transformed, return_transpose, True)

    assert_exact("localCTA row_fp4 vs decomposed", naive[0], ref_quant[0])
    assert_exact("localCTA row_sc vs decomposed", naive[1], ref_quant[1])
    if return_transpose:
        assert_exact("localCTA col_fp4 vs decomposed", naive[2], ref_quant[2])
        assert_exact("localCTA col_sc vs decomposed", naive[3], ref_quant[3])
    assert_close("localCTA row_sg vs decomposed", naive[4], ref_quant[4], atol=1e-6, rtol=1e-6)
    assert_close("localCTA col_sg vs decomposed", naive[5], ref_quant[5], atol=1e-6, rtol=1e-6)

    ref_inv_rms = torch.rsqrt(x.float().square().mean(dim=1) + epsilon)
    assert_close("localCTA inv_rms vs torch", naive[6], ref_inv_rms, atol=2e-4, rtol=2e-4)

    row_recon = mod.tk_localcta_reconstruct_row(naive[0], naive[1], naive[4])
    metrics = {"row": error_metrics(row_recon, ref_transformed)}
    if return_transpose:
        col_recon = mod.tk_localcta_reconstruct_col(naive[2], naive[3], naive[5])
        metrics["col"] = error_metrics(col_recon, ref_transformed.t().contiguous())
    return metrics


def split_sass_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^\s*Function : (.+)$", text))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[start:end]
    return sections


def find_sass_section(sections: dict[str, str], candidates: list[str]) -> tuple[str | None, str | None]:
    for candidate in candidates:
        for name, body in sections.items():
            if candidate in name:
                return name, body
    return None, None


def summarize_sass_section(body: str) -> dict[str, int]:
    return {
        "xorsign_abs": len(re.findall(r"XORSIGN", body)),
        "fmax": len(re.findall(r"\bFMAX\b", body)),
        "fmnmx": len(re.findall(r"\bFMNMX\b", body)),
        "imnmx": len(re.findall(r"\bIMNMX\b", body)),
    }


def dump_and_summarize_sass(module_name: str, module_file: Path, output_dir: Path) -> None:
    if not CUOBJDUMP.exists():
        print(f"[sass] cuobjdump not found at {CUOBJDUMP}, skipping {module_name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / f"{module_name}.sass.txt"
    text = subprocess.check_output(
        [str(CUOBJDUMP), "--dump-sass", str(module_file)],
        text=True,
        stderr=subprocess.STDOUT,
    )
    dump_path.write_text(text)
    print(f"[sass] wrote {dump_path}")

    sections = split_sass_sections(text)
    targets = {
        "v6_naive": [
            "_ZN5tk_v531persistent_norm_quantize_kernelILb0ELb0ELb0EE",
            "persistent_norm_quantize_kernel<true, true, false>",
            "persistent_norm_quantize_kernel<false, true, false>",
            "persistent_norm_quantize_kernel",
        ],
        "v6_ilp": [
            "_ZN5tk_v531persistent_norm_quantize_kernelILb0ELb0ELb1EE",
            "persistent_norm_quantize_kernel<true, true, true>",
            "persistent_norm_quantize_kernel<false, true, true>",
            "persistent_norm_quantize_kernel",
        ],
        "localcta_naive": [
            "_ZN11tk_localcta35fused_localcta_norm_quantize_kernelILb0ELb0ELb0ELb1EE",
            "fused_localcta_norm_quantize_kernel<true, true, false",
            "fused_localcta_norm_quantize_kernel<false, true, false",
            "fused_localcta_norm_quantize_kernel",
        ],
        "localcta_ilp": [
            "_ZN11tk_localcta35fused_localcta_norm_quantize_kernelILb0ELb0ELb1ELb1EE",
            "fused_localcta_norm_quantize_kernel<true, true, true",
            "fused_localcta_norm_quantize_kernel<false, true, true",
            "fused_localcta_norm_quantize_kernel",
        ],
    }

    for label, candidates in targets.items():
        if not label.startswith(module_name):
            continue
        section_name, body = find_sass_section(sections, candidates)
        if body is None:
            print(f"[sass] {label}: section not found in dump")
            continue
        counts = summarize_sass_section(body)
        print(f"[sass] {label}: section={section_name}")
        print(
            "[sass] "
            f"{label}: XORSIGN={counts['xorsign_abs']} "
            f"FMAX={counts['fmax']} FMNMX={counts['fmnmx']} IMNMX={counts['imnmx']}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this harness.")

    torch.manual_seed(0)
    v6 = import_extension(V6_DIR, "_tk_quant_v6")
    localcta = import_extension(LOCALCTA_DIR, "_tk_quant_localcta_v2")

    if not args.skip_sass:
        dump_and_summarize_sass("v6", Path(v6.__file__), args.sass_dir)
        dump_and_summarize_sass("localcta", Path(localcta.__file__), args.sass_dir)

    shapes = [parse_shape(spec) for spec in args.shapes]

    for module_name, module, runner, naive_name, ilp_name in (
        (
            "v6",
            v6,
            run_v6_case,
            "tk_fused_norm_quantize_naive",
            "tk_fused_norm_quantize_ilp",
        ),
        (
            "localcta",
            localcta,
            run_localcta_case,
            "tk_localcta_fused_norm_quantize_naive",
            "tk_localcta_fused_norm_quantize_ilp",
        ),
    ):
        print(f"\n== {module_name} ==")
        for m, k in shapes:
            for with_silu_i in args.with_silu:
                with_silu = bool(with_silu_i)
                for return_transpose_i in args.return_transpose:
                    return_transpose = bool(return_transpose_i)
                    metrics = runner(module, m, k, args.epsilon, with_silu, return_transpose)

                    x = torch.randn((m, k), device="cuda", dtype=torch.float32).to(torch.bfloat16)
                    gamma = torch.randn((k,), device="cuda", dtype=torch.float32).to(torch.bfloat16)
                    naive_ms = benchmark_ms(
                        lambda: getattr(module, naive_name)(
                            x, gamma, args.epsilon, with_silu, return_transpose
                        ),
                        args.warmup,
                        args.iters,
                    )
                    ilp_ms = benchmark_ms(
                        lambda: getattr(module, ilp_name)(
                            x, gamma, args.epsilon, with_silu, return_transpose
                        ),
                        args.warmup,
                        args.iters,
                    )
                    speedup = naive_ms / ilp_ms if ilp_ms > 0 else float("inf")
                    print(
                        f"{m:5d}x{k:<5d} "
                        f"silu={int(with_silu)} transpose={int(return_transpose)} "
                        f"naive={naive_ms:8.3f} ms ilp={ilp_ms:8.3f} ms speedup={speedup:6.3f}x"
                    )
                    row = metrics["row"]
                    print(
                        " " * 13 +
                        f"row_err max_abs={row['max_abs']:.6f} "
                        f"mean_abs={row['mean_abs']:.6f} "
                        f"rmse={row['rmse']:.6f} rel_rmse={row['rel_rmse']:.6f}"
                    )
                    if "col" in metrics:
                        col = metrics["col"]
                        print(
                            " " * 13 +
                            f"col_err max_abs={col['max_abs']:.6f} "
                            f"mean_abs={col['mean_abs']:.6f} "
                            f"rmse={col['rmse']:.6f} rel_rmse={col['rel_rmse']:.6f}"
                        )


if __name__ == "__main__":
    main()
