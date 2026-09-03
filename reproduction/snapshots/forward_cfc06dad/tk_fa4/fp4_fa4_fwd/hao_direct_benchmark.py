#!/usr/bin/env python3
"""Apples-to-apples benchmark for the isolated HAO-structured TK port."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXTENSIONS = {
    "nvfp4": (
        Path(
            "/tmp/_C_tk_hao_direct"
            ".cpython-312-aarch64-linux-gnu.so"
        ),
        "_C_tk_hao_direct",
    ),
    "mxfp4": (
        Path(
            "/tmp/_C_tk_hao_direct_mxqk_fp8pv"
            ".cpython-312-aarch64-linux-gnu.so"
        ),
        "_C_tk_hao_direct_mxqk_fp8pv",
    ),
}
ROUTES = {
    "nvfp4": "real_fwd_tk_hao_direct_nvfp4_fp8pv",
    "mxfp4": "real_fwd_tk_hao_direct_mxfp4_fp8pv",
}


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison(a: Any, b: Any) -> dict[str, float]:
    import torch

    a32 = a.float()
    b32 = b.float()
    delta = a32 - b32
    error_rms = delta.square().mean().sqrt()
    reference_rms = b32.square().mean().sqrt()
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                a32.flatten().unsqueeze(0),
                b32.flatten().unsqueeze(0),
            ).item()
        ),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(error_rms.item()),
        "reference_rms": float(reference_rms.item()),
        "relative_l2": float((error_rms / reference_rms).item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument(
        "--qk-format",
        choices=("nvfp4", "mxfp4"),
        default="nvfp4",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-provider",
        choices=("tk", "hao", "bf16"),
        help="Warm the selected provider, launch it once, and exit",
    )
    return parser.parse_args()


def quantize_mxfp4_qk(rows_ref: Any) -> tuple[Any, Any]:
    import torch

    batch, seqlen, heads, dim = rows_ref.shape
    if dim != 128 or seqlen % 128:
        raise ValueError("MXFP4 direct QK requires D128 and S divisible by 128")
    quant_root = REPO_ROOT / "TK_quantisation" / "mxfp4_v3"
    sys.path.insert(0, str(quant_root))
    try:
        import mxfp4_quant_v3
    finally:
        sys.path.pop(0)

    payload = torch.empty(
        (batch, heads, seqlen, 64),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    scales = torch.empty(
        (batch, heads, seqlen // 128, 32, 16),
        device="cuda",
        dtype=torch.uint8,
    )
    for batch_idx in range(batch):
        for head in range(heads):
            head_rows = (
                rows_ref[batch_idx, :, head]
                .to(torch.bfloat16)
                .contiguous()
            )
            head_payload, head_scales = (
                mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                    head_rows, 0
                )
            )
            payload[batch_idx, head].view(torch.uint8).copy_(
                head_payload.contiguous().view(torch.uint8)
            )
            scales[batch_idx, head].copy_(
                head_scales[:, 0]
            )
    prepared_scale = (
        scales.permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch, seqlen // 128, heads, 512)
        .view(torch.float8_e4m3fn)
    )
    return payload, prepared_scale


def prepare_nvfp4_tk_inputs(
    q_fp4: Any,
    k_fp4: Any,
    v_fp8: Any,
    q_scale: Any,
    k_scale: Any,
) -> Any:
    import torch

    batch, seqlen, heads, packed_dim = q_fp4.shape
    if packed_dim != 64 or tuple(k_fp4.shape) != tuple(q_fp4.shape):
        raise ValueError("NVFP4 direct QK requires matching D128 Q and K")
    expected_scale = (
        32,
        4,
        seqlen // 128,
        4,
        2,
        heads,
        batch,
    )
    if tuple(q_scale.shape) != expected_scale:
        raise ValueError(
            f"Q scale shape {tuple(q_scale.shape)} != {expected_scale}"
        )
    if tuple(k_scale.shape) != expected_scale:
        raise ValueError(
            f"K scale shape {tuple(k_scale.shape)} != {expected_scale}"
        )

    def prepare_scale(scale: Any, duplicate_depth: bool) -> Any:
        prepared = (
            scale.permute(6, 2, 5, 4, 0, 1, 3)
            .contiguous()
            .reshape(batch, seqlen // 128, heads * 2, 512)
        )
        if prepared.dtype == torch.uint8:
            prepared = prepared.view(torch.float8_e4m3fn)
        if duplicate_depth:
            prepared = prepared.repeat_interleave(2, dim=1)
        return prepared

    q_local = (
        q_fp4.view(torch.uint8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.float4_e2m1fn_x2)
    )
    k_local = (
        k_fp4.view(torch.uint8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.float4_e2m1fn_x2)
    )
    return SimpleNamespace(
        q_fp4_bhsd=q_local,
        q_scale_prepared=prepare_scale(q_scale, False),
        q_global_scale=torch.ones(
            (batch, heads), device="cuda", dtype=torch.float32
        ),
        k_fp4_bhsd=k_local,
        k_scale_prepared=prepare_scale(k_scale, True),
        k_global_scale=torch.ones(
            (batch, heads), device="cuda", dtype=torch.float32
        ),
        v_fp8_bhds=v_fp8.permute(0, 2, 3, 1).contiguous(),
    )


def main() -> None:
    args = parse_args()

    import cutlass
    import torch
    import triton.testing
    from flash_attn.cute import interface
    from flash_attn.cute.benchmarks import bench_fp4

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.seqlen % 256:
        raise ValueError("direct two-query topology requires S divisible by 256")
    if min(args.batch, args.heads) < 1:
        raise ValueError("batch and heads must be positive")
    (
        q_fp4,
        k_fp4,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        q_ref,
        k_ref,
        v_ref,
    ) = bench_fp4.create_nvfp4_attention_tensors(
        args.batch,
        args.seqlen,
        args.seqlen,
        args.heads,
        args.heads,
        128,
        128,
        device="cuda",
        dtype_gen=torch.bfloat16,
        pv_mode="fp8",
        pv_fp8_dtype=cutlass.Float8E4M3FN,
    )
    if v_scale is not None:
        raise RuntimeError("plain FP8 V unexpectedly has scales")

    hao_q_fp4 = q_fp4
    hao_k_fp4 = k_fp4
    hao_q_scale = q_scale
    hao_k_scale = k_scale
    extension_path, module_name = EXTENSIONS[args.qk_format]
    if args.extension is not None:
        extension_path = args.extension
    extension = load(extension_path, module_name)
    if args.qk_format == "nvfp4":
        prepared = prepare_nvfp4_tk_inputs(
            q_fp4, k_fp4, v_fp8, q_scale, k_scale
        )
    else:
        q_fp4, q_scale = quantize_mxfp4_qk(q_ref)
        k_fp4, k_scale = quantize_mxfp4_qk(k_ref)
        q_sg = torch.full(
            (args.batch, args.heads),
            1.0 / 6.0,
            device="cuda",
            dtype=torch.float32,
        )
        k_sg = q_sg.clone()
        prepared = SimpleNamespace(
            q_fp4_bhsd=q_fp4,
            q_scale_prepared=q_scale,
            q_global_scale=q_sg,
            k_fp4_bhsd=k_fp4,
            k_scale_prepared=k_scale.repeat_interleave(
                2, dim=1
            ),
            k_global_scale=k_sg,
            v_fp8_bhds=v_fp8.permute(
                0, 2, 3, 1
            ).contiguous(),
        )
    tk_output = torch.empty(
        (args.batch, args.seqlen, args.heads, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    tk_lse = torch.empty(
        (args.batch, args.heads, 1, args.seqlen),
        device="cuda",
        dtype=torch.float32,
    )
    direct = {
        "causal": False,
        "return_lse": True,
        "num_splits": 1,
        "pack_gqa": False,
        "_compute_capability": 10,
    }
    q_bf16 = q_ref.to(torch.bfloat16)
    k_bf16 = k_ref.to(torch.bfloat16)
    v_bf16 = v_ref.to(torch.bfloat16)

    def run_tk(*, store_lse: bool) -> None:
        extension.forward_hao_direct_fp8pv(
            prepared.q_fp4_bhsd,
            prepared.q_scale_prepared,
            prepared.q_global_scale,
            prepared.k_fp4_bhsd,
            prepared.k_scale_prepared,
            prepared.k_global_scale,
            prepared.v_fp8_bhds,
            tk_output,
            tk_lse,
            0,
            True,
            store_lse,
        )

    def run_tk_timed() -> None:
        run_tk(store_lse=False)

    def run_hao_timed() -> Any:
        return interface.flash_attn_func(
            hao_q_fp4,
            hao_k_fp4,
            v_fp8,
            causal=False,
            mSFQ=hao_q_scale,
            mSFK=hao_k_scale,
            mSFV=None,
        )

    def run_hao_correctness() -> Any:
        return interface._flash_attn_fwd(
            hao_q_fp4,
            hao_k_fp4,
            v_fp8,
            mSFQ=hao_q_scale,
            mSFK=hao_k_scale,
            mSFV=None,
            **direct,
        )

    def run_bf16_timed() -> Any:
        return interface.flash_attn_func(
            q_bf16,
            k_bf16,
            v_bf16,
            causal=False,
        )

    def run_bf16_correctness() -> Any:
        return interface._flash_attn_fwd(
            q_bf16,
            k_bf16,
            v_bf16,
            **direct,
        )

    previous_route = os.environ.get("TK_FA4_FP4PV_FWD_CONFIG")
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = ROUTES[args.qk_format]
    try:
        if args.profile_provider is not None:
            profile_function = {
                "tk": run_tk_timed,
                "hao": run_hao_timed,
                "bf16": run_bf16_timed,
            }[args.profile_provider]
            for _ in range(3):
                profile_function()
            torch.cuda.synchronize()
            profile_function()
            torch.cuda.synchronize()
            print(
                json.dumps(
                    {
                        "profile_provider": args.profile_provider,
                        "topology": dict(
                            extension.read_hao_direct_topology()
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        run_tk(store_lse=True)
        hao_output, hao_lse = run_hao_correctness()
        bf16_output, bf16_lse = run_bf16_correctness()
        torch.cuda.synchronize()

        timings = {}
        tk_name = f"tk_hao_direct_{args.qk_format}_fp8pv"
        for name, function in (
            (tk_name, run_tk_timed),
            ("hao_native_nvfp4_fp8pv", run_hao_timed),
            ("hao_native_bf16", run_bf16_timed),
        ):
            timings[name] = float(
                triton.testing.do_bench(
                    function,
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                    return_mode="median",
                )
            )
    finally:
        if previous_route is None:
            os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
        else:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = previous_route

    n_flops = (
        args.batch
        * args.heads
        * 2
        * args.seqlen
        * args.seqlen
        * (128 + 128)
    )
    result = {
        "shape": {
            "batch": args.batch,
            "seqlen": args.seqlen,
            "heads": args.heads,
            "dim": 128,
        },
        "protocol": {
            "factory": "HAO create_nvfp4_attention_tensors",
            "tk_qk_format": args.qk_format,
            "timer": "triton.testing.do_bench median",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
        },
        "topology": dict(extension.read_hao_direct_topology()),
        "timing_ms": timings,
        "tflops": {
            name: n_flops / (timing * 1e-3) / 1e12
            for name, timing in timings.items()
        },
        "speedup_vs_hao_bf16": {
            name: timings["hao_native_bf16"] / timing
            for name, timing in timings.items()
            if name != "hao_native_bf16"
        },
        "correctness": {
            "tk_vs_bf16_output": comparison(tk_output, bf16_output),
            "tk_vs_hao_output": comparison(tk_output, hao_output),
            "tk_vs_bf16_lse": comparison(tk_lse.squeeze(2), bf16_lse),
            "tk_vs_hao_lse": comparison(tk_lse.squeeze(2), hao_lse),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
