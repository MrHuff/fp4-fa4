from __future__ import annotations

from typing import Any

from cute_dsl_mxfp4_forward_scaffold import load_reference_blockscaled_gemm_module


def run_mxfp4_qk_gemm_smoke(
    *,
    m: int = 128,
    n: int = 128,
    k: int = 256,
    l: int = 12,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: tuple[int, int] = (1, 1),
    device: str = "cuda:0",
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
) -> dict[str, Any]:
    mod = load_reference_blockscaled_gemm_module()

    import torch

    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    exec_time_us = mod.run(
        (m, n, k, l),
        mod.cutlass.Float4E2M1FN,
        mod.cutlass.Float8E4M3FN,
        16,
        mod.cutlass.Float16,
        "k",
        "k",
        "n",
        mma_tiler_mn,
        cluster_shape_mn,
        tolerance=1e-1,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        skip_ref_check=skip_ref_check,
        use_cold_l2=False,
    )

    return {
        "status": "ok",
        "device": str(device),
        "mnkl": (m, n, k, l),
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "exec_time_us": float(exec_time_us),
        "exec_time_ms": float(exec_time_us) / 1000.0,
        "skip_ref_check": bool(skip_ref_check),
    }
