#!/usr/bin/env python3
"""Capture one annotated saturated update with CUDA-profiler range control.

Nsight Systems does not reliably start a capture from the legacy NVTX
push/pop API used by ``torch.cuda.nvtx.range`` on this stack.  This wrapper
keeps the benchmark itself unchanged, enables its fine-grained low-precision
stage ranges, and brackets only the requested ``--profile-update`` with
``cudaProfilerStart``/``cudaProfilerStop``.  Profiled timings are diagnostic;
headline throughput must come from an uninstrumented benchmark process.

Unlike the throughput harness, the profiler accepts one measured update: its
purpose is to attribute a single fully warmed step, not estimate a latency
distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _run_profiled_update(
    original: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    if not bool(kwargs.get("profile", False)):
        return original(*args, **kwargs)
    torch.cuda.profiler.start()
    try:
        return original(*args, **kwargs)
    finally:
        # The underlying timed update synchronizes its final CUDA event before
        # returning, so this stop lands after every nested NVTX range and GPU
        # kernel in the selected optimizer update.
        torch.cuda.profiler.stop()


def main() -> None:
    from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_e2e as runtime_module
    from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_saturated as benchmark

    runtime_module._PROFILE_STAGE_RANGES = True
    runtime_module._PROFILE_STAGE_NVTX = True
    benchmark.MINIMUM_MEASURED_UPDATES = 1
    original = benchmark._timed_update

    def profiled(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _run_profiled_update(original, *args, **kwargs)

    benchmark._timed_update = profiled
    benchmark.main()


if __name__ == "__main__":
    main()
