#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tk_fa4.fp4_pv_experiments import benchmark_mxfp4_forward_hbm_pv_bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = (
        (16, (4096, 8192, 16384)),
        (4, (16384,)),
    )
    rows: list[dict[str, object]] = []
    for heads, seqlens in groups:
        rows.extend(
            benchmark_mxfp4_forward_hbm_pv_bound(
                seqlens=seqlens,
                chunk_cols=(1024, 2048, 4096, 8192, 16384),
                seed=94601,
                batch=1,
                heads=heads,
                device=args.device,
                warmup=args.warmup,
                iters=args.iters,
                max_ring_payload_mib=1024.0,
                measure_forward_baselines=False,
                measure_chunked_accumulation=True,
                timeout_ms=30_000.0,
            )
        )
        args.output.write_text(
            json.dumps(
                {
                    "task": "large-shape compact MXFP4 materialized-P consumer floor",
                    "device": args.device,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "producer_cost_included": False,
                    "rows": rows,
                },
                indent=2,
                allow_nan=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
