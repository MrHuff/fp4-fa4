import argparse
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "flash-attention"))

from tk_fa4 import _C


def time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force2cta", choices=("0", "1"))
    parser.add_argument("--backend", choices=("tk_raw", "tk_wrapper", "cute", "all"), default="all")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=32)
    args = parser.parse_args()

    batch, seqlen, heads, heads_kv, head_dim = 1, args.seqlen, args.heads, args.heads_kv, 128
    q = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    scale = head_dim ** -0.5
    q_bhsd = q.permute(0, 2, 1, 3).contiguous()
    k_bhsd = k.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v.permute(0, 2, 1, 3).contiguous()

    if args.backend in ("tk_raw", "tk_wrapper", "all"):
        force = args.force2cta or "0"
        os.environ["TK_FA4_FWD_MODE"] = "cluster"
        os.environ["TK_FA4_FORCE_2CTA"] = force
        if args.backend in ("tk_raw", "all"):
            raw_ms = time_ms(
                lambda: _C.mha_fwd(q_bhsd, k_bhsd, v_bhsd, False, scale, seqlen),
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"tk_force2cta={force} raw_ms={raw_ms:.3f}")
        if args.backend in ("tk_wrapper", "all"):
            from tk_fa4.interface import flash_attn_func as tk_flash
            wrapper_ms = time_ms(
                lambda: tk_flash(q, k, v, causal=False),
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"tk_force2cta={force} wrapper_ms={wrapper_ms:.3f}")

    if args.backend in ("cute", "all"):
        from flash_attn.cute.interface import flash_attn_func as cute_flash
        cute_ms = time_ms(lambda: cute_flash(q, k, v, causal=False), warmup=args.warmup, iters=args.iters)
        print(f"cute_ms={cute_ms:.3f}")


if __name__ == "__main__":
    main()
