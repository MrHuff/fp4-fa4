import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tk_fa4 import _C


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--heads", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    shape = (1, args.heads, args.seqlen, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = q.shape[-1] ** -0.5

    os.environ["TK_FA4_FORCE_2CTA"] = "1"
    os.environ["TK_FA4_FWD_MODE"] = "cluster"
    out, l_aux = _C.mha_fwd(q, k, v, False, scale, args.seqlen)
    torch.cuda.synchronize()
    print("out_finite", torch.isfinite(out).all().item())
    print("l_aux_finite", torch.isfinite(l_aux).all().item())


if __name__ == "__main__":
    main()
