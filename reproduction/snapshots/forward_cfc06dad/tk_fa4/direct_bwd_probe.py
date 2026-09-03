import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_FLASH_ROOT = _REPO_ROOT / "flash-attention"


def finite_and_max(x: torch.Tensor) -> tuple[bool, float]:
    finite = bool(torch.isfinite(x).all().item())
    max_val = float(torch.nan_to_num(x.abs(), nan=0.0, posinf=0.0, neginf=0.0).max().item())
    return finite, max_val


def summarize_nonfinite_rows(x: torch.Tensor) -> dict[str, object]:
    finite_mask = torch.isfinite(x)
    row_ok = finite_mask.all(dim=-1).all(dim=0).all(dim=-1)  # [seqlen]
    bad_rows = torch.nonzero(~row_ok, as_tuple=False).flatten()
    if bad_rows.numel() == 0:
        return {
            "count": 0,
            "first_rows": [],
            "first_tiles64": [],
            "min_row": None,
            "max_row": None,
            "min_tile64": None,
            "max_tile64": None,
            "tile64_count": 0,
        }
    first_rows = [int(v) for v in bad_rows[:16].tolist()]
    bad_tiles64 = torch.div(bad_rows, 64, rounding_mode="floor")
    unique_tiles64 = torch.unique(bad_tiles64)
    first_tiles64 = [int(v) for v in unique_tiles64[:16].tolist()]
    return {
        "count": int(bad_rows.numel()),
        "first_rows": first_rows,
        "first_tiles64": first_tiles64,
        "min_row": int(bad_rows[0].item()),
        "max_row": int(bad_rows[-1].item()),
        "min_tile64": int(unique_tiles64[0].item()),
        "max_tile64": int(unique_tiles64[-1].item()),
        "tile64_count": int(unique_tiles64.numel()),
    }


def compute_row_max_absdiff(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    diff = (x - ref).abs().to(torch.float32)
    return diff.amax(dim=-1).amax(dim=0).amax(dim=-1)  # [seqlen]


def summarize_row_max(row_max: torch.Tensor, threshold: float = 1e-2) -> dict[str, object]:
    bad_rows = torch.nonzero(row_max > threshold, as_tuple=False).flatten()
    topk = min(8, row_max.numel())
    top_vals, top_idx = torch.topk(row_max, k=topk)
    if bad_rows.numel() == 0:
        return {
            "max_absdiff": float(row_max.max().item()),
            "count": 0,
            "first_rows": [],
            "first_tiles64": [],
            "min_row": None,
            "max_row": None,
            "min_tile64": None,
            "max_tile64": None,
            "tile64_count": 0,
            "top_tiles64": [int(v) for v in torch.div(top_idx, 64, rounding_mode="floor").tolist()],
            "top_rows": [int(v) for v in top_idx.tolist()],
            "top_row_maxabsdiff": [float(v) for v in top_vals.tolist()],
        }
    bad_tiles64 = torch.div(bad_rows, 64, rounding_mode="floor")
    unique_tiles64 = torch.unique(bad_tiles64)
    return {
        "max_absdiff": float(row_max.max().item()),
        "count": int(bad_rows.numel()),
        "first_rows": [int(v) for v in bad_rows[:16].tolist()],
        "first_tiles64": [int(v) for v in unique_tiles64[:16].tolist()],
        "min_row": int(bad_rows[0].item()),
        "max_row": int(bad_rows[-1].item()),
        "min_tile64": int(unique_tiles64[0].item()),
        "max_tile64": int(unique_tiles64[-1].item()),
        "tile64_count": int(unique_tiles64.numel()),
        "top_tiles64": [int(v) for v in torch.div(top_idx, 64, rounding_mode="floor").tolist()],
        "top_rows": [int(v) for v in top_idx.tolist()],
        "top_row_maxabsdiff": [float(v) for v in top_vals.tolist()],
    }

def summarize_absdiff_rows(x: torch.Tensor, ref: torch.Tensor, threshold: float = 1e-2) -> dict[str, object]:
    return summarize_row_max(compute_row_max_absdiff(x, ref), threshold)


def summarize_grouped_row_max(
    row_max: torch.Tensor,
    group_size: int,
    threshold: float = 1e-2,
) -> dict[str, object]:
    group_max = torch.stack([chunk.max() for chunk in row_max.split(group_size)])
    bad_groups = torch.nonzero(group_max > threshold, as_tuple=False).flatten()
    topk = min(8, group_max.numel())
    top_vals, top_idx = torch.topk(group_max, k=topk)
    return {
        "group_size": group_size,
        "max_absdiff": float(group_max.max().item()),
        "count": int(bad_groups.numel()),
        "first_groups": [int(v) for v in bad_groups[:16].tolist()],
        "min_group": int(bad_groups[0].item()) if bad_groups.numel() > 0 else None,
        "max_group": int(bad_groups[-1].item()) if bad_groups.numel() > 0 else None,
        "top_groups": [int(v) for v in top_idx.tolist()],
        "top_group_maxabsdiff": [float(v) for v in top_vals.tolist()],
    }


def summarize_grouped_row_max_parity(
    row_max: torch.Tensor,
    group_size: int,
    threshold: float = 1e-2,
) -> dict[str, object]:
    group_max = torch.stack([chunk.max() for chunk in row_max.split(group_size)])
    result: dict[str, object] = {"group_size": group_size}
    for label, offset in (("even", 0), ("odd", 1)):
        parity_max = group_max[offset::2]
        if parity_max.numel() == 0:
            result[label] = {
                "max_absdiff": None,
                "count": 0,
                "top_groups": [],
                "top_group_maxabsdiff": [],
            }
            continue
        bad_groups = torch.nonzero(parity_max > threshold, as_tuple=False).flatten()
        topk = min(8, parity_max.numel())
        top_vals, top_idx = torch.topk(parity_max, k=topk)
        result[label] = {
            "max_absdiff": float(parity_max.max().item()),
            "count": int(bad_groups.numel()),
            "top_groups": [int((2 * v) + offset) for v in top_idx.tolist()],
            "top_group_maxabsdiff": [float(v) for v in top_vals.tolist()],
        }
    return result


def summarize_subtiles16_in_64(row_max: torch.Tensor) -> list[dict[str, object]]:
    blocks64 = [chunk for chunk in row_max.split(64)]
    summaries: list[dict[str, object]] = []
    for subtile_idx in range(4):
        subtile_block_max = torch.stack([
            block[subtile_idx * 16:min((subtile_idx + 1) * 16, block.numel())].max()
            for block in blocks64
            if block.numel() > subtile_idx * 16
        ])
        topk = min(8, subtile_block_max.numel())
        top_vals, top_idx = torch.topk(subtile_block_max, k=topk)
        summaries.append({
            "subtile_in_64": subtile_idx,
            "max_absdiff": float(subtile_block_max.max().item()),
            "mean_absdiff": float(subtile_block_max.mean().item()),
            "top_blocks64": [int(v) for v in top_idx.tolist()],
            "top_block64_maxabsdiff": [float(v) for v in top_vals.tolist()],
        })
    return summaries


def load_backend_state(forward_backend: str) -> tuple[dict[str, object], object]:
    from tk_fa4.interface import _C, b300_mha_bwd, b300_mha_bwd_experimental, b300_mha_fwd

    backends = {
        "exact": b300_mha_bwd,
        "hot": b300_mha_bwd_experimental,
        "ref": b300_mha_bwd_experimental,
        "base": _C.b300_mha_bwd_hot_cute16_internal,
        "base_nopatch": _C.b300_mha_bwd_hot_cute16_nopatch_internal,
        "candidate": _C.b300_mha_bwd_hot_cute16_candidate_internal,
        "candidate_bf16_dkdv": _C.b300_mha_bwd_hot_cute16_candidate_bf16_dkdv_internal,
        "candidate2": _C.b300_mha_bwd_hot_cute16_candidate2_internal,
        "trusted": _C.b300_mha_bwd_hot_trusted_internal,
        "legacy": _C.b300_mha_bwd_hot_legacy_internal,
    }
    if forward_backend == "tk":
        def forward_with_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
            return b300_mha_fwd(q, k, v, causal=causal, return_lse=True)

        return backends, forward_with_lse
    if forward_backend != "cute":
        raise SystemExit(f"unknown forward backend: {forward_backend}")

    for path in (_FLASH_ROOT, _REPO_ROOT):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    import flash_attn

    flash_attn_path = Path(flash_attn.__file__).resolve()
    if not flash_attn_path.is_relative_to(_FLASH_ROOT):
        raise RuntimeError(
            "direct_bwd_probe.py must import vendored flash_attn from "
            f"{_FLASH_ROOT}, got {flash_attn_path}"
        )

    from flash_attn.cute.interface import flash_attn_func as cute_flash_attn_func

    def forward_with_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = cute_flash_attn_func(q, k, v, causal=causal, return_lse=True)
        if lse.ndim == 3 and lse.shape == (q.shape[0], q.shape[2], q.shape[1]):
            lse = lse.permute(0, 2, 1).contiguous()
        return out, lse

    return backends, forward_with_lse


def run_probe(
    backend: str,
    seqlen: int,
    device_index: int,
    backends: dict[str, object],
    forward_with_lse: object,
    forward_backend: str,
    time_iters: int,
    ref_backend: str | None,
    dq_debug_groups: bool = False,
) -> dict[str, object]:
    if backend not in backends:
        raise SystemExit(f"unknown backend: {backend}")

    device = f"cuda:{device_index}"
    dtype = torch.bfloat16
    b, h = 1, 16
    qk_dim = 192
    v_dim = 128

    q = torch.randn((b, seqlen, h, qk_dim), device=device, dtype=dtype)
    k = torch.randn((b, seqlen, h, qk_dim), device=device, dtype=dtype)
    v = torch.randn((b, seqlen, h, v_dim), device=device, dtype=dtype)
    out, lse = forward_with_lse(q, k, v, causal=True)
    dout = torch.randn_like(out)

    softmax_scale = q.shape[-1] ** -0.5
    def run_backward(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if name == "exact":
            return backends[name](
                q,
                k,
                v,
                out,
                lse,
                dout,
                causal=True,
                deterministic=False,
            )
        if name in ("hot", "ref"):
            return backends[name](
                q,
                k,
                v,
                out,
                lse,
                dout,
                causal=True,
                deterministic=False,
                implementation="hot" if name == "hot" else "ref",
            )
        if name == "legacy":
            return backends[name](
                q,
                k,
                v,
                out,
                lse,
                dout,
                True,
                float(softmax_scale),
                seqlen,
            )
        return backends[name](
            q,
            k,
            v,
            out,
            lse,
            dout,
            True,
            float(softmax_scale),
            seqlen,
            False,
        )

    time_us = None
    if time_iters > 0:
        for _ in range(2):
            _ = run_backward(backend)
        torch.cuda.synchronize(device_index)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(time_iters):
            _ = run_backward(backend)
        end.record()
        torch.cuda.synchronize(device_index)
        time_us = start.elapsed_time(end) * 1000.0 / time_iters

    dq, dk, dv = run_backward(backend)
    torch.cuda.synchronize(device_index)

    dq_finite, dq_max = finite_and_max(dq)
    dk_finite, dk_max = finite_and_max(dk)
    dv_finite, dv_max = finite_and_max(dv)
    dk_nonfinite = summarize_nonfinite_rows(dk)

    result = {
        "backend": backend,
        "forward_backend": forward_backend,
        "device": device_index,
        "seqlen": seqlen,
        "dq_finite": dq_finite,
        "dk_finite": dk_finite,
        "dv_finite": dv_finite,
        "dq_max": dq_max,
        "dk_max": dk_max,
        "dv_max": dv_max,
        "dk_nonfinite": dk_nonfinite,
        "time_us": time_us,
    }
    if ref_backend is not None:
        dq_cmp = dq.clone()
        dk_cmp = dk.clone()
        dv_cmp = dv.clone()
        ref_dq, ref_dk, ref_dv = run_backward(ref_backend)
        torch.cuda.synchronize(device_index)
        result["ref_backend"] = ref_backend
        dq_row_max = compute_row_max_absdiff(dq_cmp, ref_dq)
        result["dq_refdiff"] = summarize_row_max(dq_row_max)
        result["dk_refdiff"] = summarize_absdiff_rows(dk_cmp, ref_dk)
        result["dv_refdiff"] = summarize_absdiff_rows(dv_cmp, ref_dv)
        if dq_debug_groups:
            result["dq_refdiff_blocks64"] = summarize_grouped_row_max(dq_row_max, 64)
            result["dq_refdiff_groups128"] = summarize_grouped_row_max(dq_row_max, 128)
            result["dq_refdiff_blocks64_parity"] = summarize_grouped_row_max_parity(dq_row_max, 64)
            result["dq_refdiff_subtiles16_in_64"] = summarize_subtiles16_in_64(dq_row_max)
    return result


def parse_optional_args(argv: list[str]) -> tuple[str | None, bool]:
    ref_backend: str | None = None
    dq_debug_groups = False
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--dq-debug-groups":
            dq_debug_groups = True
            idx += 1
            continue
        if arg.startswith("--"):
            raise SystemExit(f"unknown option: {arg}")
        if ref_backend is None:
            ref_backend = None if arg.lower() == "none" else arg
            idx += 1
            continue
        raise SystemExit(f"unexpected positional argument: {arg}")
    if dq_debug_groups and ref_backend is None:
        raise SystemExit("--dq-debug-groups requires ref_backend")
    return ref_backend, dq_debug_groups


def main() -> int:
    backend_arg = sys.argv[1] if len(sys.argv) > 1 else "candidate2"
    seqlen_arg = sys.argv[2] if len(sys.argv) > 2 else "4096"
    device_index = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    forward_backend = sys.argv[4] if len(sys.argv) > 4 else "tk"
    time_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    ref_backend, dq_debug_groups = parse_optional_args(sys.argv[6:])
    backends_to_run = backend_arg.split(",")
    seqlens = [int(v) for v in seqlen_arg.split(",")]

    torch.manual_seed(0)
    _ = torch.cuda.device_count()
    torch.cuda.set_device(device_index)
    backends, forward_with_lse = load_backend_state(forward_backend)
    for backend in backends_to_run:
        for seqlen in seqlens:
            torch.manual_seed(0)
            print(
                run_probe(
                    backend,
                    seqlen,
                    device_index,
                    backends,
                    forward_with_lse,
                    forward_backend,
                    time_iters,
                    ref_backend,
                    dq_debug_groups=dq_debug_groups,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
