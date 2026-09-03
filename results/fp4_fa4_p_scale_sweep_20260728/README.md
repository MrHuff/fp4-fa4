# NVFP4 probability-scale sweep

This is a same-input sweep of the NVFP4 probability scale lift `G` at
B1/S4096/H24/D128 on one NVIDIA GB200. It uses HAO's seed-0 tensor factory
and exact online row-maximum stabilization. QK is tested in both NVFP4 and
MXFP4; PV is NVFP4 in every swept case.

This is an appendix-only numerical diagnostic. Its approximately
`0.188416 ms` latency is caused by exact stabilization and full denominator
handling, not by `G`. The corresponding fast-path experiment is
[`../fp4_fa4_shiftless_p_scale_sweep_20260728/README.md`](../fp4_fa4_shiftless_p_scale_sweep_20260728/README.md).

## Direct result

Native HAO NVFP4/NVFP4, normalized against its BF16 output on the same
input, reaches cosine `0.981414557` and relative L2 `0.192766950`.

For stabilized TK NV/NV:

- `G=320` gives the best cosine: `0.981422722`.
- `G=448` gives the best relative L2: `0.192405358`.
- `G=2048` gives cosine `0.981298447` and relative L2 `0.193749979`.

For stabilized TK MX/NV, `G=416` is best by both metrics: cosine
`0.973068535` and relative L2 `0.232149839`.

The scale lift is a small calibration effect, not the source of the large
accuracy difference between the stabilized and shiftless kernels. The fast
shiftless NV/NV control remains cosine `0.961778760`, relative L2
`0.279943943`, and `0.104448 ms`. Stabilized `G=320` is about `0.188416 ms`.
Restoring exact row-max/denominator handling recovers most of the accuracy;
changing `G` within that stabilized path only moves cosine by about
`0.00017`.

## Normalized controls

| Route | G | Time (ms) | Cosine vs BF16 | Relative L2 vs BF16 | RMSE vs BF16 |
|---|---:|---:|---:|---:|---:|
| HAO BF16 | - | 0.162816 | 1.000000000 | 0.000000000 | 0.000000000 |
| HAO native NV/NV | - | 0.192400 | 0.981414557 | 0.192766950 | 0.004946504 |
| TK fast shiftless NV/NV | n/a | 0.104448 | 0.961778760 | 0.279943943 | 0.007183513 |
| TK stabilized NV/NV | 1 | 0.188416 | 0.981250703 | 0.194010243 | 0.004978408 |
| TK stabilized NV/NV | 320 | 0.188416 | **0.981422722** | 0.192460001 | 0.004938628 |
| TK stabilized NV/NV | 448 | 0.188416 | 0.981414318 | **0.192405358** | **0.004937225** |

The shiftless timing comes from the matched-format matrix's 100 ms timing
window. The stabilized sweep uses 10 ms warmup and a 25 ms timing window.

## Full NV/NV sweep

| G | Time (ms) | Cosine vs BF16 | Relative L2 vs BF16 | RMSE vs BF16 |
|---:|---:|---:|---:|---:|
| 1 | 0.188416 | 0.981250703 | 0.194010243 | 0.004978408 |
| 2 | 0.189440 | 0.981295228 | 0.193767980 | 0.004972191 |
| 4 | 0.188416 | 0.981298327 | 0.193750814 | 0.004971751 |
| 8 | 0.188416 | 0.981298447 | 0.193750024 | 0.004971731 |
| 16 | 0.188416 | 0.981298447 | 0.193749949 | 0.004971729 |
| 32 | 0.188416 | 0.981298447 | 0.193749979 | 0.004971729 |
| 64 | 0.188704 | 0.981298447 | 0.193749979 | 0.004971729 |
| 128 | 0.188416 | 0.981298447 | 0.193749949 | 0.004971729 |
| 256 | 0.188416 | 0.981298447 | 0.193749979 | 0.004971729 |
| 288 | 0.188416 | 0.981408596 | 0.192795277 | 0.004947231 |
| 320 | 0.188416 | **0.981422722** | 0.192460001 | 0.004938628 |
| 352 | 0.189280 | 0.981342793 | 0.193410844 | 0.004963027 |
| 384 | 0.188416 | 0.981415391 | 0.192760497 | 0.004946338 |
| 416 | 0.188704 | 0.981261194 | 0.194043368 | 0.004979258 |
| 448 | 0.188416 | 0.981414318 | **0.192405358** | **0.004937225** |
| 480 | 0.188416 | 0.981408238 | 0.192796126 | 0.004947253 |
| 512 | 0.188416 | 0.981298447 | 0.193749979 | 0.004971729 |
| 1024 | 0.188416 | 0.981298447 | 0.193749979 | 0.004971729 |
| 2048 | 0.188416 | 0.981298447 | 0.193749979 | 0.004971729 |
| 2688 | 0.188416 | 0.981409609 | 0.192793190 | 0.004947178 |

## Full MX/NV sweep

| G | Time (ms) | Cosine vs BF16 | Relative L2 vs BF16 | RMSE vs BF16 |
|---:|---:|---:|---:|---:|
| 1 | 0.188256 | 0.973014712 | 0.232443854 | 0.005964635 |
| 2 | 0.188416 | 0.973044395 | 0.232329547 | 0.005961702 |
| 4 | 0.188736 | 0.973046064 | 0.232323810 | 0.005961555 |
| 8 | 0.188480 | 0.973046184 | 0.232323349 | 0.005961543 |
| 16 | 0.188416 | 0.973046184 | 0.232323334 | 0.005961542 |
| 32 | 0.188416 | 0.973046184 | 0.232323334 | 0.005961542 |
| 64 | 0.188416 | 0.973046184 | 0.232323334 | 0.005961542 |
| 128 | 0.188416 | 0.973046184 | 0.232323334 | 0.005961542 |
| 256 | 0.188736 | 0.973046184 | 0.232323334 | 0.005961542 |
| 288 | 0.188416 | 0.972919941 | 0.233203933 | 0.005984139 |
| 320 | 0.188416 | 0.972753763 | 0.234221503 | 0.006010251 |
| 352 | 0.188416 | 0.973026395 | 0.232493803 | 0.005965917 |
| 384 | 0.188416 | 0.972928643 | 0.233190939 | 0.005983806 |
| 416 | 0.188416 | **0.973068535** | **0.232149839** | **0.005957090** |
| 448 | 0.188416 | 0.972667277 | 0.234704986 | 0.006022657 |
| 480 | 0.188416 | 0.972922921 | 0.233193353 | 0.005983868 |
| 512 | 0.188416 | 0.973046184 | 0.232323334 | 0.005961542 |
| 1024 | 0.188416 | 0.973046184 | 0.232323334 | 0.005961542 |
| 2048 | 0.188416 | 0.973046184 | 0.232323334 | 0.005961542 |
| 2688 | 0.188416 | 0.972926736 | 0.233176261 | 0.005983429 |

Powers of two become effectively identical once scale underflow clears:
they shift the E4M3 exponent and cancel in the normalized result.
Non-power-of-two factors change where values land on E4M3 mantissa bins,
which explains the narrow local optima around 320, 416, and 448.

`G` is only valid here because exact stabilization guarantees `0 < P <= 1`.
The aggressive shiftless path forms unnormalized exponentials that can
exceed one, so copying a large factor such as `448` directly can overflow
the E4M3 scale. The fast-path follow-up divides each candidate by a power of
two, preserving its E4M3 mantissa-grid phase while avoiding the high
exponent. None of the rebased stabilized optima improves the fast default.

## Reproduction

From `tk_fa4/fp4_fa4_fwd`:

```bash
python hao_p_scale_sweep.py \
  --output-dir ../../results/fp4_fa4_p_scale_sweep_20260728
```

The tracked [`summary.json`](summary.json) contains every metric above.
Per-case records, build logs, and extensions are intentionally untracked.
