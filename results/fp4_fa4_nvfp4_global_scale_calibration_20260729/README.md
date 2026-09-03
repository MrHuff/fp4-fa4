# NVFP4 global-scale calibration

This experiment tests whether tensor-level NVFP4 scaling can improve the
fast shiftless FP4 FlashAttention forward path without restoring stabilized
softmax or adding work to its latency-critical loop.

## Implementation

For an input tensor `X`, the benchmark quantizes `a X` and supplies the
matching reciprocal `1/a` to the attention kernel. Q and K reciprocals are
folded into the existing QK score scalar. The V reciprocal is folded into
the existing output normalization scalar:

```text
QK scale = (1 / a_Q) (1 / a_K) / sqrt(D)
output scale = (1 / sum(P)) (1 / a_V)
```

There is no separate rescale kernel, stabilized-P path, synchronization,
shared memory, or TMEM allocation. The V path is compile-time optional and
the identity-scale ABI remains available.

In exact arithmetic the encode/decode factors cancel. They can still change
the result because NVFP4 block scales are rounded onto the E4M3 scale grid.
The factor therefore selects a grid phase; it does not create more FP4
levels.

There are two distinct parameterizations:

1. The Transformer Engine rule replaces `448` with a target `C`:
   `a = 6*C/amax(X)`.
2. A fixed encode phase sets `a` directly, independently of tensor amax.

The fixed `1.25` factors below are the second policy. They are not shorthand
for one universal replacement of `448`.

## Main result

The best broad long-sequence candidate from this search was
`a_Q = a_K = 1.25`, `a_V = 1.40625`, with the P scale left at `1`.

| Shape | Seeds | Baseline cosine | Calibrated cosine | Baseline rel. L2 | Calibrated rel. L2 |
|---|---:|---:|---:|---:|---:|
| B1 S4096 H24 D128 | 5 | 0.962183 | 0.962251 | 0.278307 | 0.277572 |
| B1 S8192 H24 D128 | 5 | 0.962603 | 0.962689 | 0.276622 | 0.275802 |
| B4 S4096 H32 D128 | 5 | 0.962335 | 0.962359 | 0.277758 | 0.277168 |

The relative-L2 reductions are respectively `0.26%`, `0.30%`, and `0.21%`.
At B1/S4096/H24 the mean RMSE falls from `0.00718451` to `0.00716552`.
At B1/S8192/H24 it falls from `0.00505318` to `0.00503821`; at
B4/S4096/H32 it falls from `0.00717971` to `0.00716448`.

This calibration is not universal. At B1/S1024/H16 the long-sequence
candidate regressed seed 0. A separately fitted short-sequence phase
(`1.09375`, `1.40625`, `1.9375`) improved the five-seed mean relative L2
from `0.284267` to `0.283967`, but one seed regressed. Identity therefore
remains the production default and explicit factors are exposed for
calibration.

## Direct replacement of 448

The benchmark also sweeps the requested Transformer Engine policy directly:

```text
a_Q = 6*C_QK/amax(Q)
a_K = 6*C_QK/amax(K)
a_V = 6*C_V/amax(V)
```

On B1/S4096/H24 seed 0, a shared `C=334` reduced relative L2 from
`0.27994469` to `0.27960849`. It did not survive five-seed validation:
mean relative L2 became `0.27854209`, versus `0.27830735` at identity.
Coarse five-seed sweeps of `C` from `240` through `416` all lost to
identity when the same target was applied to Q, K, and V.

Changing only V from `448` to `256` was the best direct replacement found
at the primary shape. Its five-seed mean relative L2 was `0.27821104`,
but it did not generalize:

| Shape | Identity rel. L2 | V `C=256` rel. L2 | Change |
|---|---:|---:|---:|
| B1 S1024 H16 D128 | 0.284267 | 0.284526 | worse |
| B1 S4096 H24 D128 | 0.278307 | 0.278211 | better |
| B1 S8192 H24 D128 | 0.276622 | 0.276683 | worse |
| B4 S4096 H32 D128 | 0.277758 | 0.277878 | worse |

Direct Q/K targets `C=256`, `300`, and `336` also regressed the primary
five-seed mean. There is therefore no single smaller replacement for `448`
supported by this benchmark matrix.

## Speed

Paired timings used the same feature-enabled binary on one NVIDIA GB200.
The measured differences are within clock and timer variation.

| Shape | Identity (ms) | Calibrated (ms) |
|---|---:|---:|
| B1 S1024 H16 D128 | 0.016672 | 0.016672 |
| B1 S4096 H24 D128 | 0.104448 | 0.104416 |
| B1 S8192 H24 D128 | 0.378032 | 0.377152 |
| B4 S4096 H32 D128 | 0.450144 | 0.450192 |

The calibrated B1/S4096/H24 full comparison measured `0.104416 ms`,
versus `0.163872 ms` for HAO BF16 and `0.192512 ms` for HAO native NV/NV.
That is a `1.569x` speedup over the measured BF16 reference.

The optional V decode compiles at 128 registers with zero spills and
unchanged shared-memory and barrier use. Static SASS differs by two `FMUL`
instructions and one `LDCU`; these are folded into the existing epilogue
and did not produce a measurable wall-time cost.

## Negative controls

Applying the fixed factor only to P does not materially improve precision.
The best local P phase, `1.28125`, moved seed-0 relative L2 from
`0.27994469` to `0.27992132`; the exact Transformer Engine mantissa phase
`1.3125` produced `0.27996153`.

For normalized P, `amax(P)=1`, so Transformer Engine's `C=448` gives
`a=2688`. E4M3 exponent rebasing makes this equivalent to mantissa phase
`2688/2048=1.3125`; the P phase probe therefore did test the direct
constant without risking exponent overflow. The fast shiftless
representation is not itself bounded by one, so applying the unreduced
`2688` inside that path is unsafe without restoring row-max stabilization.

Interpreting a literal reciprocal-448 adjustment as the unrelated phase
`512/448=8/7` regressed seed-0 relative L2 to `0.27999306`.

The conclusion is narrow: free E4M3 grid-phase calibration can recover a
small amount of Q/K/V quantization error. It does not address the dominant
error from the fast approximate P/softmax path.

## Reproduction

Build with `HAO_FP4PV_NV_V_GLOBAL_DECODE=1`, then pass fixed encode factors
to the benchmark:

```bash
python hao_direct_fp4pv_benchmark.py \
  --extension /tmp/tk_hao_nv_vscale_b1s4096h24.so \
  --extension-module _C_tk_hao_nv_vscale_b1s4096h24 \
  --qk-format nvfp4 --pv-format nvfp4 \
  --nv-q-global-encode 1.25 \
  --nv-k-global-encode 1.25 \
  --nv-v-global-encode 1.40625
```

To replace `448` directly for prequantized tensors, select the TE policy:

```bash
python hao_direct_fp4pv_benchmark.py \
  --extension /tmp/tk_hao_nv_vscale_b1s4096h24.so \
  --extension-module _C_tk_hao_nv_vscale_b1s4096h24 \
  --qk-format nvfp4 --pv-format nvfp4 \
  --nv-qk-global-scale te \
  --nv-v-global-scale te \
  --nv-qk-e4m3-max 448 \
  --nv-v-e4m3-max 256
```

The tracked [`summary.json`](summary.json) records the validation means,
timings, generated-code audit, and negative controls. The focused P probes
are retained in the adjacent shiftless phase-probe result directories.
