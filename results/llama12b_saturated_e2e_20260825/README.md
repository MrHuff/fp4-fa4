# Saturated Llama-1.2B causal FA4 end-to-end comparison (2026-08-25)

This directory records a single-GPU, memory-resident comparison on one NVIDIA
GB200 (SM100) of:

- packed-parameter BF16 QKV projections plus BF16 causal attention;
- fused NVFP4-QK publication plus causal FA4 with FP8 PV;
- fused NVFP4-QK publication plus causal FA4 with MXFP4 PV.

This is a **system-recipe comparison**, not an attention-format-only comparison:
the low-precision routes include fused QKV projection/RoPE/publication while the
BF16 route uses eager BF16 Q/K/V projection and BF16 attention. Use isolated
kernel results for attention-only claims.

## Protocol

The uninstrumented headline bracket was run in order
`BF16-FP8-MX-MX-FP8-BF16`. Every process used the same GB200 GPU, seed
`20260825`, canonical initial checkpoint, pinned Dolma corpus/tokenizer, and
packed QKV parameter layout. The model preset was `llama3.2-1b` (16 layers,
hidden size 2048, 32 query heads, 8 KV heads, head dimension 64). Batch 16 x
sequence 4096 gives 65,536 tokens/update. Each run used 3 warm-up updates and 20
measured updates.

Optimization was fused AdamW (`lr=0.00048828125`, betas `(0.9, 0.95)`, weight
decay `0.1`) with foreach global-norm clipping at 1.0. Low-precision routes used
loss scale 262,144 and the same low-precision backward implementation. The loss
was the `cut-cross-entropy` package's torch-compiled linear CE path; the logical
operation is `e @ c.T`, and full-logit buffer elision was not proven.

## Headline uninstrumented results

Values below are arithmetic averages of the two per-route run summaries.
"Matched speedup" is the geometric mean of the two matched BF16/candidate
mean-step ratios. Useful MFU is the harness's p50 BF16-equivalent useful MFU
against 2,250 TFLOP/s.

| Route | Mean step (ms) | Mean p50 (ms) | Sustained tok/s | Matched speedup | Useful MFU | Decoder (ms) | Backward (ms) | Final heldout loss | Peak alloc/reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 packed | 666.280 | 666.284 | 98,159 | 1.00000x | 35.934% | 193.706 | 435.545 | 8.05047 | 160.733 / 173.174 |
| NVFP4-QK + FP8-PV | 607.521 | 607.063 | 107,693 | 1.09672x | 39.440% | 176.105 | 394.782 | 8.18021 | 155.530 / 167.754 |
| NVFP4-QK + MXFP4-PV | 609.380 | 609.414 | 107,366 | 1.09337x | 39.288% | 176.478 | 396.226 | 8.11407 | 155.530 / 167.754 |

Sustained-throughput ratios are 1.09713x for FP8-PV and 1.09380x for
MXFP4-PV. A paired circular moving-block bootstrap over the measured windows
(block length 4, 200,000 draws) gives conditional 95% intervals
`[1.09410, 1.09940]` and `[1.09142, 1.09549]`, respectively. These intervals
condition on only two captured run windows. MX was 1.860 ms slower than FP8 on
average, but a conservative two-run, `df=1` interval is `[-0.048, +3.767]` ms;
do not treat the sub-2-ms ordering as settled.

Low precision saved 5.203 GiB peak allocated and 5.420 GiB peak reserved versus
BF16. Peak reserved memory was 93.96% of physical memory for BF16 and 91.02% for
low precision.

## Diagnostic one-step Nsight profile

The profile is **not** a throughput result. It captured one measured update
after three warm-ups under Nsight instrumentation; headline latency and speedup
come only from the uninstrumented bracket above.

| Route | Step kernels | Backward kernels | Kernel-union busy | Max internal gap | Mean GR Active | Mean SMs Active |
|---|---:|---:|---:|---:|---:|---:|
| FP8-PV | 2,050 | 1,147 | 98.061% | 0.744 ms | 99.398% | 94.681% |
| MXFP4-PV | 2,050 | 1,147 | 98.141% | 0.596 ms | 99.472% | 94.681% |

Busy time merges all-stream kernel intervals from first kernel start to last
kernel end. GPU metrics are 100-us sample means over that same window. Both
traces therefore pass the predeclared saturation gates: at least 97.5% union
busy, under 1 ms maximum internal gap, at least 99% GR Active, and at least 94%
SM Active.

The ordered backward signature `(demangled kernel, grid, block,
registers/thread, static shared memory, dynamic shared memory)` is exactly
identical across routes: 1,147 launches and SHA256
`6e62e02b6944238aed9e329910d54b029c718fa155d4e75a0735c44da2620a86`.
The total signatures differ only at 32 expected forward positions: 16 QKV
publisher kernels and 16 FP8-PV versus MXFP4-PV attention kernels. Each trace
has 16 dual output-weight preparation ranges, exactly one prepared-weight
transpose kernel per range, and zero legacy forward-pack or
backward-transpose-pack ranges.

The profile localizes why the isolated MXFP4-PV advantage nearly disappears
end to end:

| One-step kernel sum | FP8-PV (ms) | MXFP4-PV (ms) | MX - FP8 (ms) |
|---|---:|---:|---:|
| Causal attention, 16 layers | 30.825056 | 29.445537 | -1.379519 |
| QKV projection + RoPE + publication | 14.844544 | 16.504224 | +1.659680 |
| Combined route-specific pair | 45.669600 | 45.949761 | +0.280161 |
| Whole decoder | 181.307168 | 181.289893 | -0.017275 |

Thus MXFP4-PV attention itself is 1.04685x faster (4.48% less kernel time), but
its current publisher is 11.18% slower and cancels the gain. The profile's
backward kernel sum happened to favor MX by 3.664 ms despite an identical graph,
while the uninstrumented bracket favored FP8 backward by 1.444 ms; this sign
reversal is run/system noise, not evidence of route-specific backward work.

Nsight emitted generic capture-range warnings about possibly missing NVTX/CUDA
events. Nevertheless, every expected range is present, outer-stage launch
counts sum exactly to the step total in both traces, route counts match, and no
buffer-drop diagnostic was recorded.

## Numerical scope and caveats

These are 20-update diagnostics, not pretraining convergence results. Against
matched BF16 runs, final heldout loss increased by 0.12974 for FP8-PV and
0.06360 for MXFP4-PV. Training-loss trajectory MAE/RMSE was approximately
`0.176/0.318` for FP8-PV and `0.134/0.249` for MXFP4-PV. Final sampled-logit
cosine similarity was approximately 0.99496 for FP8-PV and 0.99441 for
MXFP4-PV; relative L2 was 0.104 and 0.117, and sampled KL was 0.108 and 0.092.
Same-route repeat spreads and gradients are noisy at this horizon, so these
measurements support short-run numerical comparability only. They do not
establish long-run loss parity.

## Rejected exact MX publisher follow-up

An exact shared-memory reread variant reduced the active projection kernel from
203 to 200 registers, but increased stack use from 48 to 64 bytes. All accepted
forward MXFP4 and backward E4M3 hashes matched. In an immediately bracketed
1,001-sample microbenchmark it measured 1.021312 ms p50 versus 1.022304 ms for
the accepted publisher: a gain of only 0.000992 ms/layer, or about 0.0159
ms/update across 16 layers. The measured end-to-end break-even is approximately
0.281 ms/update, so the candidate was rejected and is not part of this branch.

Other exact-byte candidates either regressed substantially or offered no
credible path across the break-even threshold without a larger producer and
publication redesign. In particular, backward E4M3 publication is already a
separate path and was not changed by these MX forward experiments.

## Authenticated inputs and code

| Item | SHA256 | Bytes |
|---|---|---:|
| FP8-PV forward extension | `88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208` | 1,817,256 |
| MXFP4-PV forward extension | `cc06fe4337fdc3a7c900f81d68fabc4a8e0c375ea536fbe6405754237a393717` | 1,958,000 |
| Headline dual-prep projection/backward extension | `5a85d7d9703d89f2303677571c7fb6967a73790562e75a4dbb327dff2e826099` | 18,316,872 |
| Public-packaged projection extension used by the profile | `ad4f03d71eeac317124856765a6b3a0b6267a7ba08b12683702dd340d76ed287` | 18,316,872 |
| Backward control source | `cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1` | 220,876 |
| Initial checkpoint | `2760f5eb47fd0241317dfd69bd0e2d906909d948d81a5a93f0fd371944f0d2bc` | 2,471,682,787 |
| Saturated harness | `552157c539410c1759dd8b90652ad41329da92ab212873138a0346a8b9bb986a` | 54,208 |
| Runtime | `def0512d1b4db1f756db68c1811aff7367b0a16d03b25c0216d08ad57f33d9c9` | 147,177 |
| Packed-BF16 QKV helper | `a4f39a5e578766996c42d6625d800932e3efb242bca64d1784bbd9fb16bbb795` | 12,789 |
| Profiler wrapper | `c78722574637b77f79c456a0a22317f075ae03989c1932a94e069d34abb59888` | 1,964 |

Pinned data identities: Dolma JSONL
`860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce`;
tokenizer
`76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa`.

## Reproduction pointers

The source entry points are:

- `tk_fa4/lowp_fa4_bwd/benchmark_llama12b_saturated.py` for uninstrumented
  route runs;
- `tk_fa4/lowp_fa4_bwd/profile_llama12b_saturated.py` for the one-update
  CUDA-profiler/NVTX capture;
- `tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py` for route and stage
  implementation;
- `tk_fa4/lowp_fa4_bwd/packed_bf16_qkv.py` for canonical checkpoint packing.

Reproduce the headline in the same mirrored `BF16-FP8-MX-MX-FP8-BF16`
order with `--batch 16 --warmups 3 --updates 20 --seed 20260825`, the optimizer
arguments above, and the authenticated checkpoint/data/artifacts listed here.
Low-precision runs require the selected projection extension both in
`TK_FA4_LOWP_BWD_EXTENSION_SOURCE` and `--projection-extension`, plus its
declared SHA256 and byte count. The harness authenticates inputs and refuses to
overwrite outputs. For profiling, use
`--warmups 3 --updates 1 --profile-update 3` under Nsight's `cudaProfilerApi`
capture; never use the instrumented update for headline throughput.

The original headline JSON sequence is `v6a_bf16.json`, `v6b_fp8.json`,
`v6c_mx.json`, `v6d_mx.json`, `v6e_fp8.json`, `v6f_bf16.json`. The JSON
`configuration`, `artifacts`, `checkpoint`, `data`, `source_files`, `records`,
and `steady_state` fields are the authoritative machine-readable record. The
profile harness outputs are `profile_fp8_dual.json` and
`profile_mx_dual.json`; Nsight-derived utilization and launch-graph audit values
are recorded above rather than embedded in those harness JSON files.
