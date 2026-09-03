# HAO-direct NVFP4/NVFP4 log-quantization checkpoint

## Scope

- GPU: NVIDIA GB200, driver 580.126.09
- Shape: B1, S4096, H24, DQK128, DVO128
- Attention: non-causal
- QK: NVFP4
- PV: NVFP4
- Timing: `triton.testing.do_bench`, median, 100 ms warmup, 500 ms repeat

## Landed changes

1. Fuse softmax exponentiation and P quantization in the log domain:
   - compute the block maximum before exponentiation;
   - exponentiate `score - row_max - block_scale`;
   - emit the NVFP4 block scale directly;
   - reconstruct the softmax denominator as `block_scale * sum(P)`.
2. Scan score quarter Q2 last and consume it first. This removes one x32
   TMEM score reload without retaining another register fragment.
3. Use a scalar native `EX2` for each block scale instead of duplicating the
   scalar into an `f32x2` instruction and discarding half the result.
4. Add benchmark switches for a named extension and a TK-only run, including
   direct comparison against Torch BF16 SDPA.
5. Define the HAO pipeline's full K tile and TMA descriptor in its isolated
   wrapper, removing a hidden dependency on the dirty research config file.
6. Move each `setmaxnreg` request into its disjoint warp-role branch. This
   reproduces HAO's 192/80/48 score-reader/correction/issuer allocation
   without making the low-register issuer budget cap the score-reader code.

## Final comparison

| Provider | Time (ms) | Relative to HAO NV/NV |
| --- | ---: | ---: |
| HAO BF16 | 0.163840 | 0.849x |
| HAO NVFP4/NVFP4 | 0.192960 | 1.000x |
| TK HAO-direct NVFP4/NVFP4 | 0.194848 | 1.010x |

These are matched seed-0 runs with 200 ms warmup and 1000 ms repeat. The
pre-optimization TK path measured about 0.219136 ms, and the immediately prior
fused-QK checkpoint measured 0.202624 ms. The final path is 11.1% faster than
the former, 3.84% faster than the latter, and only 0.98% behind native HAO.

Correctness against Torch BF16 SDPA at seed 0:

- output cosine: 0.981251
- output RMSE: 0.004978
- TK versus HAO output cosine: 0.999270

The kernel compiles with 128 registers per thread, no stack frame, and no
register spills.

## Rejected experiments

| Experiment | Time (ms) | Result |
| --- | ---: | --- |
| Exact HAO SM100 12.5% ALU/MUFU split | 0.237568 | 8.4% slower than the old TK path |
| Light 3.125% ALU/MUFU split on log quant | 0.207168 | Slower than all-MUFU |
| Register split 184/64/80 | 0.205472 | 32-byte loads/stores spilled |
| Grouped pre-role `setmaxnreg` at 192/48/80 | 0.223936 | 220-byte stores and 264-byte loads spilled |
| Retain Q2 as BF16 registers | 0.223232 | Conversion cost exceeds the saved reload |
| Quantized-denominator/direct E2M1 classification | 1.720576 | Excess scalar control and ALU work |

HAO uses its 12.5% ALU/MUFU split on SM100. Reproducing that exact placement in
this nvcc kernel does not reproduce CUTE DSL's scheduling and increases the
critical path, so the TK default remains native MUFU.

## Remaining gap

Nsight Compute replay profiles of the final role-local TK path and HAO show:

| Metric | TK | HAO |
| --- | ---: | ---: |
| Replay duration | 329.76 us | 327.94 us |
| Issue active | 42.95% | 46.80% |
| Eligible warps per scheduler | 0.539 | 0.611 |
| Active warps per scheduler | 3.715 | 3.721 |
| Average warp latency | 8.65 cycles | 7.95 cycles |
| Tensor active | 11.14% | 11.15% |

TK now has essentially identical occupancy and tensor activity. It still
issues less often because fewer warps are eligible, but the replay duration is
within 0.6% and the wall-time result is within 1.0% of HAO.

The earlier diagnosis that nvcc could not reproduce HAO's role-local
allocation was incorrect. The old TK kernel executed one grouped
`setmaxnreg` decision before branching by warp role, so ptxas constrained
later role code against the wrong request. Applying 192/80/48 to that grouped
layout did spill. Moving the requests into the disjoint score-reader,
correction, issuer, loader, epilogue, and idle branches allows the exact
192/80/48 HAO split to compile with zero stack and zero spills.

This closes the structural TK-to-HAO porting gap. The remaining work is no
longer a missing HAO topology: it is reducing the cost of the full-FP4
softmax/P quantization and handoff relative to BF16, or selecting FP8 PV where
that tradeoff is preferable.

## Interim compact producer follow-up

Before the role-local fix, the TK producer constructed FP4 and scale
descriptors from compact low-word
state and kept the four NVFP4 Q/K scale copies plus both K64 QK MMAs in one
inline assembly block. This matched HAO's QK issue structure while preserving
the then-current spill-free `176/80/80` register split.

Long paired runs show a small but repeatable improvement over the prior compact
producer:

| Run | Prior producer (ms) | Fused QK producer (ms) |
| --- | ---: | ---: |
| Seed 0, 200/1000 ms | 0.202784 | 0.202624 |
| Seed 1, 200/1000 ms | 0.203936 | 0.203232 |

A separate `TK_HAO_DIRECT_FP4PV_FUSED_PV_ISSUE` switch also puts all four P/V
scale copies, the first K64 PV MMA, the P-tail wait, and the second K64 PV MMA
in one block. That path is correct and spill-free at `176/80/80`, but measures
0.203776 ms versus 0.202304 ms for split PV under the 100/500 ms seed-0
protocol. It therefore remains an explicit experiment rather than the default.

That fused-QK/split-PV checkpoint remained correct at seed 2:

- time: 0.202752 ms;
- output cosine against Torch BF16 SDPA: 0.981630;
- output RMSE: 0.004980;
- 128 registers per thread, zero stack, and zero spills.

## Role-local register checkpoint

The final role-local kernel is correct at seed 2 under a 100/500 ms run:

- time: 0.194848 ms;
- output cosine against Torch BF16 SDPA: 0.981630;
- output RMSE: 0.004980;
- 128 registers per thread, zero stack, and zero spills.

Two attempts to spend the recovered score-reader registers on longer-lived
score data were rejected:

| Experiment | Time (ms) | Result |
| --- | ---: | --- |
| Retain all four N32 score fragments | 0.282144 | Correct and spill-free, but nvcc serialized the long live-range schedule |
| Retain Q0 in addition to Q2 | 0.217184 | Correct and spill-free, but slower than quarter reloads |

The retained-Q2 quarter-reload schedule therefore remains the default. The TK
full-FP4 implementation is now a faithful performance reference for HAO's
structure: it is 0.98% slower than HAO NVFP4/NVFP4, while both full-FP4
implementations remain about 18.9% slower than the matched HAO BF16 route.
