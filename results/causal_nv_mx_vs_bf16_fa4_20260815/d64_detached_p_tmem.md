# D64 detached P/PV TMEM layout

Date: 2026-08-15

## Question

D64 uses only 384 of the 512 physical TMEM columns required by the D128
kernel. This experiment tests whether the spare final 128 columns are more
useful as dedicated scale/P storage than the D128-style score-overwrite
scheme.

## Layout

The original D64 allocation was:

```text
[  0,128)  score S0, with packed P0 at [64,80)
[128,256)  score S1, with packed P1 at [192,208)
[256,320)  output O0
[320,384)  output O1
[384,512)  unused
```

Packed P occupies 16 physical columns, not 64. The retained layout uses one
64-column slab per query stage:

```text
[384,448)  stage 0: P [0,16), V scale [16,32), P scales [32,64)
[448,512)  stage 1: P [0,16), V scale [16,32), P scales [32,64)
```

Q/K scales remain in the opposite score bank. A stage needs 32 more columns
for Q and K scales, so all Q/K/P/V scales plus P do not fit in one 64-column
slab. Temporally overlaying every scale page in the spare tail was valid but
did not improve performance.

## Scheduler change

Relocation alone was insufficient. The retained scheduler waits until the
next K slot and represented denominator are ready, then emits the stage-local
`PV -> QK` pair back-to-back. This is enabled only for the guarded D64 layout;
D128 and all other routes retain their prior issue order.

## Isolated comparison

The following paired runs load the candidate and old D64 control in one
process, use identical tensors, and report median `triton.testing.do_bench`
timings after 200 ms warmup over a 1500 ms measurement window.

| S | Hq/Hkv | Old D64 (ms) | Detached P/PV (ms) | Change |
|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 0.059744 | 0.057632 | -3.54% |
| 4096 | 32/8 | 0.156736 | 0.148512 | -5.25% |
| 8192 | 32/8 | 0.293152 | 0.274752 | -6.28% |
| 4096 | 64/8 | 0.157408 | 0.149504 | -5.02% |

Every candidate output is bitwise identical to the old kernel. Every causal
leakage test preserves the protected output prefix and LSE bitwise.

## BF16 comparison

Fresh full-provider runs use the same process and inputs for TK NV/MX and HAO
BF16 FA4. These are single 1500 ms timing windows and therefore supplement,
rather than replace, the three-repeat headline table in `README.md`.

| S | Hq/Hkv | Detached P/PV (ms) | BF16 FA4 (ms) | Speedup | Cosine | Relative L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 0.057696 | 0.081920 | 1.420x | 0.950939 | 0.317329 |
| 4096 | 32/8 | 0.147904 | 0.199680 | 1.350x | 0.950577 | 0.317902 |
| 8192 | 32/8 | 0.468992 | 0.600480 | 1.280x | 0.949400 | 0.322035 |
| 4096 | 64/8 | 0.252480 | 0.325312 | 1.288x | 0.949415 | 0.321690 |

## Profile read

At S4096, Hq32, Hkv8, D64, NCU changed as follows:

| Metric | Old D64 | Detached P/PV |
|---|---:|---:|
| Kernel time (us) | 156.576 | 148.512 |
| Tensor-core active (%) | 19.1723 | 19.6294 |
| Tensor pipe (%) | 8.0866 | 8.4985 |
| Issue active (%) | 46.3751 | 48.3241 |
| Eligible warps/cycle | 0.5972 | 0.6158 |
| Long-scoreboard stalls | 4.4816 | 4.1229 |
| Barrier stalls | 0.0419 | 0.0430 |

The gain comes from fewer P/score lifetime conflicts and a denser issue
sequence, not from reducing barrier stalls. The release build uses 128
registers, one hardware barrier, no stack, and no spills.

## Rejected controls

- Moving Q/K/P/V scale pages to the spare tail while retaining old waits was
  1.5% slower.
- Removing the now-unnecessary alias waits from that scale-only layout was
  2.6% slower. Tensor-core active fell from 19.17% to 16.39%.
- Rotating the detached layout to `QK -> PV` was correct but 1.15% slower than
  the retained `PV -> QK` order.

## Exhaustive third-payload and scale-overwrite sweep

The 128-column tail can be compacted enough to hold a third packed-P payload:

```text
[384,400)  P payload 0
[400,416)  P payload 1
[416,432)  P payload 2
[432,448)  shared V-scale page
[448,480)  stage-0 P-scale bank
[480,512)  stage-1 P-scale bank
```

This is a three-payload layout, not a complete three-operand buffer. Each P
payload occupies 16 columns, while the corresponding block scales occupy 32
columns. The tail therefore fits three payloads but only two live P-scale
banks. Producing the third usable operand still requires a legal handoff of a
scale bank from PV back to the P producer.

All distinct layouts and handoff mechanisms that fit in the existing 128
columns were compiled and exercised at S4096, Hq32/Hkv8, D64, causal. A mode
was accepted only if output, protected causal prefixes, and LSE state all
matched the release kernel. Timings for valid modes use long paired windows;
timings from racing modes are not treated as performance results.

| Mode | Configuration | Correct | Candidate/control (ms) | Change |
|---:|---|:---:|---:|---:|
| 0 | Two independent 64-column slabs (release) | yes | 0.147776 / 0.147776 | 0.00% |
| 1 | Compact two-payload layout, no early reuse | yes | 0.147744 / 0.148000 | -0.17% |
| 2 | Third P for stage 0, completion token, `PV -> QK` | yes | 0.160288 / 0.147936 | +8.35% |
| 3 | Stage-0 third P, `QK -> PV`, full release | no, race | - | - |
| 4 | Mode 3 with first/tail split releases | no, leakage | - | - |
| 5 | Stage-1 third P, `QK -> PV`, full release | yes | 0.156000 / 0.147872 | +5.50% |
| 6 | Mode 5 with first/tail split releases | yes | 0.158720 / 0.147808 | +7.38% |
| 7 | Stage-0 third P using the existing V-slot token | no, LSE | - | - |
| 8 | Stage-1 third P using the existing V-slot token | yes | 0.155600 / 0.147840 | +5.25% |
| 9 | Stage-0 `QK -> PV` without a release token | no, leakage | - | - |
| 10 | Stage-1 counterpart of mode 9 | no, leakage | - | - |
| 11 | Retired QK scratch for V scales plus full release | no, leakage | - | - |
| 12 | Retired QK scratch without a release token | no, leakage | - | - |
| 13 | Mode 11 with one waiter and warpgroup handoff | no, leakage | - | - |
| 14 | Stage-1 counterpart of mode 13 | no, LSE | - | - |
| 15 | V scales overwrite retired first-half P scales | no, leakage | - | - |
| 16 | V scales overwrite retired tail-half P scales | no, LSE | - | - |
| 17 | Mode 15 with fences before and after overwrite | no, leakage | - | - |
| 18 | Mode 16 with fences before and after overwrite | no, leakage | - | - |

Mode 1's 0.17% difference is within run-to-run noise and does not change the
schedule. Mode 8 is the fastest robust configuration that actually uses the
third P payload, but it is 5.25% slower than mode 0. Seeded correctness runs
at seeds 0, 1, and 17 were all bitwise exact for mode 8.

### Why the legal third buffer loses

NCU isolates release mode 0 from the best legal third-payload mode 8:

| Metric | Mode 0 | Mode 8 |
|---|---:|---:|
| Kernel time (us) | 147.360 | 154.592 |
| Tensor-core active (%) | 19.6273 | 18.1796 |
| Tensor pipe (%) | 8.4890 | 8.1591 |
| Issue active (%) | 48.7447 | 47.8272 |
| Eligible warps/cycle | 0.6229 | 0.6158 |
| Wait stalls | 0.7368 | 0.7748 |
| Long-scoreboard stalls | 4.0910 | 4.0918 |
| Barrier stalls | 0.0426 | 0.0404 |
| Registers/thread | 128 | 128 |
| Static shared memory (bytes) | 416 | 416 |

The regression is not caused by extra hardware barriers, spills, static
shared memory, or DRAM traffic. The scale-bank lease adds a dependency and
wait on the issue path; issue eligibility and tensor-core activity both fall.
The useful overlap unlocked by the third payload is smaller than this
ownership cost. Removing the handoff or overwriting scales earlier produces
incorrect output or LSE state.

## Conclusion

The spare D64 tail is useful, but not as a scale dump by itself. It is large
enough to detach both live packed-P operands and their P/V scales from the
score banks. This removes one ownership conflict and makes a better issue
schedule possible.

The release remains a true two-operand buffer. A third packed payload does fit
after compaction, but a third independently ready P operand does not: it also
needs another 32-column P-scale bank. The exhaustive ownership and overwrite
sweep found no legal schedule that recovers the handoff cost. A genuine
three-operand pipeline therefore needs a smaller physical scale descriptor or
more TMEM, not only different addresses. D128 cannot reuse this layout on
GB200 because its two 128-column output accumulators already consume
`[256,512)`.
