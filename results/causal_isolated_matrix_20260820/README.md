# Causal FP4 FA4 isolated matrix — 2026-08-20

This directory is the post-baseline causal validation bundle for the isolated
branch `codex/causal-isolated-matrix-20260820`. The branch starts at
`0dbbfdb47b1bb32f03c25ab1013d058d7c379f62`. The current production-backward
artifacts record optimized source tip
`3535353a2f1983ecfd9ed89d404b63403f90ea31` and embed hashes for the harness,
extension, policy, CuTe source, and every applied patch.

The central verified result is:

- For prepared causal attention at D64, NVFP4 QK + MXFP4 PV (`d4q01`) becomes
  faster than NVFP4 QK + exact E4M3 FP8 PV at S=2048 and stays faster through
  S=16384 and across the tested S=4096 head counts. This is an MX-versus-FP8
  statement; CuTe BF16 is faster than both low-precision kernels at the longer
  prepared-attention shapes.
- The optimized retained-low-precision causal backward is faster than CuTe
  BF16 at every tested D64 sequence from S=512 through S=16384 and across all
  tested S=4096 head counts.
- S=8192 and S=16384 use the measured d1/p2 EX2, head-fast raster, detached
  FP8-P TMEM, and direct-TMA dK/dV policy. The old S=8192 crossover belongs to
  the superseded pre-optimization route.
- The experimental split-V publication preserves the deployed forward
  contract while publishing backward V directly from the projection
  accumulator. Across three report-grade 177-batch C4 seeds, both low-precision
  routes are 1.24–1.26x faster than BF16; exact FP8 has a small 0.03–0.32%
  full-step lead over MX, and final-loss ordering varies. This supports a
  short-horizon no-regression check, not MX quality superiority.

These statements are narrower than “FP4 is always faster.” The forward matrix
times prepared attention only; the backward matrix uses one matched,
E4M3-represented synthetic state; and the short training runs use the local C4
test corpus, not the missing canonical DCLM shard.

## Verification contract

Unless a section says otherwise, measurements are from one NVIDIA GB200
(SM100), batch 1, causal GQA ratio 4, head dimension 64. Forward timings are
rotating-provider CUDA-event medians. The primary backward boundary includes
every route-required clear but excludes forward and quantization. Training is
the 1,235,814,400-parameter, 16-layer Llama variant at sequence 4096 with 2D
NVFP4 projection-weight scaling, E4M3 QKV projection publication, K16
represented Q/K, and 1D MXFP4 V scaling.

Route names used below:

- **FP8:** NVFP4 E4M3-block16 QK + exact E4M3 FP8 PV.
- **MX:** NVFP4 E4M3-block16 QK + MXFP4 E8M0-block32 PV.
- **d4q01:** MX native density 4, native quarter mask `0x3`.
- **d4all:** MX native density 4, native quarter mask `0xf`.

## Verified causal forward matrix

Sources: [complete d4q01 summary](forward_matrix_d4q01_full/summary.json) and
the [S=16384 extension](forward_matrix_d4q01_s16384/summary.json). All 8/8
cases completed, and every provider passed the future-V causal leakage check.
`MX/FP8` is `FP8 time / MX time`, so values above 1 mean MX is faster.

| S | Q/KV heads | BF16 (us) | FP8 (us) | MX d4q01 (us) | MX/FP8 | FP8/BF16 cosine | MX/BF16 cosine |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 512 | 32/8 | 70.656 | 27.104 | 27.744 | 0.977x | 0.991634 | 0.978237 |
| 1024 | 32/8 | 68.384 | 28.416 | 33.312 | 0.853x | 0.991291 | 0.978002 |
| 2048 | 32/8 | 49.280 | 51.872 | 50.048 | 1.036x | 0.991041 | 0.977044 |
| 4096 | 32/8 | 94.624 | 139.712 | 119.328 | 1.171x | 0.990758 | 0.976845 |
| 8192 | 32/8 | 313.536 | 456.832 | 395.392 | 1.155x | 0.990497 | 0.976616 |
| 16384 | 32/8 | 1141.952 | 1649.024 | 1424.896 | 1.157x | 0.990650 | 0.976647 |
| 4096 | 16/4 | 50.272 | 88.736 | 75.904 | 1.169x | 0.990529 | 0.975653 |
| 4096 | 64/16 | 180.704 | 246.208 | 213.120 | 1.155x | 0.990901 | 0.976958 |

The non-monotonic BF16 medians reflect shape-specific compiled kernels. This
prepared-attention table excludes projection, QKV publication, RoPE,
allocation, output projection, backward, and optimizer; it must not be treated
as an end-to-end model timing table.

### d4q01 versus d4all

The focused S=4096, H=32/8 reruns are
[d4q01](forward_matrix_d4q01/summary.json) and
[d4all](forward_matrix_d4all_focus/summary.json).

| MX policy | MX (us) | FP8 (us) | MX/FP8 | MX/BF16 cosine | Leakage |
|:---|---:|---:|---:|---:|:---:|
| d4q01 | 119.392 | 140.736 | 1.179x | 0.9768449 | pass |
| d4all | 133.280 | 139.808 | 1.049x | 0.9768476 | pass |

d4q01 is 1.116x as fast as d4all (10.42% lower latency) while changing the
MX/BF16 cosine by only `2.68e-6`. This reproduces the previously remembered
fast-MX causal behavior at the isolated attention boundary.

`forward/s4096_h32_kv8_d64.json` is an earlier d4all-style focused result and
is superseded by the two explicit policy summaries above.

## Verified causal backward matrix

Sources: the [S=512–4096 sequence
sweep](backward/production_sequence_s512_s4096_current.json), independent
[S=8192](backward/production_sequence_s8192_current.json) and
[S=16384](backward/production_sequence_s16384_current.json) sweeps, and the
[S=4096 head-count sweep](backward/production_head_s4096_h16_h64_current.json).
All 8/8 cells completed. Timing uses seed 20260820, 13 warmups, 101 rotating
samples, and includes every route-required clear. The cosine column gives the
range of aggregate dQ/dK/dV cosine over three independent accuracy seeds.

| S | Q/KV heads | CuTe BF16 (us) | Retained lowp (us) | Lowp/BF16 speedup | Aggregate cosine range |
|---:|:---:|---:|---:|---:|:---:|
| 512 | 32/8 | 111.456 | 102.176 | 1.091x | 0.999671–0.999684 |
| 1024 | 32/8 | 149.504 | 138.560 | 1.079x | 0.999668–0.999682 |
| 2048 | 32/8 | 203.808 | 189.632 | 1.075x | 0.999677–0.999689 |
| 4096 | 32/8 | 347.104 | 319.808 | 1.085x | 0.998416–0.998481 |
| 8192 | 32/8 | 879.168 | 770.848 | 1.141x | 0.998496–0.998512 |
| 16384 | 32/8 | 2765.984 | 2621.376 | 1.055x | 0.998422–0.998525 |
| 4096 | 16/4 | 235.552 | 218.592 | 1.078x | 0.998493–0.998599 |
| 4096 | 64/16 | 575.392 | 528.448 | 1.089x | 0.998457–0.998533 |

The measured-shape dispatch is:

- S=512–2048: native d2/p0 EX2, key-fast raster, score-alias P TMEM;
- S=4096: selective d1/p2 EX2, key-fast raster, score-alias P TMEM;
- S=8192 and S=16384: selective d1/p2 EX2, head-fast raster, detached
  FP8-P TMEM;
- every row: one-lane direct-TMA dK/dV and workspace forward statistics.

This is an intrinsic represented-E4M3 backward-core comparison. Both routes
consume one exact E4M3-represented Q/K/V/dO state and causal BF16 forward
statistics; output is BF16 dQ/dK/dV. Forward, projection/publication,
quantization, optimizer, and parameter gradients are excluded. The retained
route reconstructs probability from scores/LSE through its E4M3 probability
path. A route-complete adapter that consumes forward-published MXFP4
probability payload and E8M0 scales is not present, so this table must not be
described as MXFP4 probability replay.

### Superseded diagnostics

The older [S=1024–8192 matrix](backward/d64_sequence_matrix_s1024plus.json)
used the pre-optimization policy and measured only 0.893x at S=8192. It is
retained as historical evidence for the original crossover, not as the
production result. The [direct-TMA
A/B](backward/direct_tma_ab_s4096_s8192.json) on that route measured direct
TMA 3.91% faster at S=4096 and 3.01% faster at S=8192 with effectively
unchanged accuracy, so direct TMA remains enabled. The later head-fast and
detached-P policy—not disabling direct TMA—removed the long-sequence
crossover.

The post-Aug-14 technical-report shape sweep based on `0dbbfdb` is likewise a
historical, broader-boundary comparison. Its approximately 0.95 dQ/dK cosines
must not be compared directly with this matrix: it covers the
model-distributed projection/represented-QK boundary, whereas this matrix is a
matched-E4M3 backward-core test. No cross-boundary accuracy improvement is
claimed.

`backward/d64_sequence_matrix.json` is deliberately excluded from every table
and hash manifest here. It has top-level status `running`: only S=512 finished
before the S=1024 record was interrupted. It is a partial artifact, not a
completed sweep.

## Experimental split-V publication

The split-V path keeps represented K16 Q/K and the 1D MXFP4 forward V payload,
but publishes the E4M3 backward V directly from the projection accumulator
instead of decoding/lifting/transposing the represented MXFP4 V.

### Contract validation

Primary source: [S=4096 publication validator](projection/split_v_publication_s4096.json),
`passed=true`. The earlier
[S=256 validator](projection/split_v_publication_s256.json) also passes.

- Forward Q/K payloads and scales, the contractual 1D MXFP4 V payload/scales,
  and backward Q/K are byte-identical to the deployed represented-K16 route.
- Split backward V is byte-identical to the direct-accumulator control.
- The intended split-V versus deployed backward-V difference has cosine
  0.9938975 and relative L2 0.110348 at S=4096.
- The D64 V scale page reserves 512 bytes but defines only 256 bytes. The
  validator compares those defined bytes and explicitly excludes unwritten
  padding.

### Composed causal boundary

Source: [S=4096 boundary A/B](forward_boundaries_d4q01_split_v_s4096.json),
14 rotating samples after 5 warmups. These rows use represented K16 Q/K and
store LSE where indicated.

| Boundary | FP8 (us) | Deployed MX (us) | Split-V MX (us) |
|:---|---:|---:|---:|
| QKV + RoPE + publication | 96.432 | 101.568 | 100.272 |
| Attention, store LSE | 141.200 | 119.728 | 119.920 |
| Prepacked publication + attention, store LSE | 234.672 | 220.848 | 218.112 |
| Full allocated one-layer attention boundary | 329.568 | 316.912 | 313.504 |

The split path saves 1.296 us in the publisher and 2.736 us in the composed
prepacked publication-plus-attention boundary versus deployed MX. At the full
allocated one-layer boundary it is 1.051x as fast as exact FP8 (4.87% lower
latency). A sample-and-position-controlled fixed-effect audit estimates split
versus deployed MX at -2.56 us (95% CI -3.88 to -1.24) and split MX versus
exact FP8 at -15.41 us (95% CI -17.12 to -13.69). Split and deployed MX
produce exactly equal forward output and LSE in this artifact.

One raw boundary-audit flag compares the entire allocated V-scale page and is
false because the unused D64 padding is allocator state. The dedicated
post-fix contract validator above compares only ABI-defined bytes and passes;
this is not a forward-numerics discrepancy.

### 1.2B, 24 unique C4 batches

The deployed and split runs use identical C4 train-token SHA256
`d53ae13b46e8b595fc6ef6d1b854e726c23b5801e0caf52cb746c42d6ed69781`
and validation-token SHA256
`bb2217d8e2d759886b86a9163418fdfb491005b1f15854a292d573eb57ccfbac`.
All recorded steps are finite.

The [split-V run](training/c4_unique_24_d4q01_split_v.json) gives:

| Route | Forward (ms) | Backward (ms) | Step (ms) | Speedup/BF16 | Final validation loss |
|:---|---:|---:|---:|---:|---:|
| BF16 CuTe | 28.279 | 34.410 | 69.705 | 1.000x | 8.471398 |
| Exact FP8 PV | 19.878 | 33.319 | 59.791 | 1.166x | 8.469778 |
| MXFP4 PV + split V | 19.768 | 33.322 | 59.677 | 1.168x | 8.470499 |

Within this run, MX is 0.55% faster in forward and 0.19% faster per step than
exact FP8 PV. Its final validation loss is +0.000721 versus FP8 and -0.000899
versus BF16. These are independent route-median point estimates; with only
eight complete execution-order cycles, paired timing intervals cross zero.

For context, the [deployed 24-step control](training/c4_unique_24_d4q01.json)
measured FP8 at 61.228 ms/step and MX at 61.428 ms/step; MX was 0.33% slower.
The absolute BF16 and low-precision times shifted between executions, so the
cross-run delta must not be attributed wholly to split V. The isolated
publication A/B above supplies the causal evidence; the training rerun verifies
that the desired MX-over-FP8 ordering is attainable end to end.

### 1.2B, 177 unique C4 batches

The completed [split-V run](training/c4_unique_177_d4q01_split_v.json) uses
the same 725,169-token training stream (SHA256
`f0199e90fb28b2eea59faf24824d0ac51de8cc7e887f492deb0440874aa61b8b`)
for every route. All steps are finite.

| Route | Forward (ms) | Backward (ms) | Step (ms) | Speedup/BF16 | Final validation loss |
|:---|---:|---:|---:|---:|---:|
| BF16 CuTe | 31.020 | 40.225 | 78.419 | 1.000x | 7.522856 |
| Exact FP8 PV | 22.008 | 33.841 | 62.932 | 1.246x | 7.520906 |
| MXFP4 PV + split V | 22.013 | 33.942 | 63.103 | 1.243x | 7.508097 |

Across the 177 paired batches, MX minus FP8 forward time has mean +0.003955 ms
and median +0.018944 ms: effectively a tie. In the corresponding
[non-split run](training/c4_unique_177_d4q01.json), the paired forward debit
was +0.156545 ms mean and +0.178432 ms median. Resampling the 59 complete
three-round execution-order cycles gives a 95% bootstrap CI of +0.1050 to
+0.2078 ms for the old gap, -0.0444 to +0.0516 ms for split, and -0.2270 to
-0.0779 ms for the difference-in-differences. This supports that split V closes
the prior integrated-forward debit. The longer split run does **not** make MX
the fastest full step: paired MX-minus-FP8 step time is +0.1253 ms (95% CI
+0.0504 to +0.1999), and the cross-run step improvement is not statistically
resolved. MX final validation loss is 0.012809 lower than FP8 and 0.014759
lower than BF16; one seed with eight validation batches supports
no-regression/loss parity, not a claim that MX improves model quality.

### 1.2B, three 177-batch C4 seeds with optimized backward

The report-grade v2 reruns for [seed
20260818](training/c4_unique_177_d4q01_split_v_auto_exp2_v2_seed20260818.json),
[seed 20260819](training/c4_unique_177_d4q01_split_v_auto_exp2_v2_seed20260819.json),
and [seed 20260820](training/c4_unique_177_d4q01_split_v_auto_exp2_v2_seed20260820.json)
contain 1,593 finite route-steps in total. Each seed uses a different
deterministic document split and token stream, so comparisons below are
within-run; absolute loss or timing must not be compared across rows. All
three use the current v3 automatic d1/p2 backward policy at S=4096.

| Seed | BF16 final val | FP8 final val | MX final val | FP8 speedup/BF16 | MX speedup/BF16 | FP8 time / MX time |
|---:|---:|---:|---:|---:|---:|---:|
| 20260818 | 7.522466 | 7.514042 | 7.508663 | 1.2569x | 1.2566x | 0.9997x |
| 20260819 | 7.373127 | 7.351347 | 7.369396 | 1.2464x | 1.2425x | 0.9969x |
| 20260820 | 7.294057 | 7.306645 | 7.300637 | 1.2413x | 1.2405x | 0.9994x |

Exact FP8 has the integrated route-median step lead in these reruns, but only
by 0.027%, 0.315%, and 0.062%. Paired mean MX-minus-FP8 step deltas are
+0.0978, +0.1287, and +0.0622 ms; simple normal 95% intervals span zero for
seeds 20260818 and 20260820, while seed 20260819 is +0.0051 to +0.2523 ms.
MX-minus-FP8 final validation loss is -0.00538, +0.01805, and -0.00601, so
loss ordering also varies. This is a short-horizon no-regression result; no
predeclared timing-equivalence test or model-quality superiority claim is
made. The isolated attention matrix above remains the causal evidence that
the MX prepared-attention kernel is faster than exact-FP8 PV from S=2048
through S=16384; the full training route includes projection/publication,
backward, optimizer, and other model work.

Each v2 artifact records the raw command, selected and resolved Python
executable, trainer SHA256, Git HEAD plus tracked-diff SHA256, corpus/token
hashes, backward extension SHA256, and the exact MX and FP8 forward-extension
SHA256 values. The three files therefore authenticate the binaries actually
selected by their CLI, unlike the earlier pre-v2 177-batch artifacts.

## Artifact identity

Key binary identities:

| Binary role | SHA256 |
|:---|:---|
| Historical clean backward/projection extension used by the original forward and non-split C4 line | `6c7c232534f1579ee3bc30efc3d4b553c0680b3caefb4a3b0d33bd5fea51c30a` |
| Current split-V projection and optimized backward extension | `aeed2603d40290b815218cc77142ddacda0c734384429f26c0d4a6a200fbe884` |
| Full S=4096 d4q01 MX forward extension used by current training | `ae648be9308a2ec74e9d6b70d3441d6af1d8452af85cad3b62150445785b9de9` |
| Full S=4096 exact-FP8 forward extension used by current training | `fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3` |
| S=16384 d4q01 MX forward extension | `cb0d5faed17dc317061884f8bdc3cacca7d1b8e79fe8b604fd349e0ec339f3ac` |
| S=16384 exact-FP8 forward extension | `f020fb2eed23a732ad03ae8bdf96b350069a1c510ca135c8b3a9eda05487e287` |
| Focused d4q01 MX forward extension | `f84ecb9a58a1022dfd7e49308e0b942f8d06e7e6877723afadc426dc3f91b027` |
| Focused exact-FP8 forward extension | `d71ee8eb6e9ff3ecac980594af47fc9b9f7cedeca339f8be8325254dbcede527` |

Key JSON identities at handoff:

| Artifact | SHA256 |
|:---|:---|
| `forward_matrix_d4q01_full/summary.json` | `7e8e7bbc27ee1969202a392105b703d251b997c0bd5f30090dc2e688d18cf60f` |
| `forward_matrix_d4q01_s16384/plan.json` | `b787f6b7bc4cfd571e23bd8a88c7677b5eba9deebfb70558243148573a1405fe` |
| `forward_matrix_d4q01_s16384/summary.json` | `ad97b651264229b39184d0ea290fc9797ec6c4003b0ea9d6497363c317cf7679` |
| `forward_matrix_d4q01_s16384/.../manifest.json` | `26813637cfd285f865a9cfae87ab8cf89c774e836f4382bad6af5d3af2cc7bf9` |
| `forward_matrix_d4q01_s16384/.../result.json` | `4eabee9456a6f1ea0d4331a36b229d49f9dab437b5776ed0150b1d2ceec87925` |
| `forward_matrix_d4q01/summary.json` | `f686f85bdd636689d646a432d4cb4de313c0a3a3b9a1f10bbb494ed1745c3c48` |
| `forward_matrix_d4all_focus/summary.json` | `8acda2b114d196f02989fa6550a833c1e4d7f51c275560bab9d7ddb71c0775c1` |
| `backward/production_sequence_s512_s4096_current.json` | `d8513047c327fa407a59c253293581167fe2ca18f2bd573f9f9693fee807895d` |
| `backward/production_sequence_s8192_current.json` | `48757343cd807d07d4bd6aac26d27109fa14432924a60f75a42629bcd487cf5c` |
| `backward/production_sequence_s16384_current.json` | `72d775ee372f4f1b48c6623157ae9dd1bbe7872f5c5b8d8505f44292ec83f4ce` |
| `backward/production_head_s4096_h16_h64_current.json` | `c0760cc23384c23d6f9dce1f49eb4069ce1488b550858d6472d90291dfa0889c` |
| `backward/d64_sequence_matrix_s1024plus.json` | `99297e20e5da47b59d3ca9061f23b480bf164b804d2b114e0e0fa10a05a576b1` |
| `backward/direct_tma_ab_s4096_s8192.json` | `eb971f3bd9c75940c05462b3dcfd9d168366cf84fc061458dba3d9e480daa00d` |
| `projection/split_v_publication_s256.json` | `9ebd2efad3daca83f6d56903857aa18d2f965ee2be2950c01030fac0facffeef` |
| `projection/split_v_publication_s4096.json` | `a2942c5e889dec168b3092231cc705512ae9f65d8e8fe37dd464779fd6f643f2` |
| `forward_boundaries_d4q01_split_v_s4096.json` | `2eacff0faf9750c469cf4fa3bfc459a73628d5ebd4736216524f3237c37a46c6` |
| `training/c4_unique_24_d4q01.json` | `7443361330a8c11a58e7920eff4d31b8de14d45e28a88c90ab69c69754606c0d` |
| `training/c4_unique_24_d4q01_split_v.json` | `0b8bd50dc39ca57d067bd2263a37c5426b1c0609e6d9c3013575e53a608f5153` |
| `training/c4_unique_177_d4q01.json` | `fa3e91d07cfccc224a84734fc10eb158f468cd3d084c610911cc0b9298f65bd6` |
| `training/c4_unique_177_d4q01_split_v.json` | `b1a0526101abc09eb34204fe89ab6b851cdf0f3dc0da05513bd3215c67c56b56` |
| `training/..._v2_seed20260818.json` | `21c47513b9ec4da0608e575f0405637e3090e5c05abbf9732b8817e74e95886c` |
| `training/..._v2_seed20260819.json` | `09d78ab99a48693156204bc57687d21d807082847be9ac46f1da695cab5d6d71` |
| `training/..._v2_seed20260820.json` | `cefccedb6794fad5ccfe58db957cd7f7d4d041a1790cc5f4cc0cbb500f9490cd` |

The temporary `.so` paths are not durable identifiers; use the hashes and the
embedded source/config hashes in each manifest.

### Commit lineage

- `0dbbfdb`: coherent low-precision training controls (branch base).
- `d172ed5`–`872b7d4`: matched causal forward worker and explicit projection
  artifact support.
- `fa67303`–`3aaf4a6`: causal backward harness, separate timing/accuracy draws,
  and serialized forward matrix runner.
- `2d0fae7`–`6333929`: d4q01/d4all control, exact-FP8 patch repair, and selected
  virtual-environment preservation.
- `5f65d09`: no-direct-TMA D64 control.
- `bb31884`: experimental split-V publication path.
- `5fcf4f8`: split-V runtime validator.
- `223b5d6`: causal forward boundary profiler.
- `2181b62`: packed-publication and D64 valid-scale-page audit fixes.
- `4e73875`: initial causal validation-matrix results.
- `4055ab6`–`6bc4a46`: measured-shape EX2 policy and validation.
- `a134153`–`637d4a7`: S=8192 head-fast/detached-P optimization and final
  validation.
- `bcea11c`–`3535353`: S=16384 dispatch and final validation.

The boundary JSON records HEAD `223b5d6` plus tracked harness changes; its
embedded worker SHA256 matches the version subsequently committed by
`2181b62`.

## Unresolved reconstruction and training work

### Canonical DCLM 2K rerun is blocked

Verified locally on 2026-08-20: `/tmp/dolma-dclm-data-00000.arrow` is absent,
and a read-only search of the workspace, `/tmp`, `/var/tmp`, `/root/.cache`,
and the root overlay found no exact-size Arrow file, MDS shard/index, or
plausible packed-token cache. The preserved authoritative manifest expects:

The configured read-only AWS profile was also checked without exposing its
credential material. Signed STS and S3 metadata requests return
`ExpiredToken`; anonymous listing is denied and unsigned metadata requests
return 403. No exact source-object key can therefore be verified or fetched
until the read-only session is refreshed.

- size: 469,413,680 bytes;
- SHA256: `846edebc43fa909016d6499240e2fb8101b8fd23bf6d9e07a69be347f708be93`;
- 88,669 rows, 88,638 unique rows, columns `text`, `id`, and `source_ds`;
- all `source_ds` values DCLM and all `id` values null;
- canonical 2K train-token SHA256:
  `fb8d70332aae02c39c8dd17893505a7286db9a11df6019bd92a3a09a58f86223`;
- canonical validation-token SHA256:
  `cc526d9251fb25f1c76e65cd36c214fc46dc1db1b5a2bc08cb652836a7689b54`;
- train/validation document-order SHA256:
  `3b8b2ae797661e1b32fe36a22227a2728e4bc5362fbd61a8bc98ea8668ee683e`
  and
  `ca20d4f109e2c4929e295cc410353b5b4bc8c17e2c207c44424f38156e7053c4`.

The converted MDS source is recorded as
`<private-object-store-location-redacted>`
with split `train`. Repository conversion scripts prove that MDS was produced
by recursively listing and lexicographically sorting the original HF Arrow
files without row reordering. They make a direct copy from the original HF
prefix the strongest low-disk recovery path. With fresh read access, first
fetch `dclm/train/hf_shards_info.json` from the converted MDS prefix and select
the entry whose row range is `[0, 88669]`; its key should identify the original
Arrow object under
`datasets/shuffled-olmo-mix-1124/snapshots/dummy/data/dclm/`. If that auxiliary
mapping is absent, authenticated-list the original prefix and filter candidates
to 469,413,680 bytes. In either case, accept only the file with the expected
SHA256. The exact object key remains an inference until metadata and hash
verification succeed. The last recorded read credentials are expired and an
unsigned metadata request returned 403. Fresh read access is required before
recovery and verification.
Repeated C4 or another corpus must not be substituted for the canonical run.
The current trainer has no checkpoint/resume path, so the requested 2K
comparison must be a fresh run.

The preserved 500-step train-token hash, useful for an intermediate gate, is
`e928a334304dc353b3051300e1e55d9b23ad854989f34370c289007748ec721c`.

### Remaining measurements

- The isolated forward and backward matrices and three report-grade 177-batch
  seeds are complete. The next training result is the fresh canonical 2K
  DCLM comparison, not another repeated-C4 substitute.
- If further long-sequence backward tuning is desired, collect NCU counters on
  the now-faster S=8192/S=16384 head-fast, detached-P route. The original
  crossover is resolved, so this is incremental optimization rather than an
  outstanding correctness/performance fix.
- Isolate the remaining full-training forward overhead: MX prepared attention
  is faster than exact FP8 from S=2048 upward, but the three complete S=4096
  training-route medians leave exact FP8 ahead by 0.03–0.32% per full step.
- Implement and validate a production MXFP4-probability replay adapter if the
  backward claim is meant to include forward MX payload reuse rather than the
  retained E4M3 reconstruction route.
- After restoring the exact DCLM shard, gate on corpus/token/document hashes,
  run a short split-V check, then execute the fresh BF16/FP8/MX 2K comparison.
- No new SFU kernel or paper-result evidence is generated by this matrix; that
  remains a separate optimization/report update.
