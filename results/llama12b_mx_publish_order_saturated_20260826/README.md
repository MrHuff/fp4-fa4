# Saturated Llama-1.2B bracket after the MX publication reorder

This bracket is the optimizer-step gate for projection/backward extension
`abd96216`, which issues the existing E4M3 backward-V publication before the
existing MXFP4 forward-V publication.  The measured end-to-end result is a
**tie, not an MX speedup**: MX is 0.459 ms slower at the pooled step p50 and
2.997 ms (0.539%) slower at the pooled mean.  That mean difference coincides
with a 3.325 ms difference in total `loss.backward()` timing even though the
two routes have the same serialized low-precision backward contract.  The
receipt cannot localize that total-backward difference to FA4.  If a common
backward time is imposed diagnostically, MX is 0.328 ms faster per step.

The defensible conclusions are therefore:

- the isolated publication reorder remains a real small improvement;
- this bracket does not establish a route-specific MX backward tax;
- it also does not establish an end-to-end MX throughput win; and
- a larger forward/publication saving is still required to rise above whole-
  step jitter reliably.

## Protocol

Four fresh processes were run in mirrored order `FP8-A, MX-A, MX-B, FP8-B` on
one NVIDIA GB200.  Every process used the same canonical checkpoint, packed
Dolma token stream, seed `20260825`, B16 x S4096, 16-layer Llama-1.2B preset,
three warm-up updates, and 20 measured updates.  B16 corresponds to 65,536
tokens/update.  Peak memory was 135.527 GiB allocated and 147.62--147.65 GiB
reserved out of 188.44 GiB physical HBM.

Both routes used fused RMSNorm/native-NVFP4 activation preparation, native
NVFP4 QKV projection with row-by-K16 activation scales and true 16-by-16
learned-weight scales, represented NVFP4 Q/K backward inputs, and direct E4M3
V for backward.  Only forward PV publication and attention differed:

- FP8 used E4M3 V and the FP8-PV causal FA4 entry point;
- MX used causal-interleaved block-32 MXFP4 V and the MXFP4-PV entry point.

The loss was `linear_cross_entropy(..., impl="torch_compile")` from the
`cut-cross-entropy` package.  This is its torch-compiled implementation, not a
custom CCE CUDA-kernel result.  The receipt records logical full logits and
does not claim that Inductor elided the full-logit buffer.  Outer-model compile
was disabled.

## Measured result

The pooled rows contain 40 measured updates per route.  Sustained throughput
is total tokens divided by total measured wall time.  The two-run `df=1`
intervals below deliberately treat processes, rather than individual updates,
as the replication unit.

| Route | Step mean / p50 (ms) | Decoder mean / p50 (ms) | Backward mean / p50 (ms) | Sustained / p50 tok/s | Useful MFU at pooled p50 | Final heldout loss, A / B |
|---|---:|---:|---:|---:|---:|---:|
| NVFP4-QK + FP8-PV | 555.582 / 558.636 | 150.195 / 149.739 | 368.466 / 370.202 | 117,951 / 117,314 | 42.859% | 8.09520 / 8.14814 |
| NVFP4-QK + MXFP4-PV | 558.579 / 559.095 | 149.724 / 149.669 | 371.791 / 372.229 | 117,318 / 117,218 | 42.824% | 7.95031 / 7.96080 |
| MX minus FP8 | +2.997 / +0.459 | -0.470 / -0.070 | +3.325 / +2.027 | -634 / -96 | -0.035 percentage points | -0.145 / -0.187 |

The per-process mean step differences, MX minus FP8, are +1.176 ms in pair A
and +4.817 ms in pair B.  Their mean is +2.997 ms, with a conservative 95%
`df=1` interval of [-20.14, +26.13] ms.  For decoder forward, both process-
mean differences favor MX (-0.616 and -0.325 ms), but the corresponding
two-process interval is [-2.32, +1.38] ms.  Averaging the two per-run p50s
gives MX-minus-FP8 differences of +1.150 ms for the step and only -0.011 ms
for decoder forward.  These runs are not sufficient to resolve a sub-
millisecond whole-step effect statistically.

## How the component timings compose

The arithmetic mean decomposition is exact apart from displayed rounding:

| Timed stage | FP8-PV (ms) | MXFP4-PV (ms) | MX minus FP8 (ms) |
|---|---:|---:|---:|
| Decoder forward | 150.195 | 149.724 | **-0.470** |
| Backward | 368.466 | 371.791 | +3.325 |
| Torch-compiled CE forward | 25.155 | 25.291 | +0.137 |
| Gradient clipping | 3.700 | 3.719 | +0.019 |
| Optimizer | 8.067 | 8.053 | -0.014 |
| Full optimizer step | 555.582 | 558.579 | +2.997 |

Canonicalized `backward_contract` JSON is identical for all four receipts,
SHA256
`3ac1d0da18ccb326b70cacc0121732df0dc19d0b418dc977791809c5fb7275f1`.
It specifies the same direct-TMA low-precision FA4 backward, represented
NVFP4 Q/K source, projection-accumulator E4M3 V source, BF16 projection dgrad,
and schedule.  The source change at commit `1286592` is forward-publication
call order only.  Thus the observed total-backward delta does not establish
that MX runs different FA4 backward work.  Replacing the route means with a
common backward mean leaves the measured non-backward stages 0.328 ms in MX's
favor.  That is a diagnostic normalization, not a claimed measured speedup;
an isolated receipt or profiler trace is required to localize the delta.

The immediately preceding isolated B16 factorial is higher-resolution
evidence for the intended component: MX saves 55.500 us at the prepared
projection-plus-attention boundary with this binary, while its remaining
publication premium is 30.441 us.  See
`results/llama12b_mx_publish_order_20260826/README.md`.

These four receipts contain no pre-reorder end-to-end MX control, so they do
not independently measure the causal effect of the reorder.  That attribution
comes only from the paired pre/post isolated factorial; this bracket is the
post-change route comparison.

## Fresh profiler diagnostic

A separate one-update Nsight Systems capture was collected for each route
after the headline bracket, with the same B16 shape and exact projection/
backward binary.  It is a launch-identity and utilization diagnostic, not a
throughput sample: profiler instrumentation changes the absolute timings.
Kernels are attributed to CPU-side NVTX ranges through CUDA-runtime
correlation IDs.  Comparing GPU timestamps directly with CPU NVTX timestamps
would be invalid for asynchronous launches and, in this capture, would omit
most of backward.

Both captures contain 1,730 kernels, of which 907 are in `backward_total`.
The ordered backward signatures are identical across FP8 and MX: kernel name,
grid, block, register count, and static/dynamic shared-memory fields hash to
`f1c727743474c3c3d5e4291882b5cc2081ea981791936e6a6b2c168d5f451fb4`
for both routes.  The full-step signatures differ at exactly 32 positions:
one QKV publisher and one attention kernel in each of 16 layers.  This is
direct evidence that the two routes dispatch the same backward work and that
their intended forward differences propagate into the measured model.

| Diagnostic counter | FP8-PV | MXFP4-PV |
|---|---:|---:|
| Kernel-union busy / first-to-last-kernel span | 98.431% | 98.322% |
| Largest internal no-kernel gap | 0.674 ms | 0.726 ms |
| Mean GR active | 99.357% | 99.234% |
| Mean SMs active | 93.483% | 93.386% |
| Mean tensor active | 43.790% | 43.793% |
| Mean GPC clock | 1748.7 MHz | 1703.9 MHz |

The GPU is therefore near-continuously fed, but these traces miss the older
94% mean-SMs-active saturation gate by 0.52 and 0.61 percentage points.  They
must not be relabeled as passing that stricter gate.  MX's mean GPC clock is
2.56% below FP8's over the whole kernel window; inside `backward_total`, it is
1628.5 MHz versus 1694.6 MHz for FP8, 3.90% lower.  Although the identical
backward launch sequence sums to 363.507 ms for FP8 and 372.177 ms for MX in
the instrumented traces, that clock difference is a material confounder; the
trace does not support a distinct MX backward implementation or tax.

Within decoder forward, the instrumented MX attention kernels save 1.490 ms
across 16 layers while the MX QKV publishers cost 0.509 ms more, a net 0.981
ms kernel-time saving at that boundary.  Decoder kernel time is 1.031 ms
lower for MX overall.  This agrees in sign with the headline decoder means,
but the profiled numbers are not substituted into the throughput result.
The read-only analysis and every counter are recorded in
`profile_analysis.json` and can be reproduced with
`tk_fa4/lowp_fa4_bwd/analyze_saturated_nsys_pair.py`.
The route/configuration receipts are `profile_fp8.json` and
`profile_mx.json`; they bind both captures to B16 x S4096 and projection
binary `abd96216`.
Both SQLite exports contain Nsight's generic warning that not all events might
have been collected.  There is no explicit buffer-drop diagnostic: identical
row/event counts, exact top-level stage closure, all expected 16-layer leaf
ranges, and equal kernel counts provide strong internal closure, but the
generic warning remains a capture caveat.

## Numerical scope

All 92 warm-up and measured records are finite.  The fixed BF16 reference
finishes at heldout loss 8.04929.  FP8 finishes at +0.04591 and +0.09884
relative to that reference; MX finishes at -0.09898 and -0.08849.  The MX
numbers are encouraging for this short diagnostic, but 23 optimizer updates
per process are far too short to rank convergence or accuracy.  They support
only the decision to proceed to matched long-run pretraining canaries.

## Authenticated artifacts

- Projection/backward extension: 23,995,968 bytes, SHA256
  `abd96216925acda6042df36dcb45dbacfdab24becff6f0a379911c176c775054`.
- FP8 forward extension: SHA256
  `88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208`.
- MX forward extension: SHA256
  `cc06fe4337fdc3a7c900f81d68fabc4a8e0c375ea536fbe6405754237a393717`.
- Backward control: SHA256
  `cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`.
- Canonical checkpoint: SHA256
  `2760f5eb47fd0241317dfd69bd0e2d906909d948d81a5a93f0fd371944f0d2bc`.
- Packed tokens: SHA256
  `0e7c735ad8794429330a23dada1a2cd26d3abe955ce4c46d31e40e161c55fd16`.

Raw receipt hashes:

- `fp8_a.json`: `3dd0314153b60d3a2a5ca07faf49844f90823d078bbce2717b043f2627c4fc1a`
- `mx_a.json`: `e880054c94fb678ee39c0d474835d82b263815e2ec062f85a7ce3e079caf0750`
- `mx_b.json`: `6d7070847e7049865e70a3651ca161e1bccb98ffa441564993b8fc41fc6de483`
- `fp8_b.json`: `1f62ce778054cf442365d03483d2a0f378212f6a829b5486546065015d6424a7`

The four approximately 25-MiB sampled-tensor sidecars remain outside Git;
their sizes and SHA256 identities are recorded in the raw receipts.  No
external profiler was attached to the headline runs, so the useful-MFU column
is the harness's BF16-equivalent arithmetic estimate, not an SM-active
counter.  The fresh diagnostic traces above are separate post-bracket
processes and are not substituted for the uninstrumented timing samples.

The raw profiler artifacts remain outside Git.  Their authenticated
identities are:

- FP8 `.nsys-rep`: 898,591 bytes, SHA256
  `8104084fb8919d58be333141a28d41898702fa337f297539fac46c1f4d65fcfe`;
- FP8 SQLite: 5,488,640 bytes, SHA256
  `43300b390ef0ce643150a71838c9f21ff0ee35bbcc09c3b6d79bdc4c94992827`;
- MX `.nsys-rep`: 890,853 bytes, SHA256
  `0dafbd098f7b6d8d618eb275bc0d3f431c56e9618ad9386864c03110c123da0c`;
  and
- MX SQLite: 5,451,776 bytes, SHA256
  `7ece064bb36baf5d6ddbe3d169fae9a62a50851875f891a888cb50037846201c`.

The committed profiler companion receipts have SHA256 identities
`a7cb00c23cdf6994d93276a90eba19fbcc30c7f8f9951b9d72a1c8182eafaecd`
for FP8 and
`7b18fb8a5df76c2ab3bdf5839c8bd0949b24a35c8d2296ac6548aa5882f06ab0`
for MX.  Their approximately 25-MiB sampled-tensor sidecars remain outside
Git and are authenticated inside those receipts.

The projection artifact field is explicitly `caller_declared`: the harness
re-authenticates the exact binary bytes, but does not embed a projection-source
hash that cryptographically binds those bytes to commit `1286592`.  The
isolated validator/factorial use the same binary identity and the checked-out
source contains the reorder, but that relationship remains build provenance
rather than a self-attested property of these four JSON files.

## Decision

Keep the publication reorder because its isolated improvement and byte-level
correctness gate are both positive.  Treat FP8-PV and MXFP4-PV as tied at the
current saturated optimizer-step boundary.  Use the old 8B B1 SlimPajama BF16
FA4 curve only as a historical numerical comparison; it is not a throughput
denominator for the new low-precision runs.
