# Corrected D=128 TK FA4 closure (2026-08-29)

## Decision

The corrected D=128 path is ready for an 8B training trial. The production
candidate uses monolithic TK attention in both directions: the forward route is
the direct TK causal GQA kernel and backward is native TK v501 E4M3. At
`B=2, S=4096, Hq=32, Hkv=8, D=128` on one GB200, the matched 32-layer 8B
BF16/FP8/v501-MX bracket and the later v503 A/control/B bracket measured:

| Route | Step p50 (ms) | Decoder fwd p50 (ms) | Backward p50 (ms) | tokens/s | Speedup vs BF16 control |
|---|---:|---:|---:|---:|---:|
| BF16 CuTe, mean of A/B | 489.821 | 138.378 | 278.063 | -- | 1.000x |
| NVFP4 QK + FP8 PV + TK v501 bwd | 434.014 | 120.929 | 240.223 | 18,874.95 | **1.1286x** |
| NVFP4 QK + MXFP4 PV + TK v501 bwd | 435.992 | 121.361 | 240.307 | 18,789.32 | **1.1235x** |
| NVFP4 QK + MXFP4 PV + TK v503 straight-MX bwd, A/B mean | 433.539 | 119.151 | 240.977 | 18,895.65 | **1.1298x vs prior BF16** |

The v503 row is from the later A/control/B experiment; its 1.1298x entry is an
arithmetic comparison with the prior 489.821 ms BF16 bracket, not a newly
interleaved BF16 measurement.

The two BF16 anchors were 490.001 and 489.641 ms (0.0735% end-to-end
drift), so the bracket is stable. FP8 and MX use the same authenticated v501
backward artifact; their measured backward difference is only 0.084 ms
(0.035%). The retained-dual-v v501 MX route is 1.978 ms, or 0.456%, slower end
to end than FP8. It should therefore be described as tied at the full-step
level, not as an MX speed win.

Launch recommendation: use BF16 and FP8-PV as the long-training controls. v503
now establishes that straight row-major MXFP4 V is a viable backward operand,
but not a material end-to-end speedup: its A/control/B bracket was 0.11% faster
at p50 and 0.06% slower in sustained time than the fresh v501 control. Keep it
as a simultaneous numerical ablation if compute is available; do not advertise
the whole-step result as an MX speed win until a longer convergence run and a
margin larger than run-to-run noise exist.

## What was corrected

### Probability-statistic ABI

Native D=128 v501 declares

```
lstat = 8 - LSE * log2(e)
```

The fused dO/stat publisher must therefore add the `+8` log2 lift for the
native D=128 route. This is an ABI correction, not a tuning gain: omitting the
offset changes reconstructed probability by `2^-8 = 1/256`. The corrected
projection artifact is the one authenticated in both matched 8B candidates.
The native kernel's separate gradient epilogue scale is also recorded as
`1/256`; the two factors serve different parts of the contract and must not be
collapsed.

### B2 output-clear elision

For exact `B=2, S=4096`, v501 dispatches v490. Its receipt records dK/dV as
complete unique-writer outputs with no destination preclear requirement. The
physical reset policy consequently needs only dQ rather than dQ+dK+dV, and the
caller-owned main entry point permits that remaining clear to be satisfied by
an already-zero caller buffer. The post-change one-layer receipt verifies the
composed clear-elided route, but does not isolate the clear's latency by itself.
Fallback shapes remain additive and must retain their full reset semantics.

## Why isolated MX is faster but end-to-end MX is tied

At the same D=128 production shape, the isolated attention-forward comparison
measured:

| PV core | Median (us) | Output cosine vs BF16 | Output relative L2 | LSE cosine vs BF16 |
|---|---:|---:|---:|---:|
| FP8 PV | 295.008 | 0.999453 | 0.033072 | 1.000000 |
| MXFP4 PV | 259.072 | 0.991812 | 0.127745 | 1.000000 |

Thus the MX attention core is **1.1387x** faster than FP8, saving 35.936 us
per layer call. Across 32 layers that is only about 1.150 ms, or 0.265% of the
434 ms FP8 step.

The matched route contracts explain where that small core gain goes:

- FP8 publishes an E4M3 V payload which is also the E4M3 V source for v501.
- MX uses `retained_dual_v`: MXFP4 V for forward plus an E4M3 accumulator V for
  the shared v501 backward.
- Both routes use the exact same native TK E4M3 backward binary.
- In the full decoder, MX is 0.432 ms slower than FP8, despite its faster
  isolated attention core; at the full-step boundary it is 1.978 ms slower.

**Inference from the original v501 route contracts and timings:** the MX tensor-core
path is not the limiting problem. Extra MX+E4M3 V publication/integration work
erases a core advantage whose theoretical whole-step contribution is already
small. The later v503 experiment below establishes straight MXFP4 V as a valid
backward operand, while confirming that its whole-step performance is tied.

## Numerical evidence and limits

The native v501 validator passed for B1 and B2. At the mandatory S=128 BF16
reference check, DQ/DK/DV cosine ranges were 0.999321/0.999321/0.999707 for B1
and 0.999300/0.999300/0.999709 for B2; norm ratios were 0.9978--0.9989. At
S=4096 all target gradients were finite and nontrivial, while exact-zero dO
produced exact-zero DQ/DK/DV. Reset-inclusive v501 medians were 302.432 us at
B1 and 504.416 us at B2.

The corrected one-layer B2 FP8 composition measured Q/K/V gradient norm ratios
of 0.9958/0.9966/1.0152 versus BF16, with cosines
0.8117/0.8169/0.8591. Sampled-logit cosine was 0.9109. Its one-layer step was
40.733 ms versus 42.316 ms for BF16 (1.0389x).

The matched full-depth synthetic-token harness remained finite but is not
convergence evidence:

| Model/route | Initial global grad-norm ratio vs BF16 | Layer-0 Q/K/V grad-norm ratios | Short heldout loss, initial -> final |
|---|---:|---|---|
| 8B FP8 PV | 0.9207 | 0.9784 / 0.9774 / 1.0036 | 12.5832 -> 12.5745 |
| 8B MXFP4 PV | 0.9335 | 0.9945 / 0.9946 / 1.0235 | 12.5985 -> 12.5693 |
| 1B FP8 PV | 1.0055 | 1.0015 / 1.0043 / 1.0072 | 12.1716 -> 11.9388 |
| 1B MXFP4 PV | 1.0292 | 1.0417 / 1.0444 / 1.0588 | 12.1746 -> 11.9360 |

For the 32-layer 8B random-initialization comparison, sampled per-parameter
gradient cosines are low (approximately -0.030 to 0.053 across the reported
FP8/MX samples) even where norms match. Sampled-logit cosine is 0.3203 for FP8
and 0.2876 for MX. These facts prevent a claim of BF16-equivalent numerics from
this local test. Long real-data training is the required next test.

As an integration cross-check, the saturated D=64 1B bracket at
`B=16, S=4096` measured 1.0937x for FP8 and 1.0952x for MX versus the mean BF16
anchor. MX was 0.840 ms faster end to end there, while backward differed by
0.749 ms. This supports treating the two routes as approximately tied after
integration rather than assuming the isolated MX core advantage transfers
directly to a whole training step.

All matched runs used synthetic tokens, one GB200, three warmups, 20 measured
updates, fused AdamW, and torch-compiled cut-cross-entropy. The 8B shape was
B2/S4096/D128 with 32 layers; the 1B cross-check was B16/S4096/D64 with 16
layers. Neither receipt measures multi-GPU communication or real-data
convergence.

## Backward schedule ceiling

A bounded Nsight Systems plus GB200 GPM profile of the exact v501 B2/S4096
artifact measured the v490 `owner4_kernel` itself at 480.528 us median over
100 no-clear launches. The kernel uses 512 threads, 128 registers per thread,
and 167,184 bytes of static shared memory. Both registers (65,536 per CTA)
and shared memory independently restrict it to one CTA per SM, giving 25%
theoretical warp occupancy and 21% measured occupancy.

The steady-state counters were 95% SM activity, 30% tensor/HMMA activity, 9%
DRAM activity, and 15% FP32 activity. These counters show that v501 is not at
the GB200 tensor-compute or DRAM-bandwidth ceiling. It is near the ceiling of
the present dependency-heavy one-CTA schedule: eight compute warps, four
reducer warps, three single-lane issue/publish roles, one idle warp, and
causal CTAs with unequal amounts of query work. Direct issue-stall counters
were unavailable because this node denies NVIDIA performance-counter access,
so the dependency diagnosis is an inference from the verified resource,
activity, schedule, and timing evidence rather than a sampled stall
breakdown.

Small post-v490 schedule changes did not expose incremental headroom. In
matched isolated comparisons, a two-stage publisher measured 513.984 us
versus 502.880 us for v490, partial D1 prestaging measured 548.928 us versus
503.840 us, and first-dstat prestaging measured 511.136 us versus 506.336 us.
Low-single-digit tuning remains possible, but a material gain requires a
structural change such as a smaller resource footprint, better causal load
balance, or a different gradient-publication pipeline.

## Straight-MXFP4 V experiment boundary

The existing projection producer can replace the backward-oriented E4M3 V
with a backward-oriented row-wise MXFP4 V payload. At B2/S4096, an
authenticated producer-only ABBA/BAAB comparison measured 251.274 us for the
retained MX-forward plus E4M3-backward publication and 206.421 us for the
MX-forward plus MX-backward candidate, a 44.853 us (1.2173x) saving per layer.
A cache-scrubbed comparison measured a 47.938 us saving. Across 32 layers,
44.853 us/layer is approximately 1.435 ms per step before accounting for any
backward-kernel change.

This candidate does **not** reuse one physical V payload. Forward PV needs a
feature-major payload and scales grouped over 32 sequence values, whereas dP
needs row-major V and scales grouped over 32 depth values. SM100 mixed
`mxf8f6f4` also requires E2M1 to be K-major. Descriptor transposition cannot
repair either the payload orientation or the scale semantics, so the bounded
candidate retains two MX physical orientations while removing the more
expensive MX-plus-E4M3 dual-format publication. True single-payload reuse
would require a different quantizer and tile-scale contract.

The exact B2/S4096 numerical screen for the current row-wise MSE-selected
MXFP4 producer measured V cosine 0.993596 and relative L2 0.112996 versus the
current E4M3 V. With Q, K, dO, and causal probability fixed, the corresponding
readable dP/dS/dQ/dK relative-L2 errors were 0.113003, 0.112915, 0.111675, and
0.112770. dV is independent of V in attention backward and can remain on the
bitwise-identical v501 path.

Experimental v502 implements that block-scaled mixed dP path and authenticates
the numerical model. Against a readable oracle using the represented MX V,
lifted E4M3 probability/dS, and BF16 gradient publication, v502 measured dQ
cosine 0.999991 with relative L2 0.004269 and dK cosine 1.000000 with relative
L2 0.000153; dV was bitwise identical to v501 and the exact-zero dO gate
passed. The route is therefore a correct control rather than a numerical
failure. It is not a performance candidate: no-clear latency was 620.512 us
versus 487.456 us for v501, and reset-inclusive latency was 631.808 us versus
498.688 us. The 133 us backward penalty is almost three times the producer's
44.853 us saving. The regression comes from serializing score reuse around the
scale-TMEM publication required by the block-scaled MMA.

These are bounded numerical screens, not convergence evidence. v502 remains
experimental and must not be selected by production dispatch.

A follow-up readable screen found a more promising consumer contract which
does not require scale TMEM. Requantizing the existing four block32 scales to
one common E8M0 scale per complete D128 V row gives dP/dS/dQ/dK relative-L2
errors of 0.116919/0.116984/0.115709/0.116775 with the cheap common-maximum
rule. Selecting the common scale by MSE improves those values to
0.115756/0.115712/0.114435/0.115515; it chooses the maximum of the four
existing scale codes for 97.57% of rows. This can be implemented during the
already-required compact-to-padded shared-memory restage with a small
power-of-two E2M1 lookup table, then apply the one row scale in the existing
dP/dstat epilogue.

Experimental v503 implements that row-maximum contract without changing the
producer. It preserves v490's tensor-issue overlap and dV path, uses an
unscaled E2M1-by-E4M3 MMA, and applies `(2/3)*2^(common_e8m0-127)` to dP before
the existing x16 dstat centering. Two independent GB200 checks measured v503
about 13.4--14.3 us slower than v501 (approximately 501.5--501.8 us versus
487.4--488.0 us), while dV remained bitwise identical and exact-zero dO
produced exact-zero gradients. Relative to v501, dQ/dK cosine was
0.98938/0.98974 and relative L2 was 0.14557/0.14308. These differences are the
intended represented-MX approximation, not a layout failure. The exact
readable oracle also passed: kernel-versus-represented-row-scale-oracle dQ
cosine/relative-L2 was 0.999991/0.004267, dK was
0.99999997/0.000160, dV was bitwise unchanged, and the exact-zero gate passed.

The complete producer-plus-backward chain was measured directly rather than
only inferred from separate medians. In an independently conditioned
ABBA/BAAB run, retained dual-format MX publication plus v501 took 758.288 us,
while backward-oriented MX publication plus v503 took 724.992 us. The raw
saving was 33.296 us/layer and the paired-block saving was 33.792 us/layer,
or 1.0459x for that composed slice and approximately 1.08 ms across 32 layers
before whole-model integration. v503 still requires
a second row-major MX orientation in addition to the feature-major forward V;
it removes the duplicate *format*, not the duplicate physical orientation.

The matched full 8B integration used v503-A, a fresh v501 control, and v503-B
on the same GB200, checkpoint, token stream, compiler cache, and 3+20 timing
protocol. The v503 A/B mean was 433.539 ms p50 versus 434.025 ms for v501
(0.485 ms or 1.00112x faster), but sustained time was 435.415 versus
435.173 ms (0.242 ms or 0.06% slower). Stage medians explain the result:
v503 saved 1.325 ms in decoder forward and cost 0.744 ms in backward; CE,
optimizer, clipping, and run variance reduced the approximately 0.58 ms
attention-side gain to a 0.485 ms p50 win, while sustained aggregation reversed
it. All updates were finite. This is short-run stability evidence, not
real-data convergence evidence.

### Final direct-common producer ceiling (v506 follow-up)

The final native-D128 direct-common-row producer was tested at
`B=2, S=4096, H=4096, Hq=32, Hkv=8, D=128` on one GB200 with a
self-conditioned, rotated 30-block ABBA/BAAB protocol (60 samples per route).
These are verified measurements: the direct-common candidate measured
237.568 us median versus 206.848 us for the retained per-D32 MX producer. The
raw median regression was 30.720 us and the paired-block median regression was
29.696 us. It therefore failed the predeclared `<=209 us` producer gate. The
payload and repeated common-row scales nevertheless matched the direct-common
oracle byte-for-byte, and all route-neutral Q/K and forward-V publications
were bitwise unchanged. Against BF16 V, the candidate measured cosine
0.993513, relative L2 0.113732, and norm ratio 0.995195; against the retained
per-D32 MX decode it measured cosine 0.999418 and relative L2 0.034184. This is
a performance ceiling result, not a numerical failure.

Separate SASS/resource inspection of the authenticated projection extension
found 255 registers per thread for the direct-common instantiation versus 217
for the retained instantiation, and 10,512 versus 9,624 static instructions
(83 versus 75 static `CALL.REL.NOINC` instructions). The interpretation that
the separate common-row reduction/quantization publisher raises live state and
serial instruction overhead enough to cause the measured regression is an
inference from this static disassembly, not a sampled stall attribution.

The experiment remains compile-time fail-closed and exported only as explicit
experimental C++ symbols; it is not selected by the Python interface or either
end-to-end harness. Because the producer failed the `<=209 us` gate, no
producer-plus-backward composed run of this final direct-common variant was
attempted. It is retained as a negative result, not as a training candidate.

## Authenticated artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Corrected fused projection/stat publisher | 48,223,048 | `1b88a318ffeaea9cee81d80c2167e3bc34820c370390d2350a87a978575ac9fa` |
| D=128 FP8-PV forward | 1,817,144 | `8c3ca848c6524347b99a69cb24b80c86d680b83e84c2e49c6e39be8b16118aba` |
| D=128 MXFP4-PV forward | 1,958,376 | `38d709f0b2b789664dee53dc1a9edec7bb8dc5000dadb8a47989a5b8a4faeb9d` |
| D=128 native TK v501 backward | 6,627,392 | `839c0bc55f809f6ecf6e5e31696c7f89b5a21c26559001dc6d47e80f00a5bf80` |
| Experimental D=128 TK v503 row-scale MX backward | 5,967,472 | `5e2af1d140181c344167ea2bda091970dc8fe265e3d0abd79672dd6bfd93dcf8` |
| Experimental v506 direct-common projection extension | 48,513,448 | `1de85287bd9b217c2a4bee85846a58d764370fb5a5e80cda35f2da1d7c75758a` |
| v506 direct-common projection epilogue source | 161,204 | `749f3425a3b27438609e2f72cece3214b36ce2046665d6ed7c96774ae93edefd` |
| v506 direct-common translation-unit source | 457,221 | `750fa5d50175c91f5f2b23d9926e536eb03b4f65c66d343c90f6a01e1faf63ce` |
| Durable v506 direct-common negative producer receipt | 13,074 | `35be4bb40a2226da1cb284c4267e39dad72a3ddeae7c4662ff3496a4e96224fb` |
| v501 CUDA source in validator receipt | 14,685 | `1083abf27af1f392ecb436750787498942bca4b6200dbd8fe0d909a78ca8caa2` |
| D=64 FP8-PV forward cross-check | 1,817,112 | `16d29867c4c88fdf9c646fd42c203b6315d167344bde9e549a516df9a033218d` |
| D=64 MXFP4-PV forward cross-check | 1,957,920 | `8adb08768f08e337e2f507f0637725d6f1cc1180cf184a3b7d5086ae5f0ef9d2` |
| D=64 native TK v416 backward cross-check | 3,138,360 | `d974cdc3706bc48f1f5452f31c6688f13b510672992da8536e5fe2c824ffdfd5` |

The matched 8B bracket shared initial-state SHA-256
`3a4a7a982ba57f733e0e9e264fdc5ef9c630f8a9d18d73eb9c2b4a5e5b38cb0b`
and BF16 reference-sample SHA-256
`e28222c8c32d2b817ed3207b710796b179189016f3e19a4383d6ae5cf940e209`.
The later v503 A/control/B bracket reused that initial state but used the
fresh `bf16_a_samples.pt` reference artifact with SHA-256
`8f3671deb81691802a8bdbed7ae701499ebd656f02b186b1ebc92dc68e642231`.
The 1B bracket shared initial-state SHA-256
`35f3bf75821c3b744893ebb8a512ea20517b9f39ef20c0fd4620d643ae6c9bbc`
and reference-sample SHA-256
`37ca2927cce4767b3c4f0abfb781251e7adca97861f8e5d6dcf88b576a27e4ef`.
The durable v506 receipt records the raw `/tmp` receipt's 12,726-byte identity,
SHA-256 `4ab0f81b6a2f385c8c525c9c85e6d14a4df1f84d63ecdcd5850b6047c7236623`,
and absolute source path.

## Local ephemeral evidence

The measurements above were read from the following absolute `/tmp` receipts.
They are verified local evidence for this closure, but the paths and binaries
are ephemeral and must not be treated as durable repository artifacts:

```
/tmp/tkfa4_v501_corrected_e2e_8b_b2_20260829_run1/bf16_a.json
/tmp/tkfa4_v501_corrected_e2e_8b_b2_20260829_run1/fp8.json
/tmp/tkfa4_v501_corrected_e2e_8b_b2_20260829_run1/mx.json
/tmp/tkfa4_v501_corrected_e2e_8b_b2_20260829_run1/bf16_b.json
/tmp/tkfa4_v501_corrected_e2e_8b_b2_20260829_run1/isolated_forward_mx_vs_fp8.json
/tmp/tkfa4_v416_corrected_e2e_1b_b16_20260829_run1/bf16_a.json
/tmp/tkfa4_v416_corrected_e2e_1b_b16_20260829_run1/fp8.json
/tmp/tkfa4_v416_corrected_e2e_1b_b16_20260829_run1/mx.json
/tmp/tkfa4_v416_corrected_e2e_1b_b16_20260829_run1/bf16_b.json
/tmp/tkfa4_v501_lstat_lift_fp8_b2_one_layer_20260829/numerical_proof_clear_elided.json
/tmp/d128_native_compare_output_20260829/v501_full_b1_s4096.json
/tmp/d128_native_compare_output_20260829/v501_full_b2_s4096.json
/tmp/tkfa4_d128_mx_v_publication_bound_20260829.json
/tmp/tkfa4_d128_mx_v_numerical_screen_b2s4096_20260829.json
/tmp/tkfa4_d128_rowscale_fp4_v_screen_b2s4096_20260830.json
/tmp/tkfa4_d128_block32_to_rowscale_restage_screen_b2s4096_20260830.json
/tmp/tkfa4_v502_b2_s4096_kmajor_chunk1_fenced_v3_oracle_gate_20260830.json
/tmp/tkfa4_v503_rowscale_exact_oracle_gate_20260830.json
/tmp/tkfa4_v503_rowscale_final_reg136_gate_20260830.json
/tmp/tkfa4_v503_rowscale_final_composed_chain_gate_20260830.json
/tmp/tkfa4_v503_b2_s4096_gate_20260830.json
/tmp/tkfa4_v503_b2_s4096_reg136_gate_20260830.json
/tmp/tkfa4_v503_e2e_8b_b2_20260830_run1/mx_v503.json
/tmp/tkfa4_v503_e2e_8b_b2_20260830_run1/mx_v501_control.json
/tmp/tkfa4_v503_e2e_8b_b2_20260830_run1/mx_v503_repeat.json
/tmp/tkfa4_direct_common_rowscale_producer_gate_20260830.json
/tmp/tkfa4_v501_owner4_nsys_20260830.nsys-rep
/tmp/tkfa4_v501_owner4_gpm_20260830.csv
```

## Closure boundary

Verified: corrected D=128 probability statistics, B1/B2 native-kernel
correctness, B2 clear-safe composition, matched full-model speedups, shared
E4M3 backward for FP8/MX, and a faster isolated MX attention core.

Resolved negatively for the direct block-scaled design: compiled straight-
MXFP4-V consumption is numerically faithful, but its scale/restaging schedule
does not preserve the producer saving. Resolved for the overlap-preserving
row-scale design: the exact oracle passes and it preserves a small
attention-side gain, but the matched full-step bracket is tied within run
variance. Resolved negatively for the final direct-common producer: its
numerics and byte contracts pass, but its 237.568 us median misses the
`<=209 us` gate, so it remains fail-closed and unintegrated. Unresolved:
MX-backward long-run convergence, real-data loss curves, and distributed
scaling. These are experiments, not established properties of the current
route.
