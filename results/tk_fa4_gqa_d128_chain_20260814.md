# Projection-native causal GQA chain on SM100

Date: 2026-08-14

Device: NVIDIA GB200/B300-class SM100, clean GPU 2 unless noted.

Primary geometry: B=1, S=4096, Hq=32, Hkv=8, D=128, hidden=4096. This is
the attention geometry used by the Llama-3 8B model template. The companion
D64 results represent the Llama-3.2 1B/1.2B-style template.

## Result

The retained chain now publishes the two statistics consumed by CuTe backward
directly from the NVFP4 output-projection epilogue into the first two pages of
the backward workspace. The consumer skips its `sum(O * dO)` preprocessing
launch. The production route also avoids allocating a dead BF16 dO tensor when
only the FP8 backward operand is requested.

The first clean D128 chain measured the following component medians:

| Component | Low precision (us) | BF16 control (us) | Speedup |
|---|---:|---:|---:|
| QKV projection + packed RoPE | 122.528 | 172.128 | 1.405x |
| Causal attention forward | 90.496 | 123.808 | 1.368x |
| dO projection + direct statistics | 77.856 | 113.216 | 1.454x |
| Attention backward, including destination clear | 320.640 | 489.856 | 1.528x |

The BF16 QKV projection deliberately excludes RoPE, so its row is a favorable
BF16 lower bound. The BF16 forward number is the matched SM103 HAO/CuTe DSL
control from `causal_nv_mx_gqa_20260814.md`.

Summing QKV projection, dO projection, and backward gives 521.024 us for the
low-precision route and 775.200 us for the favorable BF16 control, or 1.488x.
Including causal forward gives an indicative 611.520 versus 899.008 us, or
1.470x at the projection/attention boundary.

The producer-fused dQ clear described below removes another 18.240 us from a
paired dO-projection-plus-backward event. Because complete-chain BF16 timing
varied between warm runs, 1.47x remains the conservative stable boundary
headline rather than folding that local gain into a more optimistic ratio.

This is not a complete transformer-block or training-step benchmark. It does
not include output projection forward, projection dgrad/weight gradients,
MLP, optimizer, communication, dynamic allocation, or framework graph
integration. It is the measured boundary that those remaining operations
will dilute in model end-to-end results.

## Producer/consumer contract

For one query row, let

```text
r = sum_j O_j dO_j
l = LSE * log2(e)
```

The projection epilogue emits fixed-scale E4M3 Q, K, V, and dO operands whose
decode scale is 4. Its dPsum is formed from the same BF16-rounded dO fragment,
so it carries the product scale 4 * 4 = 16. The direct workspace pages store

```text
workspace[0] = -16 r
workspace[1] = -l
```

which is exactly the sign convention consumed by the retained CuTe mainloop.
The score scale is `(1 / sqrt(D)) / 16`, and dQ/dK/dV are decoded by 4 at the
publication boundary.

The producer selects negative multiplication constants at compile time. This
encodes the sign in the existing two scale operations and avoids repeating
sign work in every score tile of the consumer. A sign-bit-XOR variant was
correct but made the direct producer about 0.8--2.8 us slower across clean
runs, so it was removed.

## Statistics handoff attribution

A same-process control compared standalone positive-stat publication plus the
best two-page copy/sign path with direct negative workspace publication:

| Path | Median (us) |
|---|---:|
| Standalone positive-stat dO projection | 60.992 |
| Two-page copy and sign | 39.808 |
| Separate handoff total | 100.800 |
| Direct negative-stat projection | 77.856 |

Direct publication therefore saves about 22.944 us on the complete handoff,
even though the negative-stat specialization itself is about 16.864 us longer
than the standalone positive producer. Both specializations compile with the
same register count and no spill delta. The same-process direct-versus-separate
A/B is part of `profile_gqa_d128_chain.py` so later compiler or schedule
changes cannot hide this tradeoff.

## Correctness and quality

The direct statistics are not an approximation. dPsum agrees with the BF16
fragment reference to 3.37e-8 relative L2, and log2-LSE is bit exact. The
workspace aliases the two tensors at the intended byte offsets, and the
remaining dK/dV partial pages are untouched by projection.

On the actual projection-native forward state, the low-precision backward
compared with the matched BF16 backward as follows:

| Output | Relative L2 | Cosine |
|---|---:|---:|
| dQ | 0.209862 | 0.977770 |
| dK | 0.214056 | 0.976879 |
| dV | 0.510732 | 0.862404 |

The individual FP8 Q/K/V/dO publications each have about 0.999645 cosine and
0.02665 relative L2 against their BF16 projection tensors. The much larger dV
error therefore comes from the current approximate forward probability and
backward probability consistency, not from the new exact statistics handoff.
This route remains convergence-gated; timing alone is not sufficient to make
it the default training policy.

## 1.2B versus 8B

| Template | Geometry | Lowp backward (us) | BF16 backward (us) | Speedup |
|---|---|---:|---:|---:|
| Llama-3.2 1B/1.2B-style | S4096, Hq32/Hkv8, D64 | 409.832 | 440.934 | 1.076x |
| Llama-3 8B-style | S4096, Hq32/Hkv8, D128 | 320.640 | 489.856 | 1.528x |

D64 has half the tensor work but retains much of the same scheduling,
reduction, and launch overhead, so it reaches the BF16 floor sooner. D128
exposes enough low-precision MMA work and avoids enough dQ traffic for the
speedup to grow materially. The existing projection epilogue is D128-native;
a true projection-native D64 full-chain measurement still requires a dedicated
D64 QKV epilogue and must not be inferred from the D128 projection numbers.

## Fused dQ destination clear

The persistent dO projection uses one producer warp for payload TMA, one for
scales, and CTA 0's warp 0 for MMA issue. Warp 1 in both CTAs is otherwise
idle. The retained specialization assigns those two idle warps per cluster to
aligned 16-byte stores over the BF16 dQ destination while projection MMA and
the consumer epilogue continue unchanged. Backward can then launch on the same
stream without its standalone clear.

| D128 projection/backward path | Median (us) |
|---|---:|
| Projection, no clear | 80.960 |
| Projection with producer-fused clear | 92.096 |
| Backward kernel, no clear | 313.728 |
| Backward including standalone clear | 326.688 |
| Combined projection + separate clear + backward | 416.032 |
| Combined projection(fused clear) + backward | 397.792 |

The direct combined A/B therefore removes 18.240 us, or 4.59%, with unchanged
dQ/dK/dV. The fused clear adds 11.136 us to projection but removes the clear
launch and its exposed handoff from the combined event. It adds no TMEM,
shared memory, barrier, or work to the 128-register attention kernel.

The cleaned final rebuild independently measured 407.136 versus 384.128 us,
removing 23.008 us, or 5.99%. Across the two clean runs the retained gain is
therefore 18.2--23.0 us (4.6--6.0%), not a single favorable clock sample.

Using CTA 1's otherwise idle warp 0 as a third clear warp saturated this path:
the paired gain fell to 8.224 us, or 2.13%, so the simpler two-warp mapping is
retained. A separate CUDA-stream clear also lost because stream coordination
cost more than the small memory operation.

## Remaining ceiling

The next integration rung is a single captured model-boundary chain with
static output buffers and the hierarchical dQ reduction consumed by projection
backward before global BF16 materialization. More attention barriers are not
indicated: the retained main kernel remains at 128 registers with zero spills.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=2 python \
  tk_fa4/lowp_fa4_bwd/profile_gqa_d128_chain.py \
  --direct-workspace-stats \
  --output /tmp/gqa_d128_chain_direct_stats.json
```

The harness compiles both low-precision and BF16 CuTe backward controls on the
same projected tensors, measures the direct and separate statistics paths in
one process, compares producer-fused and standalone dQ clear events, and
reports all component quality metrics.
