# Owner-CTA dQ quantization for D128 GQA backward

Date: 2026-08-15

This experiment removes the completed BF16 dQ handoff between attention
backward and projection backward. The diagonal attention CTA that owns a
query tile keeps its final local contribution on chip, reads only the single
earlier-contributor lane, completes the reduction, applies inverse RoPE, and
publishes the projection-ready NVFP4 Q operand directly.

The implementation is correct and removes the requested second global
partial-lane read. A second optimization pass overlaps owner quantization with
later-query reduction stores, double-buffers the prior lane, uses approximate
reciprocals in the NVFP4 encoder, and specializes causal trip classification.
On two clean S4096/H32/Hkv8/D128 repeats the owner path is now 16.3% to 16.5%
slower than the materialized chain, down from the original 22.9% gap. It
remains an opt-in structural experiment, not the default path.

## Reduction and quantization

For query tile \(q\), let \(\Delta Q_q^{(k)}\) be the contribution produced
by key-tile CTA \(k\). Causality gives

\[
  dQ_q = \sum_{k=0}^{q}\Delta Q_q^{(k)}.
\]

The diagonal CTA \(k=q\) is the owner. It preserves
\(\Delta Q_q^{(q)}\) in a 128-by-128 BF16 shared tile. All earlier CTAs use
the existing TMA reduction path to form one global BF16 lane

\[
  P_q = \sum_{k=0}^{q-1}\Delta Q_q^{(k)}.
\]

A release counter is incremented when each contributor's two-stage TMA
stores have retired. The owner waits for epoch \(q\), cooperatively stages
\(P_q\) in four coalesced 128-by-32 chunks, and computes

\[
  dQ_q = \operatorname{BF16}(P_q + \Delta Q_q^{(q)}).
\]

For each RoPE pair \((x,y)\), packed BF16 cosine/sine values are decoded and
the inverse rotation is evaluated with FMAs:

\[
  x' = x\cos\theta + y\sin\theta, \qquad
  y' = y\cos\theta - x\sin\theta.
\]

Each 16-value block \(b\) is then encoded as NVFP4. With global decode scale
\(G\), the owner computes

\[
  a_b = \max_{i\in b}|x'_i|, \qquad
  s_b = \operatorname{E4M3}\!\left(\frac{a_b}{6G}\right),
\]

\[
  z_i = \operatorname{E2M1}\!\left(\frac{x'_i}{G s_b}\right),
  \qquad \widehat{x}'_i = Gs_bz_i.
\]

The 256 compute threads pack E2M1 pairs into 32-bit words in shared memory.
One issuer then publishes the 8 KiB payload tile and two 512-byte scale pages
with shared-to-global TMA. K/V are packed afterward by the existing tile
packer, and the assembled Q/K/V NVFP4 operand is consumed directly by
`b300_project_nvfp4`.

## Schedule

- Physical grid-x is the query head and grid-y is the key tile. This gives a
  breadth-first key wavefront and avoids diagonal owners occupying most SMs
  while waiting for long prefix contributors.
- Query tiles are traversed in reverse order inside each CTA.
- Non-owner TMA stores publish readiness as soon as both pipeline stages are
  reacquired, rather than at CTA retirement.
- The owner allocation contains 32 KiB of local BF16 dQ followed by two
  8 KiB prior-lane stages. The packed payload uses 8 KiB and the scale pages
  use 1 KiB. Zero-length, byte-aligned owner fields impose no shared-memory
  cost on BF16/non-owner kernels.
- The diagonal reduction warpgroup releases the compute warpgroups as soon as
  its local owner tile is captured. Later-query reductions continue through
  `sdQ`, while the owner consumes its disjoint alternating prior stages.
- The remaining prior lane is read as coalesced BF16 pairs. A full TMA
  global-to-shared load and half-page streaming both failed to shorten the
  critical path.
- Equal-length, tile-aligned causal runs use the trip bounds directly: all
  fully masked tiles are excluded and only `iter_index == iter_start` invokes
  the exact diagonal element mask.
- The generated hot kernel uses 128 registers, no stack, and no spills.

## Retained result

Protocol: GB200 GPU 0, batch 1, S4096, Hq32/Hkv8, D128, causal, three warmups
and nine samples. Values are medians from the same process and tensors.

| Path | Attention backward (us) | Full dO projection + backward + QKV projection (us) |
|---|---:|---:|
| Materialized BF16 dQ | 502.432 | 678.016 |
| Owner CTA to NVFP4 | 623.328 | 789.760 |

The materialized/owner chain ratio is 0.8585x; equivalently, the owner chain
is 16.5% slower. An independent clean repeat measured 16.3%, with nearly the
same owner-minus-materialized cost (111.7 and 112.9 us). Within the reported
owner chain, the post-attention K/V pack is 29.248 us and the projection over
the prepacked operand is 79.680 us.

| Check | Result |
|---|---:|
| Owner projection versus materialized cosine | 0.9997500 |
| Owner projection relative L2 | 0.0223585 |
| Owner projection norm ratio | 0.9999823 |
| dK relative L2 | 0 |
| dV relative L2 | 0 |
| Finite Q scales | 100% |
| Saturated Q scales | 0 |
| Non-finite projection values | 0 |

A separate three-sample run compiled the BF16 control through the same patched
driver. The low-precision backward including dQ clear was 501.120 us versus
737.472 us for BF16, or 1.472x faster. Its synthetic-gradient accuracy remains
the limiting issue for the overall FP4+FP8 route, independent of owner
publication:

| Low-precision result versus BF16 | Cosine | Relative L2 |
|---|---:|---:|
| dQ | 0.977770 | 0.209862 |
| dK | 0.976879 | 0.214056 |
| dV | 0.862404 | 0.510732 |
| Materialized QKV projection | 0.856269 | 0.521286 |
| Owner QKV projection | 0.856038 | 0.521694 |

The owner-versus-materialized cosine remains 0.999750, so the large BF16 gap
comes from the shared low-precision attention path—especially dV—not from the
owner dQ handoff.

## Optimization ladder

Approximate full-chain medians during development:

| Variant | Time (ms) | Outcome |
|---|---:|---|
| Naive diagonal owner | 13.49 | serialized owner waits |
| Static key-0 owner | 2.283 | correct, poor load balance |
| Reverse-query owner | 1.415 | non-finite |
| Diagonal shared owner | 1.35 | correct |
| Head-major physical grid | 1.06 | correct |
| Early readiness publication | 1.05 | correct |
| Vector pair loads and packed stores | 0.980 | correct |
| TMA payload/scale publication | 0.953 | correct |
| Packed inverse-RoPE PTX | 0.871 | correct |
| Coalesced prior-lane staging | 0.811 | correct baseline |
| Early owner release + alternating prior stages | 0.796 | retained |
| Approximate NVFP4 reciprocals | 0.791 | retained |
| Exact causal trip specialization | **0.782** | retained |

The values above are full-chain development medians and include normal
run-to-run clock variation. The retained causal specialization improved the
owner chain by 9.34 us versus the preceding reciprocal variant while leaving
the projection metric unchanged.

Restoring the prior stages after the packed publication pages was also
rejected. Its paired owner overhead was 110.8 us versus 111.7 to 112.9 us for
the simpler folded allocation; the apparent ratio gain came from a slower
materialized baseline rather than less owner work.

## Forward-style experiments

Backward reconstructs probabilities from the saved LSE,

\[
  P_{ij}=2^{S_{ij}\,\alpha\log_2 e+L_i},
\]

so it has no row-maximum or denominator scan to merge with quantization. That
is the key difference from the forward shiftless/E2M1 path. The retained
period-2 policy applies the packed degree-1 ALU approximation to half of the
probability pairs and native `exp2` to the others.

The following alternatives were measured and rejected:

- Degree-1 ALU `exp2` on every pair raised the materialized chain from about
  670 to 712 us. Native `exp2` on every pair was also slower at about 707 us.
  Score reconstruction is overlapped; extra ALU does not shorten the handoff.
- Reusing the rounded FP8 probability fragment for owner dS raised the final
  owner kernel from 619 to 653 us and worsened projection relative L2 from
  0.02236 to 0.02327.
- Folding each diagonal mask predicate into the unrolled FMA/`exp2` loop raised
  the regular kernel from 489 to 547 us and the owner kernel from 619 to
  663 us. The exact generic element-mask helper is already efficiently
  inlined, so only the outer causal trip specialization is retained.
- A bounded persistent projection consumer waiting on Q/K/V release counters
  deadlocked at 0% GPU utilization on SM100 because the waiting cluster launch
  prevented safe producer admission. The API and counters were removed.

## Interpretation

The experiment reaches the requested dataflow ceiling: the owner contribution
never becomes a second global BF16 lane, no completed BF16 dQ tensor is
written, and projection never rereads such a tensor. The avoided traffic is
not free throughput under the current CTA topology, though. Every one of the
1024 diagonal owners still executes inverse RoPE, block maxima, E4M3 scale
encoding, E2M1 packing, four cooperative staging rounds, and TMA publication.
The second pass hides part of that work, but the standalone materialized pack
still has a more regular, globally coalesced launch.

The result therefore establishes both sides of the ceiling: owner-native
publication is numerically viable and removes the global handoff, but merely
moving the quantizer into the attention owner does not beat the optimized
standalone producer. A future winning version needs projection backward to
consume the owner tile through a cluster/on-chip handoff, or an epilogue whose
quantization work overlaps otherwise idle owner latency.

## Reproduction

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 tk_fa4/lowp_fa4_bwd/owner_nvfp4_cute_smoke.py \
  --sequence 256 --q-heads 4 --kv-heads 1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 tk_fa4/lowp_fa4_bwd/profile_gqa_d128_chain.py \
  --direct-workspace-stats --owner-only --skip-bf16-control \
  --warmups 3 --samples 9 --output owner_s4096.json
```

The CuTe source is assembled from `d64_gqa_cute.patch`,
`d64_gqa_tile_ready.patch`, and `d64_gqa_owner_quantize.patch` by
`tune_d64_gqa_cute.py`.
