# Next experiments

This queue is ordered to preserve scientific information and avoid spending a
long training budget before the measurement path is trustworthy. It records
what should be learned next; it does not claim that any listed experiment has
run.

The intended sources of truth for route status and evidence boundaries are
`routes.json` and `SCIENTIFIC_STATE.md`. They are part of the continuation
capsule being assembled with this queue. The existing source maps are
[`KERNEL_MAP.md`](KERNEL_MAP.md) and
[`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md); quantitative statements below
also point to committed receipts.

## Evidence already established

- At B1/S4096/Hq32/Hkv8/D128 on GB200, the isolated backward core that
  reconstructs scores from saved quantized Q/K, scales, and log-sum-exp
  has a recorded median latency of 0.356 ms, compared with 0.501 ms for the
  BF16 FA4 backward control. Including the E5M2 dO/stat publisher raises the
  quantized boundary to 0.508 ms. The BF16 control decodes the same represented
  gradient operands, but it does not consume byte-identical saved score input,
  so this is a route-level comparison rather than a schedule-only comparison.
- Complete synthetic 8B updates at S4096 become faster as local batch rises.
  At B4, the recorded adjacent brackets are 854.516 ms versus 751.722 ms for
  BF16 and FP8 P/V, and 857.226 ms versus 751.597 ms for BF16 and MXFP4 P/V.
  FP8 and MXFP4 are tied at this boundary; these fixed-token timings do not
  establish training quality.
- In the matched 64-GPU B4 comparison, BF16 and the
  NVFP4-projection/FP8-P/V route both completed the 100,000,595,968-token
  schedule. At the final same-update validation point, losses are 2.3048148155
  and 2.3948404789. Median throughput over all 874 common post-warmup reports
  is 21,852.6656 and 24,302.9730 tokens/s/GPU, respectively, or 1.1121285x.
  This is one trajectory per arm, not a repeated-run or statistical-equivalence
  result.
- Both learned-projection formats tested with MXFP4 P/V eventually diverge,
  while their FP8-P/V controls remain non-divergent over the observed window.
  In the B4 diagnostic, MXFP4 initially tracks the controls and then shows a
  sharp loss and gradient-norm increase. This makes forward MXFP4 P/V, or the
  state it induces, the common separator; it does not yet distinguish P from
  V or prove a format-level impossibility.
- The retained D128 backward remains one CTA per streaming multiprocessor and
  schedule/dependency limited. Detailed hardware counters for the final binary
  have not been recorded, so the exact contribution of tensor-memory capacity,
  producer latency, and dependency stalls remains unresolved.

These statements trace to:

- `results/fp4_fa4_technical_report_v2_20260819/receipts/causal_d128_report_boundaries_20260901.json`;
- `results/tk_fa4_8b_batch_scaling_20260901/e2e_batch_scaling_summary.json`;
- `results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_completed_20260903.json`
  (SHA256
  `36272a35bd95c3138425e7330403f94d87e40ddd2109cdcb2bcf5e2b21c1c55e`);
  and
- `results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_matched_snapshot_20260902T1358Z.json`
  for the separate MXFP4-P/V failure diagnostic.

## P0: validate the continuation package on clean Blackwell hardware

**Question.** Can a fresh recursive clone reproduce the supported D128 source,
build, and training contracts without a historical binary or private service?

Build B1, B2, and B4 into a new external directory, then run the represented
reference, finite/nontrivial-gradient, exact-zero-dO, liveness, and repeated
timing gates. Render the BF16, FP8-P/V, and diagnostic MXFP4-P/V TorchTitan
configs and run a short distributed B4 smoke test, including checkpoint save
and fresh-process resume.

**Promotion gate.** All artifacts authenticate to the new source tree;
unsupported shapes and ABI mismatches stop; no route silently falls back; the
GPU gates and distributed save/resume pass from the clean clone. Until this
passes, later measurements are development evidence rather than public-release
validation.

## P1: repeat the matched study on fully public data and quantify variation

**Question.** Does the completed single-trajectory result reproduce on an
immutable public token stream, and how much do the validation gap and
throughput vary across independent seeds?

Use the current 8.03B, D128, S4096 recipe: local batch 4, world size 64,
gradient accumulation 4, and effective global batch 1024. Start both routes
from the same initialization. Use one newly published, immutable token stream
and tokenizer manifest with deterministic packing, rank partitioning, and
batch fingerprints. Keep the optimizer, learning-rate schedule, loss,
checkpoint cadence, and validation set identical. Compare only common token
coordinates and retain every throughput observation, including input stalls
and checkpoint windows under a prespecified summary.

Run a short data-and-resume pilot before authorizing another full
100-billion-token budget. The historical SlimPajama ordering is not known
exactly and the raw hosted histories are not public, so this is a new matched
replacement experiment, not a byte-identical rerun of the completed service
trajectories.

**Evidence gate.** Both arms reach the declared token budget, save and
resume successfully, and produce same-coordinate training and held-out
validation curves plus full-run throughput distributions. Independent seeds
must support any uncertainty or statistical-equivalence claim.

## P2: localize the MXFP4-P/V divergence before attempting a rescue

**Question.** Is the failure introduced by MXFP4 probability P, MXFP4 value V,
their interaction, or a later state transition?

Use an identical checkpoint immediately before the observed separation. If no
such checkpoint is available, rerun only to the onset region with a tighter
checkpoint cadence. Evaluate a fixed, fingerprinted minibatch sequence from
the same model and optimizer state. Hold NVFP4 Q/K, the chosen
learned-projection format, the backward that reconstructs scores from saved
quantized Q/K and statistics, loss, and optimizer fixed while testing:

1. FP8 P with FP8 V;
2. MXFP4 P with FP8 V;
3. FP8 P with MXFP4 V; and
4. MXFP4 P with MXFP4 V.

If the mixed formats require new kernels, first validate each operator against
its represented oracle; do not infer a result from a decoded proxy. At every
layer record compact, credential-free summaries of score range, row maximum,
P row sum and zero/saturation fractions, V scale and saturation, attention
output error, dO distribution, dQ/dK/dV error, activation norm, pre-clipping
gradient norm, clipping coefficient, and optimizer-moment norm. Preserve the
first layer and update where the routes separate.

The release MX route already uses the same backward binary as FP8 and publishes
a separate E4M3 V view for backward. Therefore direct-MX backward or stochastic
rounding experiments are not the first discriminator for the recorded
divergence. Test them only after the fixed-backward factorial identifies a
backward-V dependency.

**Decision gate.** A deterministic fixed-input evaluation identifies a minimal
format toggle that precedes the model-state divergence. A proposed repair must
pass its represented-operator gate, the onset-window fixed-input evaluation,
and at least two fresh short trajectories before a long run. Merely remaining
finite is not a convergence result.

## P3: recover the backward-core gain at the composed boundary

**Question.** Can the 0.356 ms backward core remain faster after dO, row
statistics, clearing, binding, and handoff are included?

Profile the complete E5M2 dO/stat publisher plus backward path, not the backward
kernel alone. Attribute time to projection epilogue work, dO conversion,
row-statistic reduction, output clearing, launch/binding overhead, and score
reconstruction plus gradient products. Test changes one at a time and retain
the corrected ABI:

- `lstat = 8 - LSE * log2(e)`;
- the separate gradient epilogue factor of `1/256`;
- E5M2 dO range, which avoids the severe E4M3 zeroing observed in the failing
  fixed-checkpoint evaluation; and
- the exact B1/B2/B4 output-ownership and clearing rules.

Promising work includes fusing or overlapping dO/stat publication with the
output-projection epilogue and removing duplicate host-side binding or
validation from steady state. Do not re-enable the direct CuTe D128 two-CTA
path to gain speed; it remains disabled because it can hang.

**Performance gate.** In an interleaved repeated bracket, publisher plus
backward must beat the matched BF16 backward median, and the improvement must
survive the projection-inclusive attention boundary. Correctness,
exact-zero-dO, and liveness must pass before reporting timing. A faster
isolated kernel with a flat complete boundary is a negative result, not a
speedup.

## P4: profile the retained schedule and test the hardware explanation

**Question.** Is the final route limited primarily by tensor-memory occupancy,
the serial score-to-gradient dependency chain, or producer/handoff bubbles?

Collect counters for the exact retained binary at the paper shape, including
resident CTAs, tensor-core activity, issue stalls, barrier waits, tensor-memory
allocation, shared-memory allocation, and launch gaps. Compare B1, B2, and B4
without transferring counters from a predecessor kernel. Where available,
repeat the same source and clock policy on the recorded Blackwell variants.

**Evidence gate.** The profiler capture, tool version, clocks, source/artifact
hashes, command, and raw counter export are publishable. The analysis must
separate an observed occupancy limit from an inferred architectural cause.
Only then should the paper turn the current tensor-memory explanation into a
general hardware claim.

## P5: decide whether direct MXFP4 V backward is worth further work

**Question.** Can a single MXFP4 V publication remove the dual-publication cost
without giving up gradient quality or complete-update speed?

The existing evidence is unfavorable: exact four-anchor consumption matches
its represented oracle but is slower than the E4M3-V control; common-row
consumption is faster but changes dQ/dK; and its whole-step advantage is within
noise. Keep v503, v506, and v507 diagnostic-only while testing any successor.

**Promotion gate.** One publication feeds both required physical orientations;
the consumer passes represented-oracle and stochastic-rounding checks at the
actual packed format; publisher plus backward beats the E4M3-V route in a
repeated composed bracket; and an onset-window fixed-input training evaluation
shows no new drift. If those conditions cannot be met, retain E4M3 V for
backward and close the line as a documented negative result.

## P6: extend shape coverage only after D128 is closed

The D64 source lineages, schema-v3 B16 builder, and TorchTitan artifact contract
are wired in this repository, but have not passed the fresh-clone GB200 or
DDP16 gates. Different head dimensions change tile geometry, CTA topology, and
tensor-memory reuse, so a D128 schedule should not be generalized by changing
one constant.

For any new shape, define a separate route identity and repeat correctness,
zero-input, liveness, isolated timing, projection-inclusive timing, and
complete-update gates. Do not place D64 on the critical path for the retained
8B result.

## Receipt requirements for every new experiment

Each result must record:

- the parentless public-release commit or private candidate commit, source-tree
  hashes, submodule revisions, and built artifact hashes;
- hardware model, GPU count, clocks/power policy when controlled, driver,
  CUDA, PyTorch, CUTLASS DSL, compiler, and profiler versions;
- exact route, shape, batch, layouts, dtypes, scale semantics, ABI, seed,
  warmup, sample count, timing boundary, and comparator;
- dataset/tokenizer/checkpoint identities and ordered batch fingerprints for
  training evidence;
- all planned observations under the declared inclusion rule, plus failures
  and retries; and
- correctness and liveness gates run before timing.

Do not put credentials, internal service URLs, scheduler objects, or private
storage locations in a receipt. Use stable scientific arm names instead of
job IDs. Update the route catalog and scientific-state handoff when a result
changes a decision.
