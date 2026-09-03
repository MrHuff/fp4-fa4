# Native NVFP4 QKV follow-up for saturated Llama-1.2B (2026-08-26)

This note records the experimental native-NVFP4 QKV projection follow-up to
`results/llama12b_saturated_e2e_20260825/README.md`.  It exists to separate two
questions that were conflated in the original saturated comparison:

1. whether MXFP4-PV attention is slower than FP8-PV attention; and
2. whether the surrounding QKV projection/RoPE/publication path erases the
   MXFP4-PV attention saving.

The answer to the first question is **no** at the measured B16, S4096, D64 GQA
shape.  The original E4M3 projection publisher erased the saving.  A true
native-NVFP4 learned QKV projection restores a forward advantage for MXFP4-PV.
The current evidence does not resolve sub-millisecond whole-step ordering.

## Format contract

Both routes in this follow-up use:

- native E2M1 activation and learned-weight operands for the QKV GEMM;
- row-by-K16 activation scaling and true 16-by-16 learned-weight scaling;
- fused QKV projection, RoPE, causal Q/K publication, and V publication;
- represented row-by-K16 Q/K values in E4M3 containers for backward;
- projection-accumulator E4M3 V for backward; and
- the same low-precision attention-backward and BF16 projection-dgrad graph.

The routes differ only in the forward V publication and attention consumer:
ordinary-order E4M3 V for FP8-PV versus causal-interleaved MXFP4 block-32 V for
MXFP4-PV.  `--qkv-projection-format nvfp4` and
`--experimental-native-nvfp4-projection-out` are both required; the harness
fails closed without this explicit opt-in.

## Uninstrumented saturated bracket

Every run used one GB200, the same model/checkpoint/Dolma data, B16 x S4096
(65,536 tokens/update), three warm-ups, and 20 measured updates.  Values are
arithmetic means of two independent same-route processes.

| Route | Mean step (ms) | Decoder (ms) | Backward (ms) | Sustained tok/s | Final heldout loss | Initial/final logit cosine vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| Native NVFP4 QKV + FP8-PV | 607.0111 | 175.1479 | 394.7948 | 107,737.8 | 8.14638 | 0.74348 / 0.99586 |
| Native NVFP4 QKV + MXFP4-PV | 607.7407 | 174.6016 | 396.3936 | 107,636.7 | 7.97653 | 0.70548 / 0.99761 |

MX is 0.5463 ms faster in decoder forward.  Its measured backward is 1.5988 ms
slower despite an identical graph, making its mean whole step 0.7296 ms slower.
The two mirrored step deltas are -0.4841 and +1.9433 ms; with only two pairs,
the conservative `df=1` 95% interval is [-14.692, +16.151] ms.  Therefore these
runs establish the forward ordering but do **not** establish a whole-step
ordering.

For descriptive context only, relative to the prior-day E4M3-projection
bracket, native QKV reduces mean decoder time by 0.9572 ms for FP8-PV and
1.8764 ms for MXFP4-PV.  This is a cross-day comparison, not a paired timing
claim.

## Matched one-update Nsight attribution

The following are GPU kernel sums from matched one-update profiles captured
after three warm-ups with `--capture-range=cudaProfilerApi`.  Instrumented
latencies are diagnostic and are not the headline throughput measurement.

| Forward range, 16 layers | FP8-PV (ms) | MXFP4-PV (ms) | MX - FP8 (ms) |
|---|---:|---:|---:|
| Native-NVFP4 input preparation | 1.955840 | 1.953600 | -0.002240 |
| Native-NVFP4 QKV-weight preparation | 0.236096 | 0.234528 | -0.001568 |
| QKV projection + RoPE + publication | 13.457920 | 13.972416 | +0.514496 |
| Causal attention | 30.813344 | 29.445408 | -1.367936 |
| Route-specific projection/attention pair | 44.271264 | 43.417824 | -0.853440 |
| Whole decoder kernel sum | 179.262336 | 178.619975 | -0.642361 |

MXFP4-PV attention uses 4.44% less kernel time (a 1.04646x ratio).  Its native
publisher still costs 0.5145 ms/update more because it emits the MX block-scale
payload, but that cost is only 38% of the 1.3679-ms attention saving.  The
combined route-specific boundary is consequently 1.01966x faster for MX.

Each trace contains 2,178 ordered kernel launches.  Exactly 32 signatures
differ: one QKV publisher and one attention kernel in each of 16 layers.  The
remaining 1,390-launch suffix is byte-identical after normalization, with
SHA256 `92125fdfe991347c1e6cff79b3d36bb8c37b21af0f2576175f930741f8668e85`.
The explicitly attributed low-precision backward signatures are also
identical: 13 kernel/configuration rows, 208 launches, SHA256
`7a1bd984131c554954d9e0184bb59733d642ade6836cec6bd16346a67eb1d52a`.
This confirms that the wall-clock backward delta is measurement/system noise,
not route-specific work.

## Numerical scope

All four uninstrumented runs remained finite.  The final heldout repeat spread
was 0.02530 for FP8-PV and 0.04759 for MXFP4-PV.  Against the prior matched BF16
reference, the mean final heldout deltas were +0.09708 and -0.07276,
respectively.  Native QKV has a visibly larger initial perturbation than E4M3
QKV, and 20 updates are much too short to infer convergence.  These results are
an implementation/numerical gate only; they do not support a long-run
pretraining-quality claim.

## Rejected MX publisher rewrite

A follow-up causal-consumer-order MX staging rewrite passed 17 source tests but
failed its static gate before GPU timing.  The exact causal split-V projection
specialization increased from 203 to 252 registers/thread and from 12,976 to
14,056 SASS instructions, with unchanged 13,200-byte shared memory and
48-byte stack use.  It was therefore not integrated or benchmarked.

## Remaining fusion opportunity

RoPE and Q/K/V publication are already inside the QKV projection epilogue.
The remaining large boundary is *before* that GEMM: attention RMSNorm is still
an eager ten-kernel sequence, followed by a separate native-NVFP4 activation
preparation sequence.  In the matched FP8 trace, the 16 attention RMSNorm
sequences sum to 27.949152 ms and the following 16 input-preparation ranges sum
to 1.955840 ms.  These are current costs, not a claim that all 29.905 ms is
recoverable.  A fused attention-RMSNorm + row/K16 NVFP4 producer is the next
implementation target; it must preserve raw input, inverse RMS, and gamma
information for RMSNorm backward while handing the packed operand and scales
directly to the native QKV GEMM.

## Provenance

The native projection extension used by every run is 22,903,528 bytes with
SHA256 `c3b3ba4e1c19d37d1ebc441d0487ca898035bc5cbcbc7422007e8c022df6a3d6`.
The FP8-PV and MXFP4-PV forward extensions retain SHA256
`88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208`
and `cc06fe4337fdc3a7c900f81d68fabc4a8e0c375ea536fbe6405754237a393717`.
The backward control source is SHA256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`.

Uninstrumented result JSON SHA256 values, in execution order, are:

- FP8-PV: `7154d42831236e908e5cd57a0273ff76aa6772aafb5facf1fde7bbe116c0fa38`;
- MXFP4-PV: `6b88407c28612082f7791a5b0e6f8e90826cec4f3fd980282315702a6c2c58c4`;
- MXFP4-PV repeat: `917fa473ae5311e85f54783dc4a25ec6a1d97c92a5fb1946958272004e60f1a5`;
- FP8-PV repeat: `568ef3354e46d761df5dae7d68a00cd7156351381954a1f2f5f75f9b75d1a78e`.

Relevant implementation commits precede this note on branch
`codex/nvfp4-qk-reexplore-20260826`: native ABI port, represented publication,
saturated-harness opt-in, and corrected native template wiring.
