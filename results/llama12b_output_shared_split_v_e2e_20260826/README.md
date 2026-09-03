# Output-shared direct MXFP4 V publication at saturated Llama-1.2B

The accepted output-shared publisher is a small but real optimization of the
native NVFP4-QK + MXFP4-PV route. It removes a redundant V reload/restaging
step while preserving direct BF16-to-MXFP4 forward V and direct
projection-accumulator E4M3 backward V.

The high-resolution isolated boundary improves from 880.784 us to 839.792 us
at p50, a 1.0488x speedup. Its balanced mean saving is 42.156 us with a
10,000-draw bootstrap interval of [41.626, 42.681] us. All eleven addressable
forward and backward publications are byte-identical to the retained path;
only valid MXFP4 scale bytes are compared because the reserved scale slots are
not part of the consumer ABI.

At the full saturated optimizer-step boundary the effect is necessarily much
smaller: the projection executes once in each of 16 decoder layers, while the
rest of training is unchanged. The mirrored `A1, B1, B2, A2` bracket found a
1.097 ms (0.203%) candidate mean-step saving and 0.639 ms (0.430%) mean
decoder-forward saving. Both adjacent process pairs favored the candidate,
but the conservative two-process interval includes zero. This is evidence of
a directionally consistent small end-to-end improvement, not a claim of a
statistically resolved 0.2% throughput gain.

## Saturated protocol and result

Each fresh process used one exclusive GB200, the same canonical checkpoint,
seed `20260825`, pinned Dolma token stream, B16 x S4096, the full 16-layer
Llama-1.2B preset, three warm-up updates, and 20 measured updates. Each update
contains 65,536 tokens. The final loss used `linear_cross_entropy` from
cut-cross-entropy with `impl="torch_compile"`. Peak HBM was 135.527 GiB
allocated and 147.617 GiB reserved.

| Publication | Step mean / pooled p50 (ms) | Decoder mean / p50 (ms) | Backward mean / p50 (ms) | Sustained tok/s | Useful MFU at step p50 |
|---|---:|---:|---:|---:|---:|
| Retained split-V | 540.475 / 539.314 | 148.661 / 148.358 | 354.861 / 353.976 | 121,042 | 44.394% |
| Output-shared split-V | 539.378 / 539.125 | 148.023 / 147.764 | 354.402 / 354.147 | 121,276 | 44.410% |
| Candidate minus retained | **-1.097 / -0.188** | **-0.639 / -0.594** | -0.459 / +0.170 | +234 | +0.016 percentage points |

The adjacent process-mean step deltas are -0.901 ms for B1 minus A1 and
-1.293 ms for B2 minus A2. Their mean is -1.097 ms; with processes as the
replication unit, the deliberately conservative `df=1` 95% interval is
[-3.591, +1.397] ms. Decoder-forward deltas are -0.939 and -0.338 ms. The
identical serialized backward contract has SHA256
`49003a811eccd2710e41741ba1cfa121887f274021e1551c92eed80ffdd67784`.
The backward mean changes sign between the two adjacent pairs, so this result
does not claim a backward optimization.

## Numerical scope

All records are finite, and every process starts at loss 12.170470. Initial
parameter, hidden-state, and logit samples are bitwise identical across all
four processes. Sampled initial gradients are not bitwise deterministic even
between retained repeats (2.626% relative L2) or candidate repeats (2.858%).
Consequently the later loss trajectories diverge within each route as well as
between routes. The four final heldout losses are 7.95382 and 8.03466 for the
retained repeats and 7.96641 and 7.97158 for the candidate repeats. These 23
optimizer updates are a smoke test, not a convergence comparison.

The appropriate semantic gate is the isolated checked-ABI authentication:
candidate and retained publications have zero byte mismatches for Q/K native
NVFP4 payloads, scale pages, and global scales; represented Q/K backward
E4M3; direct V backward E4M3; and MXFP4 V payload plus valid scales.

## Dispatch and provenance

The retained runs authenticate and then dispatch
`...perblock_qk_split_v_backward_mx_forward_out[_unchecked]`. Candidate runs
authenticate and then dispatch
`...perblock_qk_output_shared_split_v_mx_forward_out[_unchecked]`. All four
receipts name projection/backward binary SHA256
`282dcee1606dedd156e21c4f9973a3bc011636816ac3cf5c72229f3a87bb62ff`,
checkpoint SHA256
`2760f5eb47fd0241317dfd69bd0e2d906909d948d81a5a93f0fd371944f0d2bc`,
and packed-token SHA256
`0e7c735ad8794429330a23dada1a2cd26d3abe955ce4c46d31e40e161c55fd16`.
Every before/after process snapshot contains only the benchmark's own CUDA
process.

`analysis.json` contains the exact pooled and paired statistics, raw receipt
hashes, sample-parity audit, and isolated receipt identities. The projection
artifact is explicitly caller-declared: its bytes are authenticated, but the
binary does not self-attest its relationship to source commit `b4f06fc`.

## Decision

Promote output-shared publication as the automatic choice only for the exact
native-NVFP4, direct-MXFP4, split-V route. Keep an explicit retained fallback,
and fail closed for FP8-PV, E4M3-derived MX, D128, or incompatible scaling
policies. The isolated 4.88% boundary win justifies the implementation; the
measured system benefit should be described as about 0.2%, not as a broad
MXFP4 training speedup.

## Final promoted build

The promotion was subsequently hardened and rebuilt from source commit
`3dc12cfdcd0aa4cf898959f46bdedb5bd7b464ef`. The final B200 extension has
SHA256 `df11bc08cd63038bd4d9a155796addea4b43e1ccfa982f6963251bdc522dd2f7`
and byte count `24879344`. It imports as `_C_b300_lowp_bwd` and exports both
the checked and unchecked output-shared symbols. Candidate-specific resource
inspection found ten compiled projection specializations, all with zero stack
and local bytes; register use spans 242--254. Spill warnings elsewhere in the
full extension belong to unrelated backward kernels.

Selection is deliberately compatibility-preserving and fail-closed. An
omitted Python binder argument remains false and selects the retained
publisher. The CLI/runtime passes explicit `None`, which selects the candidate
only for direct rowwise MX at B16/S4096/H2048/Hq32/Hkv8/D64. Explicit false
selects retained. Explicit true rejects every other route, scaling policy, or
shape. The checked C++ entry independently enforces the authenticated shape,
packed H2048 input width, and rowwise scaling; every Python candidate call
checks both packed input and weight K widths before checked or unchecked
dispatch.

The final exclusive-GPU3 checked authentication forced first use of both the
explicit-`None` candidate and explicit-false retained binders. Both completed
their allocating-reference versus checked-ABI authentication. Their eleven
addressable publications matched bitwise across 315,623,936 bytes; only the
256 valid lanes of each 512-lane MXFP4 scale page were compared. The durable
machine-readable receipt is `promoted_build_auth_3dc12cf.json`.
