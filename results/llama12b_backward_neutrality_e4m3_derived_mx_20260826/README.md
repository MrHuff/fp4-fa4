# Aggregate backward neutrality and E4M3-derived MXFP4 V

This directory records two results at the saturated Llama-1.2B causal shape
`B16/S4096/H2048/Hq32/Hkv8/D64` on one NVIDIA GB200 (SM100):

1. Direct-MXFP4-PV and E4M3-PV now save and execute one canonical physical
   low-precision backward implementation. A fresh 64-superblock run passes the
   predeclared `+/-1%` timing-equivalence gate.
2. Deriving forward MXFP4 V from the retained backward E4M3 publication is
   exact, but is slower than the existing direct MXFP4 publisher.

The retained production choice is therefore still the existing direct MXFP4
publisher. The E4M3-derived route remains experimental and default-off. The
default direct-MX and FP8 dispatchers were not changed by that experiment.

## Format boundaries

Both routes use native NVFP4 for the QKV projection and native NVFP4 Q/K
publications for causal FA4 forward. They differ at forward `P x V`:

- direct MX uses causal-interleaved MXFP4 V with E8M0 block-32 scales;
- FP8 uses E4M3 V; and
- experimental derived MX uses `MX(E4M3(4V) / 4)`.

Training also retains projection-accumulator E4M3 V for backward. Represented
NVFP4 Q/K values are carried in E4M3-typed containers for the shared backward;
that container type does not make the forward Q/K path an FP8 path.

Direct and derived MX are intentionally different quantizers. Direct MX is
formed from the BF16 projection accumulator. Derived MX is formed after the
E4M3 representation boundary. Derived-route correctness therefore means exact
agreement with `MX(E4M3(4V) / 4)`, not byte equality with `MX(BF16 V)`.

## Verified aggregate-backward implementation

The harness allocates one 16-layer model, one fixed token/target batch, and one
immutable parameter state. MX and FP8 forward routes are crossed over, while
autograd saves one canonical backward owner.

Before and after timing, both routes shared the same runner, compiled callable,
kernel, control module, RoPE objects and storage, gradient scale, dQ/dK/dV
storage, partials, and workspace allocation. Their complete serialized
backward contracts also matched. Relevant source, loaded-module, and binary
identities remained stable.

No optimizer was constructed and no parameter update occurred. Route-natural
forward outputs, and therefore dynamic backward values, were allowed to differ.
This is a physical-implementation and timing-equivalence diagnostic, not a
training-convergence experiment.

### Definitive 64-superblock result

The protocol used 64 eight-call ABBA/BAAB superblocks, 256 timed backward
samples per route, 20,000 clustered bootstrap draws, and seed `20260826`.

| Route | Backward mean (ms) | Backward p50 (ms) | Samples |
|---|---:|---:|---:|
| E4M3-PV | 374.154 | 369.749 | 256 |
| Direct MXFP4-PV | 373.245 | 370.041 | 256 |

The clustered symmetric-relative MX-minus-FP8 point effect was `-0.240%`.
Its 95% interval was `[-0.628%, +0.138%]`, wholly inside the predeclared
`[-1%, +1%]` equivalence region. The corresponding absolute interval was
`[-2.378, +0.536] ms`. The neutrality gate passes without changing the margin.

The fixed loss was invariant within each route:

- E4M3-PV: `12.172077178955078`, zero spread;
- direct MXFP4-PV: `12.17305850982666`, zero spread.

The values need not match across routes because their forward V formats differ.
Peak reserved HBM was 131.94 GiB, below the 180-GiB gate. All 64 periodic GPU
process checks found no foreign PID. Every physical-identity, fixed-loss,
memory, source/module/binary stability, and exclusivity gate passed.

### Retained 24-superblock diagnostic

The first hardened run used 24 superblocks and 96 samples per route:

| Route | Backward mean (ms) | Backward p50 (ms) | Samples |
|---|---:|---:|---:|
| E4M3-PV | 374.194 | 368.993 | 96 |
| Direct MXFP4-PV | 372.508 | 369.151 | 96 |

Its MX-minus-FP8 point effect was `-0.451%`, with 95% interval
`[-1.104%, +0.218%]`. It missed the lower equivalence boundary by 0.104
percentage points and is formally inconclusive. It did not show an MX
backward slowdown; it could not rule out an MX advantage slightly larger than
1%. This receipt is retained rather than silently replaced by the
higher-powered follow-up.

All auxiliary gates passed in the 24-block run as well, including 24 clean
periodic process checks and the same 131.94-GiB peak reserved HBM.

## Verified E4M3-to-MX publication result

This directly tests the proposed `FP8 cast -> MXFP4` route in both standalone
and fused-inline forms.

### Standalone conversion: clear no-go

The standalone converter consumes contiguous E4M3(x4) V in `[B,H,64,S]` and
publishes causal packed E2M1 payload plus raw E8M0 scales.

At the authenticated shape:

| Quantity | Time (us) |
|---|---:|
| Converter p50 | 77.792 |
| Converter mean | 77.901 |
| Existing direct-MX projection premium | 37.626 |
| Exact FP8 projection + converter | 926.719 |
| Existing direct-MX projection | 886.553 |

The converter consumes `2.068x` the available premium and misses break-even by
`40.166 us`. A separate global FP8-to-MX conversion is therefore not viable at
this shape.

The deterministic standalone probe passes byte-exactly across 4,096 payload
bytes and 256 scale bytes. The non-finite policy also passes: an affected group
receives E8M0 `0xff` and zero payload while finite neighboring groups remain
nonzero. Nineteen GPU-exclusivity checks found no foreign PID.

### Inline staged-E4 derivation: exact but slower

The inline route removes the extra global E4 read/write round trip. It stages
the exact backward E4 words inside the projection epilogue, performs the
cross-warp causal gather, and derives forward MX from those bytes.

| Publisher | Mean (us) | p50 (us) |
|---|---:|---:|
| Existing direct MX | 886.600 | 886.688 |
| Inline staged-E4-derived MX | 889.000 | 890.352 |

Across 200 complementary-order timing blocks, derived minus direct was
`+2.400 us` at the mean, with bootstrap 95% interval
`[+2.128, +2.680] us`. The derived route is approximately 0.27% slower and is
rejected as a speed optimization.

All checked correctness gates pass:

- direct and derived Q/K payloads, scales, globals, and backward E4M3 Q/K/V
  have zero byte mismatches;
- derived and standalone output have zero mismatches across 16,777,216 payload
  bytes and 1,048,576 valid scale bytes; and
- deliberate checked-ABI input/output aliasing is rejected before launch.

All 202 GPU-exclusivity checks found no foreign PID. Source, loaded-interface,
and binary identities remained stable.

## Interpretation

The negative derived-MX result is a publication-layout cost, not evidence that
MXFP4 is fundamentally unsuitable for training.

Backward E4 pairing and forward causal MX grouping require different
sequence/depth access patterns. Exact reuse therefore needs a cross-warp gather
and synchronization. The direct publisher already has BF16 accumulator pairs
resident and can form forward MX while separately publishing backward E4.

This explains the measured inference/training asymmetry: inference only needs
the smaller forward MX publication, while training must additionally retain a
backward-friendly representation. It does not establish a general convergence
limit for MXFP4, and it does not show that all future fused publication layouts
must pay the same cost.

## Limitations

These receipts cover one model shape, one sequence length, one batch size, and
one GB200 GPU. The aggregate harness uses fixed parameters and no optimizer
updates. It does not measure multi-GPU scaling, long-run loss, BF16-relative
end-to-end speedup, tokens/s, or MFU.

The 64-block receipt contains one 330.403-ms FP8 decoder-forward outlier. Its
forward and combined timings are therefore not used for an end-to-end claim;
only the predeclared clustered backward-equivalence statistic is interpreted.

The 24-superblock result is retained because its formal equivalence gate
failed. The fresh 64-superblock receipt is the higher-powered follow-up and
supports the final neutrality claim under the unchanged `+/-1%` criterion. It
reuses the fixed seed and batch, so it is not claimed as a statistically
independent replication.

## Implementation provenance

- `1192c6ec09dcd83185b6aaf9d0b969d40e0d9938` shares one canonical aggregate
  backward across eligible PV routes.
- `7cb9caccac2b3cfa5e631ceba62599f5d2c7460e` adds authenticated E4M3-derived
  MX publication experiments.
- `96bc842c39b36302c10f6efbd91f1d1188937f6b` hardens benchmark provenance,
  process isolation, receipt creation, and loaded-module authentication.

The final serially built extension is 24,061,504 bytes with SHA256
`08734ab13795d0182089ed523b3779375deda63a89473e9db10ce7360de3a576`.
A final focused regression shard passes 318 tests.

## Retained receipts

| File | Result | Bytes | SHA256 |
|---|---|---:|---|
| `backward_neutrality_24block_inconclusive.json` | Formal gate inconclusive | 597,019 | `85854a3967f14bdf659ff8e0fadd412f76f41aa6225f08622b1a79896f6d2425` |
| `backward_neutrality_64block.json` | `+/-1%` equivalence passes | 766,634 | `a01ae63b6c195c3cf0f0221b22278cbf26ad0c18e255a75f9df5a7848b6dc66d` |
| `inline_e4m3_derived_mx.json` | Correct, slower than direct MX | 208,519 | `44a6cf61ee5dbf9c9dc72dd2a2746611eefd6f85ca78d3c497eb9e6a0354c681` |
| `standalone_e4m3_to_mxfp4_v.json` | Correct, exceeds break-even | 13,599 | `9d8a753cf2f609bcd958feacf94469ed21523f092897093ec5ed81cb7189bfa6` |

Each harness uses create-only output, authenticates the relevant sources and
extension across timing, and checks GPU process exclusivity outside measured
event intervals. The standalone converter intentionally exits with status 2
under `--require-neutral` after writing its correct no-go receipt.
