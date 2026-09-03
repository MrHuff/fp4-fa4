# HAO-Structured TK NVFP4-QK / FP8-PV on GB200

Date: 2026-08-02

## Scope

This checkpoint covers GB200 only. B300 tuning is deliberately deferred until
the GB200 implementation and benchmark protocol are stable.

The benchmark uses HAO's `create_nvfp4_attention_tensors` factory and times TK,
the checked-out HAO NVFP4-QK/FP8-PV kernel, and HAO BF16 in the same process.
Quantization setup is outside the timed region.

## Retained GB200 schedule

- one persistent 512-thread CTA owns two M128 query stages
- all four score quarters stay in softmax-reader registers
- P is published after two of four quarters
- D128 uses four alternating K/V shared-memory stages
- D64 uses five alternating K/V shared-memory stages
- role register budgets are 192/80/48 for softmax/correction/producer
- six of sixteen packed `exp2` pairs use ALU emulation (`0x3198`)
- `mbarrier.try_wait` uses the 10,000,000-cycle suspend hint
- QK and PV share the fixed 512-column HAO TMEM layout

All tested builds report zero stack, zero spill stores, and zero spill loads.

## Matched results

Times are milliseconds. Speedup is BF16 time divided by TK time.

| B | S | H | D | TK NV/FP8 | HAO NV/FP8 | BF16 | TK speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 24 | 64 | 0.129024 | 0.157856 | 0.129152 | 1.001x |
| 1 | 4096 | 64 | 64 | 0.290944 | 0.358496 | 0.288768 | 0.993x |
| 1 | 32768 | 24 | 64 | 6.652928 | 8.173568 | 6.554416 | 0.985x |
| 1 | 32768 | 64 | 64 | 17.022911 | 21.006336 | 16.852545 | 0.990x |
| 1 | 4096 | 24 | 128 | 0.136160 | 0.159744 | 0.166624 | 1.224x |
| 1 | 4096 | 64 | 128 | 0.304032 | 0.358400 | 0.394304 | 1.297x |
| 1 | 32768 | 24 | 128 | 6.917120 | 8.322272 | 9.182496 | 1.328x |
| 1 | 32768 | 64 | 128 | 18.235424 | 21.319872 | 23.776287 | 1.304x |

D128 output cosine versus BF16 is 0.98972--0.98987 with relative L2
0.14228--0.14327. D64 output cosine is 0.98914--0.98954 with relative L2
0.14462--0.14730. The TK result is close to HAO numerically; at the headline
D128 shape, TK-versus-HAO output cosine is 0.999298.

## D64 diagnosis

At B1/S32768/H24/D64, Nsight Compute reports the same 25% theoretical and
23.4% achieved occupancy for TK and BF16. TK has more eligible warps per
scheduler (0.69 versus 0.58), so the remaining gap is not insufficient waves.

TK executes about 4.164 billion instructions versus 3.451 billion for BF16.
Its dominant waits are correction consumers waiting for softmax/P production.
The extra work is the online `exp2`/FP8 pack path and its synchronization. D64
has half as much QK/PV tensor-core work as D128, so that fixed P-production
cost dominates and leaves D64 near parity. D128 has enough tensor work for the
faster NVFP4 QK path to produce a consistent 1.22--1.33x end-to-end win.

## Rejected D64 variants

All times below use B1/S32768/H24/D64.

| Variant | TK time | Result |
|---|---:|---|
| five KV stages, half-P, 192/80/48 | 6.652928 | retained |
| four KV stages | 6.714368 | slower |
| six KV stages | 6.814432 | slower |
| publish after one quarter | 6.801344 | extra handoff did not cover the tail |
| publish after three quarters | 8.646656 | too-late first PV issue |
| HAO D64 200/64/48 registers | 8.404832 | correction role under-provisioned in TK |
| all-native `exp2` | 7.205904 | SFU saturation |
| denser `0x9999` ALU mask | 7.136992 | excess ALU work |

Packing all barriers into one shared array and caching election predicates also
regressed. Those variants are not retained.

## Source defaults

The accepted defaults are in:

- `tk_fa4/fp4_fa4_fwd/Makefile.hao_direct`
- `tk_fa4/fp4_fa4_fwd/hao_direct_config.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_softmax_reader.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_host.inc`

The matched runner is `tk_fa4/fp4_fa4_fwd/hao_direct_benchmark.py`.
