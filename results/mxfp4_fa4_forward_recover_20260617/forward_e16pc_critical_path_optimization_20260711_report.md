# e16pc Critical-Path Profile And Optimization Report

Date: 2026-07-11

Task: `session6_e16pc_critical_path_optimization_20260711.md`

Hardware: NVIDIA GB200, GPU2, SM100a. Performance runs used the persistent MXFP4
forward route and direct output/LSE preallocation. Baselines and candidates used the
matched `KPIPE_STAGE=2` build with timeline, P-chain stamps, scheduler policies, and
counters off. Sparse profiles used separate stamp-enabled builds. No commit or push
was performed.

## Result

No new behavior route is retained. Candidate A, deferred denominator folding, was
implemented as a spill-free explicit sibling and did move producer P-ready earlier.
The issue lane absorbed that movement: producer-ready-to-PV-issue stayed flat or
grew, first/reverse wall time did not repeat, and h16 determinism degraded. The
candidate selector, config, traits, and behavior were removed.

The profile establishes the current e16pc critical path more precisely:

- At steady-state idx2 on h4/s2048 and h8/s4096, e16pc P-ready already precedes
  issue-side P-ready by about `0.88-0.96k` cycles. The issue lane is still completing
  output-rescale/V/descriptor work, so a further producer-only saving becomes slack.
- At h8/s1024, P observation is later and Candidate A shortens it locally, but this
  does not move repeatable kernel wall time.
- Once issue-side P-ready is observed, actual PV TCGEN issue follows in only
  `130-150` cycles. There is no large hidden post-P scheduler delay.
- The previous degree-2 split saving of about 47 cycles was absorbed because it
  advanced an already-early producer on the important steady-state shapes. It did
  not remove the output/V/descriptor dependency that selects the issue time.

Retain committed `e16pc` as the sole explicit/default-off ALU EX2 route. Ordinary
Stage2 remains the global default.

## Clean Baseline

The matched build reproduced selected ptxas and static SASS exactly:

| Route | MUFU.EX2 | packed FFMA2 | F2FP | E2M1 | BRA | Resources |
|---|---:|---:|---:|---:|---:|---|
| Stage2 | 257 | 0 | 144 | 128 | 263 | 168 regs, 2 barriers, 1904 B smem, no stack/spills |
| e16pc | 193 | 96 | 144 | 128 | 263 | same |

First/reverse cells are p50/min/max ms, followed by e16pc p50 delta versus matched
Stage2. Every route was finite.

| Shape | First Stage2 | First e16pc | Reverse Stage2 | Reverse e16pc |
|---|---:|---:|---:|---:|
| h4/s2048 | .065824/.062112/.079968 | .062368/.059744/.075392 (-5.25%) | .066960/.064032/.070464 | .063904/.061536/.069824 (-4.56%) |
| h8/s1024 | .046512/.043456/.056832 | .044880/.042496/.057120 (-3.51%) | .046896/.044288/.056576 | .045568/.043168/.056192 (-2.83%) |
| h8/s4096 | .112448/.106624/.123328 | .107104/.103552/.120928 (-4.75%) | .108336/.104704/.151584 | .103152/.096480/.113728 (-4.79%) |
| h16/s4096 | .179840/.176928/.186656 | .171552/.168768/.176544 (-4.61%) | .184064/.180640/.189280 | .175984/.171872/.188320 (-4.39%) |

One h8/s1024 reverse process entered a visibly slower regime around `.054 ms` for
both routes. GPU2 had no competing process, but that set did not match the adjacent
load regime and was replaced by the listed repeat. The recent `.054176 ms`
h4/s1024 number was a four-sample `KPIPE_STAGE=0` restore smoke, not a Stage2
performance baseline.

## Sparse Profiler

The retained diagnostic changes are compile-gated by
`MXFP4_FWD_PCHAIN_STAMPS` and a new default-zero `PCHAIN_TARGET_IDX` Make variable.
They use 21 named slots: 20 clocks and one owner. Producer lane 128 and issue lane
288 each use `clock64()`. The owner key combines block and persistent task identity,
so a later task in the same CTA cannot overwrite or cross-pair a record. Both lanes
are in the same CTA, which remains resident on one SM; their clocks therefore share
the same SM clock domain.

No synchronization, barriers, polling, phase changes, or ownership changes were
added. Every launch resets the storage. The driver rejects a record if any required
slot is zero, the owner is absent, or a required interval is negative. The
fresh-process collector uses a bounded process timeout. All 132 required Stage2 and
e16pc records (three shapes, two target indices, two routes, 11 each) were valid;
none timed out or was rejected.

When stamps are disabled, the retained e16pc SASS before and after profiler changes
is byte-identical (`sha256 01e94c22a4ee4d43f8414eed7d2d2a48c079637dde75d973b6be4f308e73f35b`).

Each profile cell below is p25/median/p75 cycles over 11 valid fresh-process records.

### Target idx0

| Span | h4/s2048 Stage2 | h4/s2048 e16pc | h8/s1024 Stage2 | h8/s1024 e16pc | h8/s4096 Stage2 | h8/s4096 e16pc |
|---|---:|---:|---:|---:|---:|---:|
| score TMEM load | 268/281/300 | 250/263/284 | 272/288/301 | 266/276/287 | 288/301/306 | 286/295/298 |
| score load to P-stage reuse | 232/239/244 | 230/233/239 | 229/231/237 | 229/230/237 | 226/234/239 | 225/227/232 |
| P-stage reuse to causal mask | 261/276/294 | 257/272/283 | 253/290/301 | 250/262/280 | 228/236/329 | 224/227/234 |
| causal mask to block/row max | 313/316/322 | 318/319/325 | 300/313/320 | 312/323/329 | 368/482/504 | 358/400/422 |
| block/row max to P-scale | 520/522/526 | 518/521/522 | 517/521/524 | 519/522/526 | 624/688/723 | 698/736/767 |
| P-scale store issue to wait | 159/159/160 | 1844/1847/1857 | 159/159/159 | 1841/1843/1852 | 338/375/378 | 2825/2853/3096 |
| P-scale to exp/pack | 1371/1372/1374 | 1454/1457/1462 | 1365/1366/1370 | 1451/1453/1464 | 1841/2016/2054 | 1780/1941/2016 |
| exp/pack to payload stores | 420/420/426 | 247/247/248 | 424/426/426 | 247/248/250 | 1020/1038/1043 | 570/703/750 |
| payload stores to proxy publish | 177/177/177 | 177/180/181 | 177/177/177 | 177/177/178 | 359/365/372 | 278/294/298 |
| publication/scale to producer P-ready | 190/190/190 | 190/190/190 | 190/190/190 | 190/190/190 | 322/324/336 | 293/321/351 |
| producer chain total | 4160/4179/4214 | 3876/3897/3915 | 4135/4176/4222 | 3881/3892/3924 | 6114/6230/6302 | 5286/5399/5483 |
| issue entry to output rescale ready | 78/81/83 | 77/77/80 | 77/79/86 | 78/79/83 | 78/80/88 | 77/77/78 |
| issue entry to V ready | 1354/1388/1397 | 1341/1383/1440 | 964/1325/1350 | 1374/1406/1417 | 1876/1977/2132 | 1899/2005/2114 |
| issue entry to P descriptor | 1438/1467/1478 | 1430/1468/1536 | 1047/1409/1434 | 1468/1498/1512 | 2000/2079/2242 | 1980/2085/2242 |
| P descriptor to P-scale observed | 2963/2977/3012 | 2691/2698/2716 | 3016/3048/3370 | 2664/2672/2736 | 4354/4479/4648 | 3700/3868/4111 |
| P-scale to issue-side P-ready | 78/79/82 | 81/86/90 | 78/80/95 | 82/85/98 | 77/77/81 | 80/82/84 |
| producer P-ready to issue P-ready | 290/292/293 | 310/318/320 | 288/290/326 | 304/323/326 | 300/305/312 | 282/290/294 |
| issue P-ready to PV issue | 131/133/135 | 130/133/134 | 133/142/148 | 133/136/138 | 138/141/143 | 132/133/140 |
| producer P-ready to PV issue | 420/426/430 | 443/450/456 | 424/432/468 | 438/457/462 | 441/444/454 | 422/426/429 |
| output rescale ready to PV issue | 4556/4573/4586 | 4292/4312/4336 | 4572/4577/4598 | 4309/4314/4326 | 6608/6799/6852 | 5933/6045/6249 |
| V ready to PV issue | 3258/3271/3308 | 2998/3010/3026 | 3318/3344/3693 | 2978/2981/3052 | 4676/4827/4958 | 4032/4166/4416 |

At idx0 the issue lane enters early and waits on P. e16pc advances producer P-ready
by 0.28k cycles on the two low shapes and 0.83k at h8/s4096; PV issue follows that
advance. This explains why e16pc itself wins over Stage2.

### Target idx2

| Span | h4/s2048 Stage2 | h4/s2048 e16pc | h8/s1024 Stage2 | h8/s1024 e16pc | h8/s4096 Stage2 | h8/s4096 e16pc |
|---|---:|---:|---:|---:|---:|---:|
| score TMEM load | 265/265/266 | 258/264/270 | 265/265/266 | 256/258/260 | 266/268/272 | 254/257/262 |
| score load to P-stage reuse | 304/304/346 | 320/340/344 | 306/307/309 | 314/340/342 | 351/352/355 | 302/317/348 |
| P-stage reuse to causal mask | 217/219/222 | 232/247/260 | 732/738/739 | 733/742/744 | 222/223/226 | 246/278/300 |
| causal mask to block/row max | 290/290/322 | 296/306/317 | 290/290/290 | 289/289/289 | 321/325/328 | 304/306/310 |
| block/row max to P-scale | 520/521/524 | 519/520/522 | 394/395/397 | 390/390/396 | 520/521/524 | 522/526/531 |
| P-scale store issue to wait | 160/160/160 | 1922/1953/1978 | 160/160/160 | 2222/2229/2243 | 160/160/160 | 1888/2246/2263 |
| P-scale to exp/pack | 1366/1391/1396 | 1466/1500/1528 | 1454/1466/1473 | 1634/1640/1655 | 1365/1368/1368 | 1487/1781/1811 |
| exp/pack to payload stores | 448/498/510 | 297/300/305 | 627/627/627 | 448/448/448 | 438/446/453 | 244/296/306 |
| payload stores to proxy publish | 178/180/181 | 182/184/186 | 177/177/177 | 177/177/177 | 181/181/183 | 184/185/186 |
| publication/scale to producer P-ready | 190/190/190 | 195/201/203 | 190/190/190 | 190/190/190 | 190/190/190 | 198/203/208 |
| producer chain total | 4246/4253/4255 | 4026/4061/4078 | 4826/4838/4847 | 4641/4693/4706 | 4256/4265/4272 | 4049/4370/4390 |
| issue entry to output rescale ready | 3842/3847/3874 | 3997/4026/4064 | 3128/3190/3232 | 3107/3131/3152 | 3794/3850/3864 | 3374/3841/3954 |
| issue entry to V ready | 4238/4261/4268 | 4428/4461/4510 | 3446/3508/3544 | 3424/3448/3469 | 4195/4274/4284 | 3814/4359/4455 |
| issue entry to P descriptor | 4318/4339/4350 | 4506/4544/4592 | 3522/3585/3621 | 3501/3525/3546 | 4273/4351/4370 | 3926/4449/4542 |
| P descriptor to P-scale observed | 208/210/224 | 206/213/234 | 1380/1410/1460 | 1271/1289/1360 | 208/211/214 | 209/234/252 |
| P-scale to issue-side P-ready | 83/85/86 | 79/82/83 | 77/77/77 | 77/77/77 | 78/79/80 | 77/78/87 |
| producer P-ready to issue P-ready | 416/423/442 | 842/877/908 | 280/280/280 | 280/280/280 | 430/448/464 | 933/957/1006 |
| issue P-ready to PV issue | 138/140/142 | 135/139/148 | 130/130/130 | 130/130/130 | 131/135/138 | 132/134/150 |
| producer P-ready to PV issue | 558/562/582 | 990/1008/1044 | 410/410/410 | 410/410/410 | 567/584/599 | 1074/1110/1137 |
| output rescale ready to PV issue | 921/923/930 | 914/972/1006 | 1974/2011/2057 | 1872/1890/1961 | 914/930/946 | 1028/1032/1055 |
| V ready to PV issue | 506/527/532 | 514/518/528 | 1665/1694/1744 | 1555/1573/1644 | 506/510/516 | 510/548/582 |

At steady state, output rescale and V staging dominate the issue-lane prefix. For
h4/s2048 and h8/s4096, the P descriptor then sees scale readiness in only about
`0.21-0.23k` cycles, while producer P-ready has already been live for about
`0.88-0.96k` cycles by issue-side P-ready. Thus the e16pc producer is no longer the
selected critical dependency there. h8/s1024 retains a `1.29k` descriptor-to-scale
wait, but a producer-only shift still did not produce a kernel-level win below.

The long e16pc scale-store issue-to-wait interval is intentional overlap: pchainc
issues the asynchronous store before exp/pack and waits after payload production. It
is not a 2k-cycle blocking store latency.

## Candidate A: Deferred Denominator Fold

The temporary explicit route kept exact degree-3 e16pc exp and hardware E2M1
packing. It reused the four scalar lanes already present in `tile_sum0/tile_sum1`,
stored one weighted qid partial in each lane, published payload and scale readiness,
then performed the original qid0 -> qid3 serial fold and row-sum update after
P-ready. No extra array or balanced numerical fold was used.

The resource and SASS gate passed:

| Route | Registers | Stack/spills | Barriers/smem | MUFU/FFMA2/F2FP/E2M1/BRA |
|---|---:|---:|---:|---:|
| e16pc | 168 | 0/0 | 2 / 1904 B | 193/96/144/128/263 |
| deferred fold | 168 | 0/0 | 2 / 1904 B | 193/96/144/128/263 |

The changed schedule was visible in SASS: `FFMA 178 -> 170`, `FADD 257 -> 260`,
and `FMUL 91 -> 95`. The h4/s1024 smoke was finite; candidate-vs-e16pc output max
abs was `6.103516e-5`.

Matched 30-sample p50/min ms and candidate delta versus e16pc:

| Shape | First e16pc / candidate | Reverse e16pc / candidate |
|---|---:|---:|
| h4/s2048 | .065712/.061920 / .063856/.060352 (-2.82%) | .065216/.061792 / .064720/.062304 (-0.76%) |
| h8/s1024 | .056496/.052096 / .056608/.052896 (+0.20%) | .050016/.048032 / .050320/.047232 (+0.61%) |
| h8/s4096 | .101040/.097952 / .099872/.097568 (-1.16%) | .100560/.097920 / .100720/.097152 (+0.16%) |
| h16/s4096 | .172528/.169248 / .172544/.170208 (+0.01%) | .171904/.167360 / .171360/.168256 (-0.32%) |

The one apparent h4 result failed the required 60-sample confirmation:

| Order | e16pc p50/min | candidate p50/min | Delta |
|---|---:|---:|---:|
| first | .072560/.064576 | .073376/.065600 | +1.12% |
| reverse | .063312/.061536 | .063136/.061152 | -0.28% |

Steady-state sparse medians show the movement and absorption directly:

| Shape | e16pc producer chain | candidate producer chain | e16pc producer-ready -> PV | candidate producer-ready -> PV |
|---|---:|---:|---:|---:|
| h4/s2048 | 4040 | 4005 (-35) | 1061 | 1064 |
| h8/s1024 | 4702 | 4582 (-120) | 410 | 412 |
| h8/s4096 | 4348 | 3981 (-367) | 1130 | 1272 |

The candidate therefore shortened its intended producer work but did not shorten the
post-ready handoff. At h8/s4096, earlier readiness increased measured producer slack
by 142 cycles. The local improvement is real and the wall-time rejection is not an
implementation no-op.

Numerically, candidate-vs-e16pc output max abs was `6.10e-5`, `6.10e-5`,
`1.22e-4`, and `2.075e-3` on h4/s2048, h8/s1024, h8/s4096, and h16/s4096.
At h16, candidate run-to-run output/LSE max abs reached `2.045e-3/1.078e-3`, versus
the retained e16pc envelope. This independently fails the high-head determinism
gate. Candidate A was removed.

## Gated Candidates

- Candidate B was not attempted. Steady-state causal-mask-to-max is about
  `0.29-0.31k` cycles on the representative e16pc paths, much shorter than the
  exposed issue prefix and not dominant after Candidate A.
- Candidate C was rejected from the profile without a cosmetic implementation.
  Publication and scale completion feed the same producer-ready event, V is already
  staged before P observation on the important paths, and actual PV issue is only
  `0.13-0.15k` cycles after issue-side P-ready.
- Candidate D was not attempted. Max-to-scale is about `0.39-0.53k` steady-state
  cycles and is not the selected issue dependency.
- Compact NCU was not run because no candidate passed first/reverse timing,
  determinism, and 60-sample confirmation.

## Final State

Rejected Candidate A code and selector were removed. The useful target-index sparse
profiler, strict record validation, p25/p75 summaries, fresh-process collector, and
off-state timeline/counter reporting remain. `e16pc` and ordinary Stage2 behavior
are unchanged when diagnostics are off.

The forced final build succeeded with:

`MXFP4_FWD_TIMELINE=0 MXFP4_FWD_PCHAIN_STAMPS=0 PCHAIN_TARGET_IDX=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0`

The final ordinary Stage2 h4/s1024 smoke is finite at p50/min
`.052304/.048512 ms`. Timeline is `[]`, P-chain raw is `[[]]`, all interval counts
are zero, and all 64 policy counters are zero. This is a default/off health smoke,
not a matched performance baseline. Scoped `git diff --check` passes.

Primary artifacts:

- `forward_e16pc_critical_path_baseline_{first,reverse}_*_gpu2.json`
- `forward_e16pc_critical_path_baseline_{stage2,e16pc}_kernel.sass`
- `forward_e16pc_critical_path_stamps_idx{0,2}_*_gpu2.json`
- `forward_e16pc_critical_path_denfold_{first,reverse}_*_gpu2.json`
- `forward_e16pc_critical_path_denfold_stamps_idx2_*_gpu2.json`
- `forward_e16pc_critical_path_denfold_confirm60_{first,reverse}_h4_s2048_gpu2.json`
- `forward_e16pc_critical_path_denfold_kernel.sass`
- `forward_e16pc_critical_path_restore_default_off_build.log`
- `forward_e16pc_critical_path_restore_default_off_h4_s1024_gpu2.json`

The shared branch advanced externally during this pass to
`a474ef5be1e848cfe873d6c3d6f433da63071fb2` (`Generalize SM100 BF16 backward
split route`). This pass did not alter or revert that unrelated history. No commit,
push, reset, checkout, or stash was performed by this pass.
