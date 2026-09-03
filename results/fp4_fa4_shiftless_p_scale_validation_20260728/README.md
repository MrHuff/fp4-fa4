# Shiftless probability-scale validation

This directory validates the only locally promising fast-path factor,
`G=1.5`, against the production default `G=1`. All rows use the same
shiftless mode-9 policy and BF16 reference as the primary sweep.

## Cross-shape result

| Shape / route | G=1 time | G=1.5 time | G=1 rel. L2 | G=1.5 rel. L2 | Error change |
|---|---:|---:|---:|---:|---:|
| B1 S1024 H16 NV/NV | 0.016384 | 0.017312 | 0.284186 | 0.284280 | +0.000093 |
| B1 S8192 H24 NV/NV | 0.380672 | 0.378800 | 0.275582 | 0.275596 | +0.000014 |
| B1 S32768 H24 NV/NV | 5.144288 | 5.106720 | 0.274687 | 0.274695 | +0.000008 |
| B4 S4096 H32 NV/NV | 0.454656 | 0.442720 | 0.278851 | 0.278856 | +0.000005 |
| B1 S4096 H24 MX/NV | 0.104736 | 0.104448 | 0.298980 | 0.298996 | +0.000016 |

Positive error change is worse. `G=1.5` loses on every broad-shape check and
on the MXFP4-QK control. Timing deltas are not treated as wins because they
are comparable to run-order and clock variation.

## Seed stability

Across seeds 0 through 4 at B1/S4096/H24 NV/NV:

| G | Mean cosine | Mean relative L2 | Mean RMSE |
|---:|---:|---:|---:|
| 1 | 0.962183178 | 0.278307349 | 0.007184514 |
| 1.5 | 0.962185717 | 0.278292179 | 0.007184122 |

The mean relative-L2 improvement is `0.0000152`. Individual-seed
`G=1.5 - G=1` changes are `-0.0000172`, `+0.0000047`, `-0.0000435`,
`+0.0000053`, and `-0.0000252`; two of five seeds regress. This is too small
and inconsistent to justify a global default change.

The tracked [`summary.json`](summary.json) records each row, the two
same-process timing orders, and the SASS instruction comparison. Raw cases
and build logs are intentionally untracked.
