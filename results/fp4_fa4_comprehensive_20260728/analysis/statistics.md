# Aggregate statistics

| Provider | Shapes | Geomean vs BF16 | Wins vs BF16 | Best speedup | Peak TFLOP/s |
|---|---:|---:|---:|---:|---:|
| TK NVFP4/NVFP4 | 23 | 2.003x | 23 | 5.239x | 2634.5 |
| HAO NVFP4/NVFP4 | 23 | 0.797x | 1 | 1.069x | 1400.7 |
| TK NVFP4/FP8 | 23 | 1.441x | 21 | 4.368x | 1799.8 |
| HAO NVFP4/FP8 | 23 | 0.915x | 7 | 1.052x | 1649.9 |

- TK FP4 vs HAO FP4: 2.514x geomean, 23/23 wins.
- TK FP8 vs HAO FP8: 1.576x geomean, 23/23 wins.
- TK FP4 cosine: median 0.962424, minimum 0.952886; relative L2: median 0.277685 (27.77%), maximum 0.317031 (31.70%).
- TK FP8 cosine: median 0.956781, minimum 0.947054; relative L2: median 0.292965 (29.30%), maximum 0.328734 (32.87%).

## Steady-state subset

D128 rows with sequence length at least 4096:
- TK NVFP4/NVFP4: 1.553x geomean vs BF16, 15/15 wins.
- HAO NVFP4/NVFP4: 0.820x geomean vs BF16, 0/15 wins.
- TK NVFP4/FP8: 1.064x geomean vs BF16, 13/15 wins.
- HAO NVFP4/FP8: 0.985x geomean vs BF16, 7/15 wins.
