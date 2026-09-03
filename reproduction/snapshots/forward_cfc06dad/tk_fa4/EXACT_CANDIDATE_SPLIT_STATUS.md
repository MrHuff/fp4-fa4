# Exact Candidate Split-Overlap Status

## Scope

This note summarizes the current state of the private exact `candidate` backward path in `tk_fa4`, the files that matter most, the commands used to validate it, and the next problems to solve.

Branch:

- `tk-fa4-sm100-rewrite`

Recent checkpoint commits:

- `5e3b595` - `Checkpoint exact split-overlap candidate path`
- `8592431` - `Optimize exact split-overlap candidate path`

## Current Runtime State

Public routing:

- Public `implementation="hot"` remains frozen on `candidate2`.
- Public routing is controlled from `interface.py`.
- `tk_fa4.cu` is not part of the latest exact split-overlap checkpoint.

Private exact routing:

- Deterministic backward still falls back to the FA4 exact path.
- Noncausal backward still uses the existing exact candidate kernel.
- Causal `2048` private exact `candidate` uses the concurrent split-overlap path:
  - `dq_only` on one CUDA stream
  - `dkdv_only` on another CUDA stream
  - cached events
  - cached helper streams
  - cached split-path scratch tensors

## Current Performance And Correctness

Primary exact gate:

- Compare `candidate` against `ref` using `forward_backend=cute`.

Current exact reads on GPU 1:

- `candidate 2048`: observed in the `~785-834 us` band
- best observed read on the current kept branch: `~784.83 us`
- representative recent reads: `~825.79 us`, `~830.56 us`, `~833.66 us`
- `candidate 4096`: `~2785 us`

Current exactness:

- `2048`
  - `dq_refdiff.max_absdiff ~ 2.42e-5`
  - `dk_refdiff.max_absdiff ~ 3.10e-5`
  - `dv_refdiff.max_absdiff = 0`
- `4096`
  - `dq_refdiff.max_absdiff ~ 5.08e-5`
  - `dk_refdiff.max_absdiff ~ 4.51e-5`
  - `dv_refdiff.max_absdiff = 0`

Conclusion:

- The private exact split-overlap path is exact at roundoff scale against `ref`.
- The current exact `2048` floor is materially better than the earlier `~940-970 us` exact path and better than the early split-overlap `~845-872 us` band.
- The path is still far from the old approximate sub-`200 us` lane; the remaining gap is architectural.

## Active Files

Primary active files:

- `b300_bwd_cute16_candidate.cuh`
  - wrapper routing
  - exact split-overlap orchestration
  - cached CUDA events
  - cached helper streams
  - cached split scratch tensors
  - stream ordering and event waits
- `b300_bwd_cute16_kernel_candidate.cuh`
  - exact candidate kernels
  - `launch_backward_dq_only`
  - `launch_backward_dkdv_only`
  - helper contracts used by the split path
- `direct_bwd_probe.py`
  - direct exactness/performance gate
- `interface.py`
  - public freeze on `candidate2`

Secondary context files:

- `tk_fa4.cu`
  - C++ binding/runtime context
  - not part of the latest exact split-overlap checkpoint
- `b300_bwd_fa4.cuh`
  - trusted exact tile-step math donor
- `b300_bwd_cute.cuh`
  - CuTe-style decomposition donor
- `b300_bwd_cute16_kernel.cuh`
  - hot-family chunk/store helpers and related ownership ideas

## Dormant Or Inactive Files

These are not part of the active exact runtime path and were intentionally not included in the focused runtime checkpoints:

- `b300_bwd_cute16_candidate_cute.cuh`
- `b300_bwd_cute16_candidate_seq2048.cu`
- `b300_bwd_cute16_candidate_seq2048.cuh`

They are useful as historical scaffolds, but they are not the current runtime path.

## Kept Split-Overlap Optimizations

The current kept branch includes these changes:

1. concurrent split exact path for causal `2048`
2. cached thread-local CUDA events
3. cached helper streams
4. cached split-path scratch tensors:
   - `dpsum`
   - `lse_log2`
   - `dqacc`
   - dummy `dqacc` backing for `dkdv_only`
5. `dqacc` zero moved off the caller stream and onto async memset on the `dq` helper stream
6. `dqacc` memset moved before the preprocess wait so it overlaps preprocess
7. `dkdv_only` launched before `dq_only`

## Commands We Actually Use

Build:

```bash
make -B -j1 _C.cpython-312-aarch64-linux-gnu.so
```

Serial exact probe for `2048`:

```bash
/workspace/codebases/fp4_matmul/.venv/bin/python \
  /workspace/codebases/fp4_matmul/tk_fa4/direct_bwd_probe.py \
  candidate 2048 1 cute 1 ref
```

Serial exact probe for `4096`:

```bash
/workspace/codebases/fp4_matmul/.venv/bin/python \
  /workspace/codebases/fp4_matmul/tk_fa4/direct_bwd_probe.py \
  candidate 4096 1 cute 1 ref
```

Useful repetition pattern:

```bash
timeout 70s /workspace/codebases/fp4_matmul/.venv/bin/python \
  /workspace/codebases/fp4_matmul/tk_fa4/direct_bwd_probe.py \
  candidate 2048 1 cute 1 ref
```

Recommended workflow:

1. rebuild `_C`
2. run `2048` serially at least 3 times
3. run `4096` serially at least once
4. only keep a change if exactness remains at roundoff scale and the `2048` band actually improves

## Relevant Inspiration

### `b300_bwd_fa4.cuh`

Use this when exactness is in doubt.

What it gives:

- trusted exact probability reconstruction
- trusted exact masking semantics
- trusted exact tile math

### `b300_bwd_cute.cuh`

Use this for structural inspiration, not as a drop-in exact candidate replacement.

What it gives:

- CuTe-style role split
- clustered ownership
- chunked `dqacc0/1/2`
- higher-level decomposition ideas

What it does not yet give us in private candidate:

- exact `dk/dv` ownership mapping for our hot private use case

### `b300_bwd_cute16_kernel.cuh`

Use this as the closest hot-family donor.

What it gives:

- chunk/store helper patterns
- related ownership/repair ideas
- useful references for `dkdv` ownership and epilogue structure

## What Still Needs Fixing

### 1. Exact `2048` is still far from the old approximate floor

The current exact path is much faster than before, but still nowhere near the old approximate sub-`200 us` branch.

Why:

- the old sub-`200 us` result came from approximate work / removed repairs
- the current path is exact and pays for that exactness

### 2. The remaining gap looks architectural, not micro-tuning sized

The current exact split-overlap path is a practical improvement, but the remaining gap likely needs a deeper decomposition change:

- ownership/work decomposition
- more CuTe-style role separation
- cleaner overlap between `dq_only` and `dkdv_only`
- possibly a better long-term `dqacc` ownership/store contract

### 3. CuTe-style hybrid work is still blocked on `dk/dv` ownership

What we already learned:

- hybrid CuTe `dQ` can be made exact
- exact probability reconstruction alone is not the missing piece
- the stubborn failure is `dk/dv` ownership/mapping, not just masking or zero-init

### 4. `4096` exactness is good, but `2048` remains the main speed target

Current policy:

- `4096` must stay exact and finite
- `2048` is the main optimization target

### 5. There is still unrelated repo dirt

The repo contains unrelated modified and untracked files. Focus future commits tightly on the active runtime files unless a change really belongs in the checkpoint.

## Best Next Steps

If continuing from the current branch, the best next directions are:

1. keep optimizing the exact split-overlap path, not the old monolithic exact kernel
2. reduce the current-stream completion tail after the two helper launches
3. look for more safe overlap between preprocess-adjacent work and the helper streams
4. revisit deeper ownership/work decomposition only when the small overlap wins flatten out
5. only re-open CuTe-style hybrid work if the plan explicitly addresses exact `dk/dv` ownership

## Short Summary

The current private exact `candidate` path is:

- exact against `ref`
- materially faster than earlier exact candidates
- built around concurrent `dq_only` and `dkdv_only` execution
- still limited by a bigger architectural gap to the old approximate sub-`200 us` line

This branch is worth building from, but future gains are increasingly likely to come from better decomposition and ownership, not another random preload tweak.
