#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 benchmark.py \
  --backend "${BACKEND:-both}" \
  --pass "${BENCH_PASS:-forward}" \
  --batches "${BATCHES:-1}" \
  --heads "${HEADS:-16}" \
  --seqlens "${SEQLENS:-2048,4096,8192,16384,32768,65536,131072,262144}" \
  --causal-values "${CAUSAL_VALUES:-0,1}" \
  --warmup "${WARMUP:-5}" \
  --iters "${ITERS:-20}" \
  ${DETERMINISTIC:+--deterministic}
