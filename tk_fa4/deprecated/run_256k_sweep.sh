#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark.py"
PYTHON_BIN="${TK_FA4_SWEEP_PYTHON:-/workspace/codebases/fp4_matmul/.venv/bin/python}"
RUNPY_BOOTSTRAP="import runpy, sys; path=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(path, run_name='__main__')"

DEFAULT_PASSES="forward backward full"
DEFAULT_SEQLENS="512,1024,2048,4096,8192,16384,32768,65536,131072,262144"
DEFAULT_CAUSAL_VALUES="0,1"
DEFAULT_BATCHES="1,4"
DEFAULT_HEAD_PAIRS="32x32,32x8,32x1"
DEFAULT_HEAD_DIMS="64,128"

PASSES_RAW="${PASSES:-$DEFAULT_PASSES}"
SEQLENS="${SEQLENS:-$DEFAULT_SEQLENS}"
CAUSAL_VALUES="${CAUSAL_VALUES:-$DEFAULT_CAUSAL_VALUES}"
BATCHES="${BATCHES:-$DEFAULT_BATCHES}"
HEAD_PAIRS="${HEAD_PAIRS:-$DEFAULT_HEAD_PAIRS}"
HEAD_DIMS="${HEAD_DIMS:-$DEFAULT_HEAD_DIMS}"
BACKEND="${BACKEND:-both}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
BUILD_TK="${BUILD_TK:-0}"
GPU_TARGET="${GPU_TARGET:-B200}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/$RUN_ID}"

IFS=' ' read -r -a PASSES <<< "$PASSES_RAW"

mkdir -p "$OUT_DIR"
INDEX_FILE="$OUT_DIR/index.tsv"

printf 'pass\tstatus\tlog_path\n' > "$INDEX_FILE"

build_tk() {
    echo "Rebuilding tk_fa4 for ${GPU_TARGET}"
    PATH="$(dirname "$PYTHON_BIN"):$PATH" make -C "$SCRIPT_DIR" clean "GPU=${GPU_TARGET}"
    PATH="$(dirname "$PYTHON_BIN"):$PATH" make -C "$SCRIPT_DIR" "GPU=${GPU_TARGET}"
}

run_pass() {
    local pass_name="$1"
    local log_file="$OUT_DIR/${pass_name}.log"
    local status=0
    local cmd=(
        "$PYTHON_BIN"
        -c "$RUNPY_BOOTSTRAP"
        "$BENCHMARK_SCRIPT"
        --backend "$BACKEND"
        --pass "$pass_name"
        --batches "$BATCHES"
        --seqlens "$SEQLENS"
        --head-pairs "$HEAD_PAIRS"
        --head-dims "$HEAD_DIMS"
        --causal-values "$CAUSAL_VALUES"
        --warmup "$WARMUP"
        --iters "$ITERS"
    )
    if [[ "$FAIL_FAST" == "1" ]]; then
        cmd+=(--fail-fast)
    fi

    echo
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pass=${pass_name}"
    echo "log=${log_file}"

    set +e
    {
        printf '+'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        "${cmd[@]}"
    } 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "$status" -eq 0 ]]; then
        printf '%s\t%s\t%s\n' "$pass_name" "ok" "$log_file" >> "$INDEX_FILE"
    else
        printf '%s\t%s\t%s\n' "$pass_name" "failed(${status})" "$log_file" >> "$INDEX_FILE"
        if [[ "$STOP_ON_ERROR" == "1" ]]; then
            echo "Stopping after failure in pass=${pass_name}" >&2
            exit "$status"
        fi
    fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Benchmark Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi

if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
    echo "Benchmark script not found: $BENCHMARK_SCRIPT" >&2
    exit 2
fi

if [[ "$BUILD_TK" == "1" ]]; then
    build_tk
fi

echo "Output directory: $OUT_DIR"
echo "Python:           $PYTHON_BIN"
echo "Backend:          $BACKEND"
echo "Passes:           ${PASSES[*]}"
echo "Batches:          $BATCHES"
echo "Seq lens:         $SEQLENS"
echo "Head pairs:       $HEAD_PAIRS"
echo "Head dims:        $HEAD_DIMS"
echo "Causal values:    $CAUSAL_VALUES"
echo "Warmup / iters:   $WARMUP / $ITERS"
echo "Fail fast:        $FAIL_FAST"

for pass_name in "${PASSES[@]}"; do
    run_pass "$pass_name"
done

echo
echo "Sweep complete."
echo "Index: $INDEX_FILE"
