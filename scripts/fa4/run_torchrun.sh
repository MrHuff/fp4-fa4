#!/usr/bin/env bash
set -Eeuo pipefail

require_nonempty() {
  local name=$1
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
}

require_positive_integer() {
  local name=$1
  require_nonempty "${name}"
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name=$1
  require_nonempty "${name}"
  if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer" >&2
    exit 2
  fi
}

require_positive_integer NNODES
require_positive_integer NPROC_PER_NODE
require_nonnegative_integer NODE_RANK
require_nonempty CONFIG_FILE
world_size=$((NNODES * NPROC_PER_NODE))

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "CONFIG_FILE is not a regular file: ${CONFIG_FILE}" >&2
  exit 2
fi

config_receipt=${CONFIG_RECEIPT:-${CONFIG_FILE}.receipt.json}
python_bin=${PYTHON_BIN:-python3}
python_bin=$(command -v "${python_bin}" || true)
if [ -z "${python_bin}" ] || [ ! -x "${python_bin}" ]; then
  echo "PYTHON_BIN must name an executable Python interpreter" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/../.." && pwd)
"${python_bin}" "${repository_root}/tools/verify_fa4_training_config.py" \
  --config "${CONFIG_FILE}" \
  --receipt "${config_receipt}" \
  --world-size "${world_size}"

if [ "${NNODES}" -gt 1 ]; then
  require_nonempty RDZV_ENDPOINT
fi

torchrun_bin=${TORCHRUN_BIN:-}
if [ -z "${torchrun_bin}" ]; then
  torchrun_bin=$(command -v torchrun || true)
fi
if [ -z "${torchrun_bin}" ] || [ ! -x "${torchrun_bin}" ]; then
  echo "TORCHRUN_BIN must name an executable torchrun" >&2
  exit 2
fi

rdzv_id=${RDZV_ID:-fa4-public-run}
export NNODES
export LOCAL_WORLD_SIZE=${NPROC_PER_NODE}

torchrun_args=(
  "--nnodes=${NNODES}"
  "--nproc-per-node=${NPROC_PER_NODE}"
  "--node-rank=${NODE_RANK}"
)
if [ "${NNODES}" -eq 1 ] && [ -z "${RDZV_ENDPOINT:-}" ]; then
  torchrun_args+=("--standalone")
else
  torchrun_args+=(
    "--rdzv-backend=c10d"
    "--rdzv-endpoint=${RDZV_ENDPOINT}"
  )
fi

preflight_args=("${torchrun_args[@]}")
training_args=("${torchrun_args[@]}")
if [ "${NNODES}" -gt 1 ] || [ -n "${RDZV_ENDPOINT:-}" ]; then
  preflight_args+=("--rdzv-id=${rdzv_id}-preflight")
  training_args+=("--rdzv-id=${rdzv_id}")
fi

echo "FA4_LAUNCH_TOPOLOGY nodes=${NNODES} local_world=${NPROC_PER_NODE} world=${world_size} node_rank=${NODE_RANK}"

if [ "${FA4_RUN_NCCL_PREFLIGHT:-1}" = "1" ]; then
  "${torchrun_bin}" "${preflight_args[@]}" -m tools.fa4_nccl_preflight
elif [ "${FA4_RUN_NCCL_PREFLIGHT:-1}" != "0" ]; then
  echo "FA4_RUN_NCCL_PREFLIGHT must be 0 or 1" >&2
  exit 2
fi

exec "${torchrun_bin}" "${training_args[@]}" \
  -m torchtitan.experiments.fa4.train --job.config-file "${CONFIG_FILE}" "$@"
