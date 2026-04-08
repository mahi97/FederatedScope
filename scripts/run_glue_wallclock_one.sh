#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <gpu_id> <task> <method> <shared_b|personalized_b> [extra matrix args...]"
  exit 1
fi

GPU_ID=$1
TASK=$2
METHOD=$3
SCENARIO=$4
shift 4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_glue_wallclock_matrix.py" \
  --gpu "${GPU_ID}" \
  --tasks "${TASK}" \
  --methods "${METHOD}" \
  --scenarios "${SCENARIO}" \
  "$@"
