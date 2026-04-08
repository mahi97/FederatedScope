#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

to_abs_path() {
  local path=$1
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

prefixed_path() {
  local path=$1
  local directory file
  directory=$(dirname "${path}")
  file=$(basename "${path}")
  printf '%s\n' "${directory}/final_${file}"
}

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gpu_id> <cfg_path> <save_to> [extra cfg opts...]"
  exit 1
fi

GPU_ID=$1
CFG_PATH=$2
SAVE_TO=$3
shift 3

SAVE_TO=$(to_abs_path "${SAVE_TO}")
mkdir -p "$(dirname "${SAVE_TO}")"

echo "[train] cfg: ${CFG_PATH}"
echo "[train] checkpoint: ${SAVE_TO}"
echo "[train] final checkpoint: $(prefixed_path "${SAVE_TO}")"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python federatedscope/main.py \
  --cfg "${CFG_PATH}" \
  device 0 \
  eval_device 0 \
  federate.save_to "${SAVE_TO}" \
  "$@"
