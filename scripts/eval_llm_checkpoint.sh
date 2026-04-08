#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_PARENT=$(cd "${REPO_ROOT}/.." && pwd)

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

resolve_checkpoint_path() {
  local requested=$1
  local path base directory candidate
  local -a candidates=()

  if [[ "${requested}" = /* ]]; then
    candidates+=("${requested}")
    candidates+=("$(prefixed_path "${requested}")")
  else
    for path in "${PWD}" "${REPO_ROOT}" "${REPO_PARENT}"; do
      candidates+=("${path}/${requested}")
      candidates+=("$(prefixed_path "${path}/${requested}")")
    done
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' "$(to_abs_path "${requested}")"
}

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <gpu_id> <gsm8k|mmlu> <cfg_path> <save_to> [extra cfg opts...]"
  exit 1
fi

GPU_ID=$1
TASK=$2
CFG_PATH=$3
SAVE_TO=$4
shift 4

SAVE_TO=$(resolve_checkpoint_path "${SAVE_TO}")

echo "[eval] task: ${TASK}"
echo "[eval] cfg: ${CFG_PATH}"
echo "[eval] checkpoint: ${SAVE_TO}"

case "${TASK}" in
  gsm8k)
    SCRIPT_PATH="federatedscope/llm/eval/eval_for_gsm8k/eval.py"
    DEFAULT_OPTS=(
      llm.gsm8k.eval_max_samples -1
      llm.gsm8k.test_size -1
    )
    ;;
  mmlu)
    SCRIPT_PATH="federatedscope/llm/eval/eval_for_mmlu/eval.py"
    DEFAULT_OPTS=(
      llm.tok_len 1024
      eval.llm.compile_model False
    )
    ;;
  *)
    echo "Unknown task: ${TASK}"
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT_PATH}" \
  --cfg "${CFG_PATH}" \
  device 0 \
  eval_device 0 \
  federate.save_to "${SAVE_TO}" \
  "${DEFAULT_OPTS[@]}" \
  "$@"
