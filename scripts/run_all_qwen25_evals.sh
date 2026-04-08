#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval_llm_checkpoint.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gpu_id> [gpu_id ...]"
  echo "Example: $0 0 1 2 3"
  exit 1
fi

GPU_IDS=("$@")
NUM_GPUS=${#GPU_IDS[@]}

CONFIGS=(
  "qwen25_3b_factor_fedavg.yaml|rebuttal_qwen25_3b_factor_fedavg.ckpt"
  "qwen25_3b_factor_fedprox.yaml|rebuttal_qwen25_3b_factor_fedprox.ckpt"
  "qwen25_3b_fa_lora.yaml|rebuttal_qwen25_3b_fa_lora.ckpt"
  "qwen25_3b_svd.yaml|rebuttal_qwen25_3b_svd.ckpt"
  "qwen25_3b_svd_no_gram.yaml|rebuttal_qwen25_3b_svd_no_gram.ckpt"
  "qwen25_3b_fedma.yaml|rebuttal_qwen25_3b_fedma.ckpt"
  "qwen25_3b_full_rank.yaml|rebuttal_qwen25_3b_full_rank.ckpt"
  "qwen25_3b_factor_fedavg.yaml|rebuttal_qwen25_3b_factor_fedavg_pb.ckpt"
  "qwen25_3b_factor_fedprox.yaml|rebuttal_qwen25_3b_factor_fedprox_pb.ckpt"
  "qwen25_3b_svd.yaml|rebuttal_qwen25_3b_svd_pb.ckpt"
  "qwen25_3b_svd_no_gram.yaml|rebuttal_qwen25_3b_svd_no_gram_pb.ckpt"
  "qwen25_3b_fedma.yaml|rebuttal_qwen25_3b_fedma_pb.ckpt"
)

run_pair() {
  local gpu_id=$1
  local cfg_name=$2
  local ckpt_name=$3
  local cfg_path="${REPO_ROOT}/federatedscope/llm/yamls/rebuttal/${cfg_name}"

  echo "[start][gpu ${gpu_id}] ${ckpt_name}"
  (
    "${EVAL_SCRIPT}" "${gpu_id}" gsm8k "${cfg_path}" "${ckpt_name}" && \
    "${EVAL_SCRIPT}" "${gpu_id}" mmlu "${cfg_path}" "${ckpt_name}"
  ) &
}

for idx in "${!CONFIGS[@]}"; do
  gpu_id="${GPU_IDS[$((idx % NUM_GPUS))]}"
  entry="${CONFIGS[$idx]}"
  cfg_name="${entry%%|*}"
  ckpt_name="${entry##*|}"
  run_pair "${gpu_id}" "${cfg_name}" "${ckpt_name}"
done

wait
echo "[done] all Qwen-2.5-3B evaluations finished"
