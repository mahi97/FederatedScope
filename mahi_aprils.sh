#!/bin/sh
# Distributed offline training on GLUE tasks.
# All paths are derived from STORAGE_ROOT (set in .env files).

set -euo pipefail

export STORAGE_ROOT="${STORAGE_ROOT:-/drive1/mahi}"
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Storage paths (all under STORAGE_ROOT)
export HF_HOME="$STORAGE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$STORAGE_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$STORAGE_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="$STORAGE_ROOT/data"
export APRILS_OUTPUT_DIR="$STORAGE_ROOT/output"
export APRILS_EXP_DIR="$STORAGE_ROOT/experiments"
export WANDB_DIR="$STORAGE_ROOT/wandb"
export XDG_CACHE_HOME="$STORAGE_ROOT/.cache"
export TORCH_HOME="$STORAGE_ROOT/torch"

# Offline mode
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SET=$(seq 1 1)

for quality in $SET; do
  export MASTER_PORT=$((29500 + RANDOM % 1000))
  echo "=== quality=${quality}, MASTER_PORT=${MASTER_PORT} ==="
  python federatedscope/main.py --cfg federatedscope/glue/yamls/svd-lora.yaml
done

wait
