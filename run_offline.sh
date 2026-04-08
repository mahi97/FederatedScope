#!/bin/sh
# run_offline.sh – run experiments in OFFLINE mode.
# All models/datasets must already be cached via cache_offline.sh.

set -eu

storage_root="${STORAGE_ROOT:-/drive1/mahi}"
hf_home="${HF_HOME:-$storage_root/huggingface}"
hf_hub_cache="${HUGGINGFACE_HUB_CACHE:-$hf_home/hub}"
transformers_cache="${TRANSFORMERS_CACHE:-$hf_home/hub}"
hf_datasets_cache="${HF_DATASETS_CACHE:-$storage_root/data}"
output_dir="${APRILS_OUTPUT_DIR:-$storage_root/output}"
exp_dir="${APRILS_EXP_DIR:-$storage_root/experiments}"
wandb_dir="${WANDB_DIR:-$storage_root/wandb}"
xdg_cache="${XDG_CACHE_HOME:-$storage_root/.cache}"
torch_home="${TORCH_HOME:-$storage_root/torch}"

cd "$(dirname "$0")"

exec env -u VIRTUAL_ENV \
  STORAGE_ROOT="$storage_root" \
  HF_HOME="$hf_home" \
  HUGGINGFACE_HUB_CACHE="$hf_hub_cache" \
  TRANSFORMERS_CACHE="$transformers_cache" \
  HF_DATASETS_CACHE="$hf_datasets_cache" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  APRILS_OUTPUT_DIR="$output_dir" \
  APRILS_EXP_DIR="$exp_dir" \
  WANDB_DIR="$wandb_dir" \
  WANDB_MODE=offline \
  XDG_CACHE_HOME="$xdg_cache" \
  TORCH_HOME="$torch_home" \
  uv run --env-file .env.offline --extra llm python federatedscope/main.py \
  "$@"
