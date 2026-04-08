#!/bin/sh
# Source this file to configure all storage paths from STORAGE_ROOT.
# Alternatively, use the .env.online / .env.offline files with uv run.

export STORAGE_ROOT="${STORAGE_ROOT:-/drive1/mahi}"
export HF_HOME="$STORAGE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$STORAGE_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$STORAGE_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="$STORAGE_ROOT/data"
export APRILS_OUTPUT_DIR="$STORAGE_ROOT/output"
export APRILS_EXP_DIR="$STORAGE_ROOT/experiments"
export WANDB_DIR="$STORAGE_ROOT/wandb"
export WANDB_CACHE_DIR="$STORAGE_ROOT/wandb/.cache"
export XDG_CACHE_HOME="$STORAGE_ROOT/.cache"
export TORCH_HOME="$STORAGE_ROOT/torch"
