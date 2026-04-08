# PERSIA

---

## Storage Layout

**All** heavy files — models, datasets, caches, experiment outputs, wandb logs — live under a single **storage root** directory (e.g. `/drive/`). Nothing is ever written to `~/.cache` or the project working directory.

```
/drive/                       ← STORAGE_ROOT
├── huggingface/                    ← HF_HOME  (model weights & tokenizers)
│   └── hub/                        ← HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE
├── data/                           ← HF_DATASETS_CACHE  (GLUE, GSM8K, etc.)
├── output/                         ← APRILS_OUTPUT_DIR  (run outputs)
├── experiments/                    ← APRILS_EXP_DIR  (experiment logs & configs)
├── wandb/                          ← WANDB_DIR  (Weights & Biases logs)
│   └── .cache/                     ← WANDB_CACHE_DIR
├── torch/                          ← TORCH_HOME
└── .cache/                         ← XDG_CACHE_HOME  (generic cache redirect)
```

The path is configured once in the `.env` files and propagated everywhere automatically.

---

## Online / Offline Modes

The project supports two execution modes controlled by environment files:

| File | HF_HUB_OFFLINE | Purpose |
|------|:-:|---------|
| `.env.online` | `0` | Allows downloads — missing resources are fetched and stored under `STORAGE_ROOT` |
| `.env.offline` | `1` | **Never** downloads — uses only what is already cached; reports what is missing |

**Rule:** run something once in **online** mode and you can run it in **offline** mode forever.

---

## Installation

```shell
uv python install 3.10
uv sync --locked --extra llm
```

Verify that PyTorch can see the GPU:

```shell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Preparing for Offline Use (one-time, online mode)

Download every model and dataset the project needs:

```shell
bash cache_offline.sh            # download everything
bash cache_offline.sh --check-only   # verify what is cached
bash cache_offline.sh --glue-only    # download only GLUE resources
bash cache_offline.sh --llm-only     # download only LLM resources
```

After this, all resources live under `STORAGE_ROOT` and offline mode works.

---

## Running Experiments

### Online mode (downloads missing resources on the fly)

```shell
bash run_online.sh --cfg federatedscope/glue/yamls/svd-lora.yaml device 0
```

### Offline mode (no network access needed)

```shell
bash run_offline.sh --cfg federatedscope/glue/yamls/svd-lora.yaml device 0
```

### Direct uv invocation

```shell
# Online
uv run --env-file .env.online --extra llm python federatedscope/main.py \
  --cfg federatedscope/glue/yamls/svd-lora.yaml device 0

# Offline
uv run --env-file .env.offline --extra llm python federatedscope/main.py \
  --cfg federatedscope/glue/yamls/svd-lora.yaml device 0
```

---

## Environment Variables Reference

All variables are set in `.env.online` / `.env.offline`. The scripts (`run_online.sh`, `run_offline.sh`, `cache_offline.sh`) forward them automatically.

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_ROOT` | `/drive/` | Root for all heavy storage |
| `HF_HOME` | `$STORAGE_ROOT/huggingface` | HuggingFace home (models) |
| `HUGGINGFACE_HUB_CACHE` | `$HF_HOME/hub` | Hub cache directory |
| `TRANSFORMERS_CACHE` | `$HF_HOME/hub` | Transformers model cache |
| `HF_DATASETS_CACHE` | `$STORAGE_ROOT/data` | Dataset cache |
| `APRILS_OUTPUT_DIR` | `$STORAGE_ROOT/output` | Run outputs |
| `APRILS_EXP_DIR` | `$STORAGE_ROOT/experiments` | Experiment configs & logs (used as `outdir`) |
| `WANDB_DIR` | `$STORAGE_ROOT/wandb` | Weights & Biases run directory |
| `WANDB_CACHE_DIR` | `$STORAGE_ROOT/wandb/.cache` | WandB cache |
| `XDG_CACHE_HOME` | `$STORAGE_ROOT/.cache` | Prevents writes to `~/.cache` |
| `TORCH_HOME` | `$STORAGE_ROOT/torch` | PyTorch hub cache |
| `HF_HUB_OFFLINE` | `0` or `1` | Disable HF Hub network calls |
| `TRANSFORMERS_OFFLINE` | `0` or `1` | Disable transformers downloads |
| `HF_DATASETS_OFFLINE` | `0` or `1` | Disable datasets downloads |

To change the storage root, edit **only** `STORAGE_ROOT` in both `.env` files — every other path derives from it.

---

## Customising the Storage Root

If your fast storage is mounted elsewhere, update the `.env` files:

```shell
# .env.online  &  .env.offline
STORAGE_ROOT=/scratch/my_user
```

Or export it before running:

```shell
STORAGE_ROOT=/scratch/my_user bash run_offline.sh --cfg ...
```

---
