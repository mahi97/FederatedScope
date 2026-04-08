# LLM Rebuttal Run Order

This file lists every config and helper script to run for the LLM rebuttal matrix.

## Files

Training helper:
- `/home/mahi/app/APRILS/scripts/run_llm_rebuttal.sh`

Evaluation helper:
- `/home/mahi/app/APRILS/scripts/eval_llm_checkpoint.sh`
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/EVAL_COMMANDS.md`

Checkpoint path behavior:
- The helper scripts resolve relative checkpoint names under `/home/mahi/app/APRILS`.
- The eval helper also searches `final_<save_to>` and older runs saved under `/home/mahi/app`, which matters because [main.py](/home/mahi/app/APRILS/federatedscope/main.py#L5) changes the working directory internally.

Shared-B configs:
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml`
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml`
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fa_lora.yaml`
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml`
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml`
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml`
- `/home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_full_rank.yaml`

## Meaning of the extra modes

- `qwen25_3b_fa_lora.yaml`: freeze `lora_A` from config. Run this only when `lora_B` is shared.
- `qwen25_3b_full_rank.yaml`: aggregate `W = B @ A`, average in full rank, then factorize back into LoRA factors by rank-`r` SVD.

Client trainer behavior:
- `fedma` and `full_rank` finish their special work at the server aggregator. Their clients now use plain update/upload behavior.
- Shared-B `svd` and `svd_no_gram` also use plain client update/upload behavior.
- Only `svd` or `svd_no_gram` with `personalization.local_param=['lora_B']` use the special `LLMSVDTrainer` client logic.
- `fedma` with `personalization.local_param=['lora_B']` remains a valid run only if you interpret it as aligned averaging over shared parameters while local `lora_B` stays private. It does not reconstruct a shared `B` on the server.

## Do not run

- Do not run `fa_lora` with `personalization.local_param=['lora_B']`.
- Do not run `full_rank` with `personalization.local_param=['lora_B']`.

Both require shared `lora_B` because they aggregate both factors.

## Phase 1: Main 24-hour table

Run these first. They are the most important rebuttal evidence.

```bash
./scripts/run_llm_rebuttal.sh 0 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml ckpt/rebuttal_qwen25_3b_svd_new.ckpt &
./scripts/run_llm_rebuttal.sh 1 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml ckpt/rebuttal_qwen25_3b_svd_no_gram_new.ckpt &
./scripts/run_llm_rebuttal.sh 2 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml ckpt/rebuttal_qwen25_3b_factor_fedavg_new.ckpt &
./scripts/run_llm_rebuttal.sh 3 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml ckpt/rebuttal_qwen25_3b_fedma_new.ckpt &
./scripts/run_llm_rebuttal.sh 4 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_full_rank.yaml ckpt/rebuttal_qwen25_3b_full_rank_new.ckpt &
./scripts/run_llm_rebuttal.sh 5 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml ckpt/rebuttal_qwen25_3b_factor_fedprox_new.ckpt &
./scripts/run_llm_rebuttal.sh 6 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fa_lora.yaml ckpt/rebuttal_qwen25_3b_fa_lora_new.ckpt &

wait
```

## Phase 3: Personalized-B runs

Run only for methods where shared aggregation still makes sense with local `lora_B`.

```bash
./scripts/run_llm_rebuttal.sh 4 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml ckpt/rebuttal_qwen25_3b_factor_fedavg_pb_new.ckpt personalization.local_param "['lora_B']"
./scripts/run_llm_rebuttal.sh 5 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml ckpt/rebuttal_qwen25_3b_factor_fedprox_pb_new.ckpt personalization.local_param "['lora_B']"
./scripts/run_llm_rebuttal.sh 6 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml ckpt/rebuttal_qwen25_3b_svd_pb_new.ckpt personalization.local_param "['lora_B']"
./scripts/run_llm_rebuttal.sh 7 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml ckpt/rebuttal_qwen25_3b_svd_no_gram_pb_new.ckpt personalization.local_param "['lora_B']"
./scripts/run_llm_rebuttal.sh 4 /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml ckpt/rebuttal_qwen25_3b_fedma_pb_new.ckpt personalization.local_param "['lora_B']"
```

## Phase 4: Final evaluation

Run GSM8K full test and MMLU only for the strongest checkpoints first.

Example GSM8K:

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml ckpt/rebuttal_qwen25_3b_svd_new.ckpt
```

Example MMLU:

```bash
./scripts/eval_llm_checkpoint.sh 4 mmlu /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml ckpt/rebuttal_qwen25_3b_svd_new.ckpt
```

## If time is tight

- Switch the same configs to `Qwen/Qwen2.5-1.5B-Instruct@huggingface_llm`.
- Keep only: `factor_fedavg`, `svd`, `svd_no_gram`, `fedma`, `full_rank`.
- Evaluate only the top 3 checkpoints on full GSM8K and MMLU.
