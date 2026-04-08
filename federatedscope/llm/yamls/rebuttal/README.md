# LLM rebuttal matrix

These configs are the fast rebuttal defaults for GSM8K training on a single 24 GB GPU per run.

Recommended primary setup:
- Model: `Qwen/Qwen2.5-3B-Instruct@huggingface_llm`
- Training task: `gsm8k@llm`
- Evaluation: validation during FL, then final checkpoint evaluation on GSM8K and MMLU

Why this setup:
- `Qwen2.5-3B-Instruct` is a current small open-weight instruct model and Qwen reports that even the 3B class is already competitive among small models.
- `Qwen2.5-1.5B-Instruct` is a safe fallback if the queue gets tight or you want a full matrix faster.
- `meta-llama/Llama-3.2-1B-Instruct` is acceptable as a cross-family sanity check, but the HF model is gated, so Qwen is the safer primary rebuttal path.

Important LLM-specific caveat:
- There is no classifier head here, so the GLUE setting "classifier personalized" has no direct LLM analogue.
- Do not run `fa-lora` with `personalization.local_param=['lora_B']`. With `freeze_A=True`, that would leave no shared trainable parameter and collapses into local-only behavior.
- Do not run `full_rank` with `personalization.local_param=['lora_B']`. It aggregates `B @ A`, so it needs both factors to be shared.

Training/eval helper behavior:
- Prefer the helper scripts below over raw `python federatedscope/main.py ...` commands.
- [run_llm_rebuttal.sh](/home/mahi/app/APRILS/scripts/run_llm_rebuttal.sh) converts a relative `save_to` into an absolute path under `/home/mahi/app/APRILS`, so the checkpoint location stays stable even though [main.py](/home/mahi/app/APRILS/federatedscope/main.py#L5) does `os.chdir('..')`.
- [eval_llm_checkpoint.sh](/home/mahi/app/APRILS/scripts/eval_llm_checkpoint.sh) resolves both `<save_to>` and `final_<save_to>`, and also searches older runs saved one directory above the repo.

Method behavior after the trainer fix:
- `factor_fedavg`, `factor_fedprox`, and `fa_lora` use normal client training.
- `fedma` and `full_rank` still use `trainer.type=llmsvdtrainer` in config, but client update/upload now fall back to plain `LLMTrainer`; their special logic is server-side aggregation only.
- `svd` and `svd_no_gram` use special client-side SVD logic only when `personalization.local_param=['lora_B']`. With shared `lora_B`, they now also fall back to plain client training.
- `fedma` with `personalization.local_param=['lora_B']` is still runnable, but it only aligns and averages the uploaded shared parameters. The local `lora_B` stays client-side; there is no server-side reconstruction of a global `B` in that variant.

## Training commands

Shared-B variants:

```bash
./scripts/run_llm_rebuttal.sh 1 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml \
  rebuttal_qwen25_3b_factor_fedavg.ckpt &

./scripts/run_llm_rebuttal.sh 2 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml \
  rebuttal_qwen25_3b_factor_fedprox.ckpt &

./scripts/run_llm_rebuttal.sh 3 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fa_lora.yaml \
  rebuttal_qwen25_3b_fa_lora.ckpt &

./scripts/run_llm_rebuttal.sh 4 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml \
  rebuttal_qwen25_3b_svd.ckpt &

./scripts/run_llm_rebuttal.sh 5 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml \
  rebuttal_qwen25_3b_svd_no_gram.ckpt &

./scripts/run_llm_rebuttal.sh 6 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml \
  rebuttal_qwen25_3b_fedma.ckpt &

./scripts/run_llm_rebuttal.sh 0 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_full_rank.yaml \
  rebuttal_qwen25_3b_full_rank.ckpt
```

Personalized-B variants:

```bash
./scripts/run_llm_rebuttal.sh 4 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml \
  rebuttal_qwen25_3b_factor_fedavg_pb.ckpt \
  personalization.local_param "['lora_B']"

./scripts/run_llm_rebuttal.sh 5 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml \
  rebuttal_qwen25_3b_factor_fedprox_pb.ckpt \
  personalization.local_param "['lora_B']"

./scripts/run_llm_rebuttal.sh 7 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml \
  rebuttal_qwen25_3b_svd_pb.ckpt \
  personalization.local_param "['lora_B']"

./scripts/run_llm_rebuttal.sh 1 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml \
  rebuttal_qwen25_3b_svd_no_gram_pb.ckpt \
  personalization.local_param "['lora_B']"

./scripts/run_llm_rebuttal.sh 4 \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml \
  rebuttal_qwen25_3b_fedma_pb.ckpt \
  personalization.local_param "['lora_B']"
```

## Final checkpoint evaluation

Keep the small GSM8K validation subset in the training configs for speed. Run full GSM8K test and full MMLU only once on the selected checkpoint at `federate.save_to`. The eval helper accepts either the exact checkpoint path or just the basename and will resolve `final_<save_to>` automatically.

GSM8K:

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml \
  rebuttal_qwen25_3b_svd.ckpt
```

MMLU:

```bash
./scripts/eval_llm_checkpoint.sh 4 mmlu \
  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml \
  rebuttal_qwen25_3b_svd.ckpt
```

## Model swaps

Full Qwen matrix faster:

```bash
model.type "Qwen/Qwen2.5-1.5B-Instruct@huggingface_llm"
```

Cross-family Llama sanity check:

```bash
model.type "meta-llama/Llama-3.2-1B-Instruct@huggingface_llm"
```

## Priority order

24-hour queue:
1. `svd`, `svd_no_gram`, `factor_fedavg`, `fedma` on Qwen 3B with shared B
2. `full_rank`, `factor_fedprox`, and `fa_lora` on Qwen 3B with shared B
3. Personalized-B variants for `svd`, `svd_no_gram`, `factor_fedavg`
4. Full GSM8K test and MMLU only for the strongest 3 to 4 checkpoints

48-hour queue:
1. Complete the remaining personalized-B variants
2. Run MMLU for the full shared-B table
3. Add one Llama-3.2-1B sanity table for `factor_fedavg`, `svd`, and `svd_no_gram`
