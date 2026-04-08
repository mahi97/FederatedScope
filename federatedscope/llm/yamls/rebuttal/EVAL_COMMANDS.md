# Qwen-2.5-3B Evaluation Commands

This file lists the full evaluation commands for every trained Qwen-2.5-3B rebuttal checkpoint.

Notes:
- Use [eval_llm_checkpoint.sh](/home/mahi/app/APRILS/scripts/eval_llm_checkpoint.sh), not raw eval python commands.
- The helper resolves both `<save_to>` and `final_<save_to>`.
- Replace the GPU id if needed.
- `fa_lora` with personalized `lora_B` and `full_rank` with personalized `lora_B` are intentionally omitted because those runs are not valid.

## Shared-B Checkpoints

### factor_fedavg

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml rebuttal_qwen25_3b_factor_fedavg.ckpt
./scripts/eval_llm_checkpoint.sh 4 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml rebuttal_qwen25_3b_factor_fedavg.ckpt
```

### factor_fedprox

```bash
./scripts/eval_llm_checkpoint.sh 0 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml rebuttal_qwen25_3b_factor_fedprox.ckpt
./scripts/eval_llm_checkpoint.sh 0 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml rebuttal_qwen25_3b_factor_fedprox.ckpt
```

### fa_lora

```bash
./scripts/eval_llm_checkpoint.sh 1 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fa_lora.yaml rebuttal_qwen25_3b_fa_lora.ckpt
./scripts/eval_llm_checkpoint.sh 1 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fa_lora.yaml rebuttal_qwen25_3b_fa_lora.ckpt
```

### svd

```bash
./scripts/eval_llm_checkpoint.sh 2 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml rebuttal_qwen25_3b_svd.ckpt
./scripts/eval_llm_checkpoint.sh 2 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml rebuttal_qwen25_3b_svd.ckpt
```

### svd_no_gram

```bash
./scripts/eval_llm_checkpoint.sh 3 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml rebuttal_qwen25_3b_svd_no_gram.ckpt
./scripts/eval_llm_checkpoint.sh 3 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml rebuttal_qwen25_3b_svd_no_gram.ckpt
```

### fedma

```bash
./scripts/eval_llm_checkpoint.sh 5 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml rebuttal_qwen25_3b_fedma.ckpt
./scripts/eval_llm_checkpoint.sh 5 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml rebuttal_qwen25_3b_fedma.ckpt
```

### full_rank

```bash
./scripts/eval_llm_checkpoint.sh 6 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_full_rank.yaml rebuttal_qwen25_3b_full_rank.ckpt
./scripts/eval_llm_checkpoint.sh 6 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_full_rank.yaml rebuttal_qwen25_3b_full_rank.ckpt
```

## Personalized-B Checkpoints

### factor_fedavg_pb

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml rebuttal_qwen25_3b_factor_fedavg_pb.ckpt
./scripts/eval_llm_checkpoint.sh 4 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml rebuttal_qwen25_3b_factor_fedavg_pb.ckpt
```

### factor_fedprox_pb

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml rebuttal_qwen25_3b_factor_fedprox_pb.ckpt
./scripts/eval_llm_checkpoint.sh 4 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml rebuttal_qwen25_3b_factor_fedprox_pb.ckpt
```

### svd_pb

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml rebuttal_qwen25_3b_svd_pb.ckpt
./scripts/eval_llm_checkpoint.sh 4 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml rebuttal_qwen25_3b_svd_pb.ckpt
```

### svd_no_gram_pb

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml rebuttal_qwen25_3b_svd_no_gram_pb.ckpt
./scripts/eval_llm_checkpoint.sh 4 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml rebuttal_qwen25_3b_svd_no_gram_pb.ckpt
```

### fedma_pb

```bash
./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml rebuttal_qwen25_3b_fedma_pb.ckpt
./scripts/eval_llm_checkpoint.sh 4 mmlu  /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml rebuttal_qwen25_3b_fedma_pb.ckpt
```


## All together

```bash
./scripts/eval_llm_checkpoint.sh 0 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedavg.yaml ckpt/final_rebuttal_qwen25_3b_factor_fedavg_new.ckpt &

./scripts/eval_llm_checkpoint.sh 1 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_factor_fedprox.yaml ckpt/final_rebuttal_qwen25_3b_factor_fedprox_new.ckpt &

./scripts/eval_llm_checkpoint.sh 2 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd.yaml ckpt/final_rebuttal_qwen25_3b_svd_new.ckpt &

./scripts/eval_llm_checkpoint.sh 3 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_svd_no_gram.yaml ckpt/final_rebuttal_qwen25_3b_svd_no_gram_new.ckpt &

./scripts/eval_llm_checkpoint.sh 4 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fedma.yaml ckpt/final_rebuttal_qwen25_3b_fedma_new.ckpt &

./scripts/eval_llm_checkpoint.sh 5 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_full_rank.yaml ckpt/final_rebuttal_qwen25_3b_full_rank_new.ckpt &

./scripts/eval_llm_checkpoint.sh 6 gsm8k /home/mahi/app/APRILS/federatedscope/llm/yamls/rebuttal/qwen25_3b_fa_lora.yaml ckpt/final_rebuttal_qwen25_3b_fa_lora_new.ckpt &

wait

```