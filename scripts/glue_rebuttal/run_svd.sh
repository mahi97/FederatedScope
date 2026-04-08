#!/bin/bash
# Run all GLUE tasks for baseline: svd
# GPU: ${1:-0}

GPU_ID=${1:-0}
REPORT_FILE="scripts/glue_rebuttal/failure_report.log"
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "========================================" >> "$REPORT_FILE"
echo "Baseline: svd | GPU: $GPU_ID | Started: $(date)" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

echo "[$(date)] Running svd | cola | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'cola@glue' \
  federate.save_to 'roberta_svd_cola_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_matthews_correlation' \
  eval.metrics "['matthews_correlation']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | cola | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | mnli | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'mnli@glue' \
  federate.save_to 'roberta_svd_mnli_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | mnli | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | mrpc | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'mrpc@glue' \
  federate.save_to 'roberta_svd_mrpc_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | mrpc | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | qnli | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'qnli@glue' \
  federate.save_to 'roberta_svd_qnli_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | qnli | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | qqp | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'qqp@glue' \
  federate.save_to 'roberta_svd_qqp_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | qqp | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | rte | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'rte@glue' \
  federate.save_to 'roberta_svd_rte_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | rte | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | sst2 | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'sst2@glue' \
  federate.save_to 'roberta_svd_sst2_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | sst2 | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | stsb | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'stsb@glue' \
  federate.save_to 'roberta_svd_stsb_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_pearson' \
  eval.metrics "['pearson']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | stsb | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | wnli | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda01.yaml \
  data.type 'wnli@glue' \
  federate.save_to 'roberta_svd_wnli_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | wnli | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | cola | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'cola@glue' \
  federate.save_to 'roberta_svd_cola_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_matthews_correlation' \
  eval.metrics "['matthews_correlation']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | cola | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | mnli | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'mnli@glue' \
  federate.save_to 'roberta_svd_mnli_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | mnli | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | mrpc | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'mrpc@glue' \
  federate.save_to 'roberta_svd_mrpc_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | mrpc | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | qnli | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'qnli@glue' \
  federate.save_to 'roberta_svd_qnli_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | qnli | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | qqp | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'qqp@glue' \
  federate.save_to 'roberta_svd_qqp_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | qqp | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | rte | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'rte@glue' \
  federate.save_to 'roberta_svd_rte_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | rte | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | sst2 | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'sst2@glue' \
  federate.save_to 'roberta_svd_sst2_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | sst2 | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | stsb | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'stsb@glue' \
  federate.save_to 'roberta_svd_stsb_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_pearson' \
  eval.metrics "['pearson']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | stsb | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running svd | wnli | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal/roberta_large_svd_lda025.yaml \
  data.type 'wnli@glue' \
  federate.save_to 'roberta_svd_wnli_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] svd | wnli | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[DONE] svd finished on GPU $GPU_ID at $(date)" >> "$REPORT_FILE"
