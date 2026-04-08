#!/bin/bash
# Run all GLUE tasks for baseline: factor_fedprox (personalized lora_B)
# GPU: ${1:-5}

GPU_ID=${1:-5}
REPORT_FILE="scripts/glue_rebuttal_pB/failure_report.log"
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "========================================" >> "$REPORT_FILE"
echo "Baseline: factor_fedprox_pB | GPU: $GPU_ID | Started: $(date)" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

echo "[$(date)] Running factor_fedprox_pB | cola | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'cola@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_cola_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_matthews_correlation' \
  eval.metrics "['matthews_correlation']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | cola | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | mnli | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'mnli@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_mnli_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | mnli | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | mrpc | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'mrpc@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_mrpc_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | mrpc | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | qnli | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'qnli@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_qnli_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | qnli | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | qqp | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'qqp@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_qqp_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | qqp | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | rte | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'rte@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_rte_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | rte | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | sst2 | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'sst2@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_sst2_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | sst2 | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | stsb | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'stsb@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_stsb_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_pearson' \
  eval.metrics "['pearson']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | stsb | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | wnli | lda0.1 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda01.yaml \
  data.type 'wnli@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_wnli_lda01.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | wnli | lda0.1 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | cola | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'cola@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_cola_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_matthews_correlation' \
  eval.metrics "['matthews_correlation']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | cola | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | mnli | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'mnli@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_mnli_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | mnli | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | mrpc | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'mrpc@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_mrpc_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | mrpc | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | qnli | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'qnli@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_qnli_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | qnli | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | qqp | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'qqp@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_qqp_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | qqp | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | rte | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'rte@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_rte_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | rte | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | sst2 | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'sst2@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_sst2_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | sst2 | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | stsb | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'stsb@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_stsb_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_pearson' \
  eval.metrics "['pearson']" \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | stsb | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[$(date)] Running factor_fedprox_pB | wnli | lda0.25 | GPU $GPU_ID"
python $PROJ_DIR/federatedscope/main.py \
  --cfg $PROJ_DIR/federatedscope/glue/yamls/rebuttal_pB/roberta_large_factor_fedprox_lda025.yaml \
  data.type 'wnli@glue' \
  federate.save_to 'roberta_pB_factor_fedprox_wnli_lda025.ckpt' \
  eval.best_res_update_round_wise_key 'val_accuracy' \
  device $GPU_ID \
  eval_device $GPU_ID

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "[FAILED] factor_fedprox_pB | wnli | lda0.25 | GPU $GPU_ID | exit=$EXIT_CODE | $(date)" >> "$REPORT_FILE"
fi

echo "[DONE] factor_fedprox_pB finished on GPU $GPU_ID at $(date)" >> "$REPORT_FILE"
