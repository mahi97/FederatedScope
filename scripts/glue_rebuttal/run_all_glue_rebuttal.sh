#!/bin/bash
# Run all GLUE rebuttal baselines in parallel, one per GPU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_FILE="$SCRIPT_DIR/failure_report.log"

# Clear previous report
echo "GLUE Rebuttal Experiment Report - Started: $(date)" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "Launching svd on GPU 0..."
bash "$SCRIPT_DIR/run_svd.sh" 0 &
PID_SVD=$!

echo "Launching svd_no_gram on GPU 1..."
bash "$SCRIPT_DIR/run_svd_no_gram.sh" 1 &
PID_SVD_NO_GRAM=$!

echo "Launching full_rank on GPU 2..."
bash "$SCRIPT_DIR/run_full_rank.sh" 2 &
PID_FULL_RANK=$!

echo "Launching fedma on GPU 3..."
bash "$SCRIPT_DIR/run_fedma.sh" 3 &
PID_FEDMA=$!

echo "Launching fa_lora on GPU 4..."
bash "$SCRIPT_DIR/run_fa_lora.sh" 4 &
PID_FA_LORA=$!

echo "Launching factor_fedprox on GPU 5..."
bash "$SCRIPT_DIR/run_factor_fedprox.sh" 5 &
PID_FACTOR_FEDPROX=$!

echo "Launching factor_fedavg on GPU 6..."
bash "$SCRIPT_DIR/run_factor_fedavg.sh" 6 &
PID_FACTOR_FEDAVG=$!

# Wait for all to finish
wait $PID_SVD $PID_SVD_NO_GRAM $PID_FULL_RANK $PID_FEDMA $PID_FA_LORA $PID_FACTOR_FEDPROX $PID_FACTOR_FEDAVG

echo ""
echo "All baselines finished at $(date)"
echo "See failure report: $REPORT_FILE"
echo ""
echo "--- Failure Summary ---"
grep "\[FAILED\]" "$REPORT_FILE" || echo "No failures!"
