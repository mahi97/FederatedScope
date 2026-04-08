#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_glue_wallclock_matrix.py" "$@"
