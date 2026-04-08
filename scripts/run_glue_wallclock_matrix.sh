#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

exec "${PYTHON_BIN}" /home/mahi/app/APRILS/scripts/run_glue_wallclock_matrix.py "$@"
