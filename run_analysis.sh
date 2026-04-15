#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
ANALYSIS_DIR="${SCRIPT_DIR}/uber-analysis"

if [[ -d "${VENV_DIR}" ]]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "Warning: no .venv found at ${VENV_DIR}, using system Python"
fi

echo "=========================================="
echo "  Uber Ride Cancellation Analysis"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

PYTHONPATH="${ANALYSIS_DIR}/src:${PYTHONPATH:-}" \
    python -m analysis.run

echo ""
echo "Done."
