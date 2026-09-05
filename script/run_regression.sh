#!/usr/bin/env bash
# ==============================================================================
# PvrGPU RDC Test Pattern Batch Regression Script
# ==============================================================================
# Usage:
#   ./script/run_regression.sh                       # Run all patterns (default: 4 jobs)
#   ./script/run_regression.sh --suite GLBench       # Run only GLBench patterns
#   ./script/run_regression.sh --suite glmark2       # Run only glmark2 patterns
#   ./script/run_regression.sh --suite dEQP --limit 50 # Run first 50 dEQP patterns
#   ./script/run_regression.sh --suite GFXBench      # Run only GFXBench patterns
#   ./script/run_regression.sh --skip-passed         # Resume / skip already passed tests
#   ./script/run_regression.sh --list-only           # Just list discovered patterns
#   ./script/run_regression.sh -j 8                  # Run with 8 parallel workers
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${REPO_DIR}/tools/run_rdc_regression.py"

# Default pattern root
PATTERNS_DIR="${PATTERNS_DIR:-/Users/linwanyi/Downloads/Working/GPU_TestPatterns}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/rdc_regression}"

# Ensure python3 exists
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not found in PATH" >&2
    exit 1
fi

exec python3 "${PYTHON_SCRIPT}" \
    --pattern-dir "${PATTERNS_DIR}" \
    --out-dir "${OUTPUT_DIR}" \
    "$@"
