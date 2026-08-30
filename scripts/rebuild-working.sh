#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${project_dir}/scripts/setup-ui.sh"
"${project_dir}/scripts/build.sh"
"${project_dir}/scripts/build-counter-mesa.sh"
"${project_dir}/scripts/build-glbench-runner.sh"
"${project_dir}/scripts/run-path-smoke.sh"
"${project_dir}/scripts/test-ui.sh"
"${project_dir}/scripts/check-source-tree.sh"

echo
echo "PvrGPU Working runtime rebuilt and verified."
