#!/usr/bin/env bash
# Catalog and optionally replay pre-recorded dEQP RDC captures.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

mkdir -p "${PVRGPU_WORK_ROOT}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PVRGPU_PROJECT_ROOT="${project_dir}"
export PVRGPU_WORK_ROOT

python_bin="${PVRGPU_COUNTER_PYTHON:-python3}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python 3 is required: ${python_bin}" >&2
    exit 2
fi

exec "${python_bin}" "${project_dir}/tools/deqp_capture_report.py" "$@"
