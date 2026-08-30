#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"

if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"
mkdir -p "${PVRGPU_WORK_ROOT}" "${PVRGPU_OUTPUT_ROOT}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"

# Clear a stale hidden flag if this environment was restored from iCloud or a
# previous dot-directory. Qt excludes hidden platform plug-ins while scanning.
if [[ "$(uname -s)" == "Darwin" ]] && command -v chflags >/dev/null 2>&1 && \
   [[ -d "${PVRGPU_UI_VENV}" ]]; then
    chflags -R nohidden "${PVRGPU_UI_VENV}"
fi

ui_python=""
python_candidates=(
    "${PVRGPU_UI_PYTHON}"
    "$(command -v python3 2>/dev/null || true)"
)
for candidate in "${python_candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if [[ "${candidate}" != /* ]]; then
        candidate="${project_dir}/${candidate}"
    fi
    if [[ -x "${candidate}" ]] && \
       QT_QPA_PLATFORM=offscreen "${candidate}" -c \
           'from PySide6.QtWidgets import QApplication; app = QApplication([])' \
           >/dev/null 2>&1; then
        ui_python="${candidate}"
        break
    fi
done
if [[ -z "${ui_python}" ]]; then
    echo "A Python environment with PySide6 is required." >&2
    echo "Run once: ${project_dir}/scripts/setup-ui.sh" >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export PVRGPU_PROJECT_ROOT="${project_dir}"
export PVRGPU_WORK_ROOT
export PVRGPU_GLBENCH_RUNNER
export PVRGPU_LLVMPIPE_MESA_PREFIX="${PVRGPU_LLVMPIPE_MESA_PREFIX:-}"
export PVRGPU_NATIVE_MESA_PREFIX="${PVRGPU_NATIVE_MESA_PREFIX:-}"
export PVRGPU_NATIVE_GLBENCH_RUNNER="${PVRGPU_NATIVE_GLBENCH_RUNNER:-}"
export PVRGPU_MODEL_STUB
export PVRGPU_OUTPUT_ROOT

exec "${ui_python}" "${project_dir}/tools/pvrgpu_control.py" "$@"
