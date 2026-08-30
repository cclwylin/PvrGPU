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

rdc_counter_output="${PVRGPU_RDC_COUNTER_OUTPUT:-${PVRGPU_WORK_ROOT}/out/rdc-counter-report}"
mkdir -p "${PVRGPU_WORK_ROOT}" "${rdc_counter_output}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"

# iCloud can restore the venv with Qt plug-ins marked hidden. QDir skips them.
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
export PVRGPU_TMP_ROOT
export PVRGPU_RDC_ROOT="${PVRGPU_RDC_ROOT:-}"
export PVRGPU_RDC_COUNTER_OUTPUT="${rdc_counter_output}"
export PVRGPU_BENCHSCOPE_ROOT="${PVRGPU_BENCHSCOPE_ROOT:-}"
export PVRGPU_LLVMPIPE_MESA_PREFIX="${PVRGPU_LLVMPIPE_MESA_PREFIX:-}"
export PVRGPU_RENDERDOC_MESA_ROOT="${PVRGPU_RENDERDOC_MESA_ROOT:-}"
default_pvrgpu_runner="${project_dir}/scripts/run-rdc-pvrgpu-driver-systemc.sh"
export PVRGPU_RDC_PVRGPU_RUNNER="${PVRGPU_RDC_PVRGPU_RUNNER:-${PVRGPU_RDC_DUT_RUNNER:-${default_pvrgpu_runner}}}"
export PVRGPU_MODEL_STUB

"${ui_python}" "${project_dir}/tools/rdc_counter_ui.py" "$@" 2> >(
    grep -v -F "error messaging the mach port for IMKCFRunLoopWakeUpReliable" >&2
)
