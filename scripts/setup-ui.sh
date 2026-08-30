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
venv_dir="${PVRGPU_UI_VENV}"
mkdir -p "${PVRGPU_WORK_ROOT}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install -r "${project_dir}/requirements-ui.txt"

# iCloud Drive can propagate the macOS hidden flag from a dot-directory to
# Qt's plug-in binaries.  QDir then skips the platform plug-ins even though
# importing PySide6 succeeds.  Clear that filesystem flag and validate a real
# headless QApplication, not only the Python import.
if [[ "$(uname -s)" == "Darwin" ]] && command -v chflags >/dev/null 2>&1; then
    chflags -R nohidden "${venv_dir}"
fi
if ! QT_QPA_PLATFORM=offscreen "${venv_dir}/bin/python" -c \
    'from PySide6.QtWidgets import QApplication; app = QApplication([])' \
    >/dev/null 2>&1; then
    echo "PySide6 installed, but its Qt platform plug-in could not start." >&2
    echo "Environment: ${venv_dir}" >&2
    exit 1
fi

echo
echo "Qt environment ready: ${venv_dir}"
printf 'Run: %q\n' "${project_dir}/scripts/run-pvrgpu-control.sh"
