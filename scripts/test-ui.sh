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
mkdir -p "${PVRGPU_WORK_ROOT}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"

ui_python="${PVRGPU_UI_PYTHON}"
if [[ "$(uname -s)" == "Darwin" ]] && command -v chflags >/dev/null 2>&1 && \
   [[ -d "${PVRGPU_UI_VENV}" ]]; then
    chflags -R nohidden "${PVRGPU_UI_VENV}"
fi
if [[ ! -x "${ui_python}" ]] || ! QT_QPA_PLATFORM=offscreen "${ui_python}" -c \
    'from PySide6.QtWidgets import QApplication; app = QApplication([])' \
    >/dev/null 2>&1; then
    echo "A Python environment with PySide6 is required." >&2
    exit 1
fi

export QT_QPA_PLATFORM=offscreen
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${project_dir}/tools"
export PVRGPU_PROJECT_ROOT="${project_dir}"
export PVRGPU_WORK_ROOT
export PVRGPU_GLBENCH_RUNNER
export PVRGPU_LLVMPIPE_MESA_PREFIX="${PVRGPU_LLVMPIPE_MESA_PREFIX:-}"
export PVRGPU_NATIVE_GLBENCH_RUNNER="${PVRGPU_NATIVE_GLBENCH_RUNNER:-}"
export PVRGPU_NATIVE_MESA_PREFIX="${PVRGPU_NATIVE_MESA_PREFIX:-}"
export PVRGPU_MODEL_STUB
export PVRGPU_OUTPUT_ROOT="${PVRGPU_WORK_ROOT}/out/ui-test"

"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" --backend llvmpipe
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" --backend pvrgpu
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --cache-bypass on
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_solid_depth_never
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_solid_depth_neq
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_solid_blended
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case triangle_setup
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case triangle_setup_all_culled
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case triangle_setup_half_culled
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case attribute_fetch_shader
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case attribute_fetch_shader_2_attr
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case attribute_fetch_shader_4_attr
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case attribute_fetch_shader_8_attr
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case varyings_shader_1
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case varyings_shader_2
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case varyings_shader_4
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case varyings_shader_8
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_tex_nearest --width 64 --height 64 --timeout-ms 30000
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_tex_bilinear --width 64 --height 64 --timeout-ms 30000
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_tex_trilinear_linear_01 \
    --width 64 --height 64 --timeout-ms 60000
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_tex_trilinear_linear_04 \
    --width 64 --height 64 --timeout-ms 60000
"${ui_python}" "${project_dir}/tests/ui_process_smoke.py" \
    --backend pvrgpu --case fill_tex_trilinear_linear_05 \
    --width 64 --height 64 --timeout-ms 60000
