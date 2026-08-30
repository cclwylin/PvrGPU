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

runner="${PVRGPU_GLBENCH_RUNNER}"
mesa_prefix="${PVRGPU_LLVMPIPE_MESA_PREFIX:-}"
output_dir="${PVRGPU_SMOKE_OUTPUT}"

if [[ ! -x "${runner}" ]]; then
    echo "GLBench runner is missing: ${runner:-<unset>}" >&2
    exit 1
fi
if [[ ! -f "${mesa_prefix}/lib/libEGL.dylib" ]]; then
    echo "Mesa prefix is missing libEGL.dylib: ${mesa_prefix:-<unset>}" >&2
    exit 1
fi

mkdir -p "${output_dir}"
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export LIBGL_DRIVERS_PATH="${mesa_prefix}/lib/dri"
export DYLD_LIBRARY_PATH="${mesa_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export MESA_SHADER_CACHE_DISABLE=true
export MESA_COUNTER_REPORT_PATH="${output_dir}/Report.md"
export MESA_COUNTER_FRAME_SELECTION_MS="PvrGPU llvmpipe smoke"

(
    cd "${PVRGPU_WORK_ROOT}"
    "${runner}" \
        --test fill_rate \
        --case fill_solid \
        --sample 1 \
        --size 256x256 \
        --outdir "${output_dir}"
)

test -s "${output_dir}/Report.md"
test -s "${output_dir}/fill_solid_sample_000001.png"
grep -q 'llvmpipe' "${output_dir}/Report.md"
grep -q 'ps_invocations' "${output_dir}/Report.md"

echo "Path verified: App(GLBench) -> Mesa -> llvmpipe"
echo "Counters: ${output_dir}/Report.md"
echo "Frame:    ${output_dir}/fill_solid_sample_000001.png"
