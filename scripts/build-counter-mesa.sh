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

benchscope_dir="${PVRGPU_BENCHSCOPE_ROOT:-$(cd "${project_dir}/.." && pwd)/BenchScope}"
builder="${PVRGPU_COUNTER_MESA_BUILDER:-${benchscope_dir}/scripts/build-mesa-counter.sh}"
counter_root="${PVRGPU_MESA_COUNTER_ROOT:-${PVRGPU_WORK_ROOT}/mesa-counter}"
default_extra_patch="${project_dir}/third_party/mesa-26.2.1-llvmpipe-gs-zero-output-stats.patch"
if [[ ! -x "${builder}" ]]; then
    echo "Counter-Mesa builder was not found: ${builder}" >&2
    echo "Set PVRGPU_COUNTER_MESA_BUILDER to the BenchScope build script." >&2
    exit 1
fi

mkdir -p "${PVRGPU_WORK_ROOT}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"
export MESA_COUNTER_ROOT="${counter_root}"
export MESA_BUILD_JOBS="${MESA_BUILD_JOBS:-4}"
if [[ -z "${MESA_EXTRA_PATCH+x}" ]]; then
    export MESA_EXTRA_PATCH="${default_extra_patch}"
else
    export MESA_EXTRA_PATCH
fi

"${builder}"

expected_prefix="${counter_root}/install"
if [[ ! -f "${expected_prefix}/lib/libEGL.dylib" ||
      ! -f "${expected_prefix}/lib/libGLESv2.dylib" ]]; then
    echo "Counter-enabled Mesa install is incomplete: ${expected_prefix}" >&2
    exit 1
fi
echo "PvrGPU counter-Mesa prefix ready: ${expected_prefix}"
