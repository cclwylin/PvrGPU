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
build_dir="${PVRGPU_BUILD_DIR}"
mkdir -p "${PVRGPU_WORK_ROOT}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"
compiler="${CXX:-}"
if [[ -z "${compiler}" && -x /usr/local/opt/llvm/bin/clang++ ]]; then
    compiler=/usr/local/opt/llvm/bin/clang++
fi

cmake_args=(
    -S "${project_dir}"
    -B "${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
)
if [[ -n "${compiler}" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER=${compiler}")
fi

"${project_dir}/scripts/check-systemc-module-layout.py"
cmake "${cmake_args[@]}"
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" --output-on-failure
"${project_dir}/scripts/check-source-tree.sh"

echo
echo "Built: ${build_dir}/bin/pvrgpu-model-stub"
