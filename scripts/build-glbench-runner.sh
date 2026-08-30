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
glbench_commit="e99bc684272bffd68b06c998e272531c9c84330f"
source_dir="${GLBENCH_SOURCE_DIR:-${PVRGPU_GLBENCH_SOURCE_DIR}}"
adapter_dir="${project_dir}/tools/glbench-mesa"
output_path="${GLBENCH_OUTPUT:-${PVRGPU_GLBENCH_RUNNER}}"
target_arch="${OSX_ARCHS:-$(uname -m)}"

for command_name in brew git pkg-config file otool; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command was not found: ${command_name}" >&2
        exit 1
    fi
done

mesa_prefix="${MESA_PREFIX:-${PVRGPU_LLVMPIPE_MESA_PREFIX:-$(brew --prefix mesa)}}"
llvm_prefix="${LLVM_PREFIX:-$(brew --prefix llvm)}"
cxx="${CXX:-${llvm_prefix}/bin/clang++}"
if [[ ! -x "${cxx}" ]]; then
    echo "C++ compiler was not found: ${cxx}" >&2
    exit 1
fi
if [[ ! -f "${mesa_prefix}/lib/libEGL.dylib" ||
      ! -f "${mesa_prefix}/lib/libGLESv2.dylib" ]]; then
    echo "Mesa EGL/GLES libraries were not found under: ${mesa_prefix}" >&2
    exit 1
fi

mkdir -p "$(dirname "${source_dir}")" "$(dirname "${output_path}")"
if [[ -e "${source_dir}" && ! -d "${source_dir}/.git" ]]; then
    echo "GLBench source path exists but is not a git checkout: ${source_dir}" >&2
    exit 1
fi
if [[ ! -d "${source_dir}/.git" ]]; then
    git clone https://chromium.googlesource.com/chromiumos/platform/glbench \
        "${source_dir}"
fi
if ! git -C "${source_dir}" cat-file -e "${glbench_commit}^{commit}" 2>/dev/null; then
    git -C "${source_dir}" fetch origin "${glbench_commit}"
fi
git -C "${source_dir}" switch --detach "${glbench_commit}"
actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${glbench_commit}" ]]; then
    echo "Unexpected GLBench source commit: ${actual_commit}" >&2
    exit 1
fi

png_cflags=()
png_libs=()
while IFS= read -r flag; do
    [[ -n "${flag}" ]] && png_cflags+=("${flag}")
done < <(pkg-config --cflags-only-I libpng | tr ' ' '\n')
while IFS= read -r flag; do
    [[ -n "${flag}" ]] && png_libs+=("${flag}")
done < <(pkg-config --libs libpng | tr ' ' '\n')

sources=(
    "${adapter_dir}/main.cc"
    "${adapter_dir}/surfaceless_egl.cc"
    "${adapter_dir}/testbase.cc"
    "${source_dir}/src/attributefetchtest.cc"
    "${source_dir}/src/fillratetest.cc"
    "${source_dir}/src/trianglesetuptest.cc"
    "${source_dir}/src/varyingsandddxytest.cc"
    "${source_dir}/src/filepath.cc"
    "${source_dir}/src/md5.cc"
    "${source_dir}/src/png_helper.cc"
    "${source_dir}/src/utils.cc"
)

temporary_output="${output_path}.new"
"${cxx}" \
    -std=c++17 -O2 -g -Wall -Wextra \
    -arch "${target_arch}" \
    -DUSE_OPENGLES -DPLATFORM=PLATFORM_NULL \
    -I"${adapter_dir}/include" \
    -I"${source_dir}/src" \
    -I"${mesa_prefix}/include" \
    "${png_cflags[@]}" \
    "${sources[@]}" \
    -L"${mesa_prefix}/lib" -lEGL -lGLESv2 \
    "${png_libs[@]}" \
    "-Wl,-rpath,${mesa_prefix}/lib" \
    -o "${temporary_output}"
mv "${temporary_output}" "${output_path}"

file "${output_path}"
if ! otool -L "${output_path}" | grep -F "${mesa_prefix}/lib/libEGL" >/dev/null ||
   ! otool -L "${output_path}" | grep -F "${mesa_prefix}/lib/libGLESv2" >/dev/null; then
    echo "Runner did not link to the requested Mesa prefix: ${mesa_prefix}" >&2
    exit 1
fi
"${output_path}" --list >/dev/null

echo
echo "GLBench runner ready: ${output_path}"
echo "Mesa prefix:         ${mesa_prefix}"
echo "Upstream commit:     ${glbench_commit}"
