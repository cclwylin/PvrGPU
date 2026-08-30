#!/usr/bin/env bash
# Build the pinned Mesa/POC runtime used by run-rdc-pvrgpu-poc.sh.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

usage() {
    cat <<'EOF'
Usage: ./scripts/build-rdc-poc-mesa.sh

Build Mesa 26.2.1 with the PvrGPU RDC POC Gallium-trace patch. The resulting
prefix defaults to PVRGPU_WORK_ROOT/mesa-poc/install.
EOF
}

if (($# > 0)); then
    if (($# == 1)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi
    echo "This builder does not accept positional arguments." >&2
    usage >&2
    exit 2
fi

benchscope_dir="${PVRGPU_BENCHSCOPE_ROOT:-$(cd "${project_dir}/.." && pwd)/BenchScope}"
builder="${PVRGPU_MESA_POC_BUILDER:-${benchscope_dir}/scripts/build-mesa-counter.sh}"
poc_root="${PVRGPU_MESA_POC_ROOT:-${PVRGPU_WORK_ROOT%/}/mesa-poc}"
poc_patch="${project_dir}/patches/mesa-26.2.1-pvrgpu-rdc-poc.patch"

if [[ ! -x "${builder}" ]]; then
    echo "BenchScope Mesa builder was not found: ${builder}" >&2
    echo "Set PVRGPU_MESA_POC_BUILDER to the build-mesa-counter.sh path." >&2
    exit 1
fi
if [[ ! -f "${poc_patch}" ]]; then
    echo "Mesa/POC patch was not found: ${poc_patch}" >&2
    exit 1
fi

mkdir -p "${PVRGPU_WORK_ROOT}" "${PVRGPU_TMP_ROOT}"
export TMPDIR="${PVRGPU_TMP_ROOT}"
MESA_COUNTER_ROOT="${poc_root}" \
MESA_EXTRA_PATCH="${poc_patch}" \
MESA_APPLY_COUNTER_PATCHES=false \
MESA_GALLIUM_DRIVERS=llvmpipe,zink \
MESA_BUILD_JOBS="${MESA_BUILD_JOBS:-4}" \
    "${builder}"

expected_prefix="${poc_root}/install"
for required_runtime in \
    "${expected_prefix}/lib/libEGL.dylib" \
    "${expected_prefix}/lib/libGLESv2.dylib" \
    "${expected_prefix}/lib/dri/swrast_dri.dylib"
do
    if [[ ! -f "${required_runtime}" ]]; then
        echo "Mesa/POC install is incomplete: ${required_runtime}" >&2
        exit 1
    fi
done
echo "PvrGPU Mesa/POC prefix ready: ${expected_prefix}"
