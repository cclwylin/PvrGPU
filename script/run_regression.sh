#!/usr/bin/env bash
# ==============================================================================
# PvrGPU RDC Test Pattern Batch Regression Script
# ==============================================================================
# Usage:
#   ./script/run_regression.sh                       # Run all patterns (default: 4 jobs)
#   ./script/run_regression.sh --suite GLBench       # Run only GLBench patterns
#   ./script/run_regression.sh --suite glmark2       # Run only glmark2 patterns
#   ./script/run_regression.sh --suite dEQP --limit 50 # Run first 50 dEQP patterns
#   ./script/run_regression.sh --suite GFXBench      # Run only GFXBench patterns
#   ./script/run_regression.sh --skip-passed         # Resume / skip already passed tests
#   ./script/run_regression.sh --list-only           # Just list discovered patterns
#   ./script/run_regression.sh -j 8                  # Run with 8 parallel workers
#   ./script/run_regression.sh --no-build            # Report on whatever is already built
#   ./script/run_regression.sh --allow-stale-mesa    # Run against a Mesa prefix older than the driver
# ==============================================================================
#
# Why this script builds before it runs.
#
# A regression run is three separately built artifacts meeting at run time: the
# SystemC bridge and the `pvrgpu` runner out of PVRGPU_BUILD_DIR, and the
# Gallium driver inside PVRGPU_MESA_PVRGPU_PREFIX, which Mesa's meson builds
# somewhere else entirely.  None of them live in this repository, so editing
# model or driver source and running this script reported on the previous build
# without saying so -- a green result describing code nobody ran.
#
# The CMake half is rebuilt here, which is a no-op when nothing changed.  The
# Mesa half cannot be built from here, so instead a prefix older than the
# driver sources stops the run and says what to rebuild.
#
# Staleness is judged by modification time rather than by git: a rebuild is a
# question about mtimes, and an uncommitted edit counts exactly as much as a
# committed one.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${REPO_DIR}/tools/run_rdc_regression.py"

die() {
    local line
    for line in "$@"; do
        printf 'run_regression.sh: %s\n' "${line}" >&2
    done
    exit 1
}

note() {
    printf 'run_regression.sh: %s\n' "$*"
}

# ------------------------------------------------------------------------------
# Arguments this script handles itself; everything else goes to the runner
# ------------------------------------------------------------------------------
opt_build=1
opt_allow_stale_mesa=0
opt_list_only=0
opt_has_pvrgpu_bin=0
forwarded=()

while (($#)); do
    case "$1" in
        --no-build)          opt_build=0 ;;
        --allow-stale-mesa)  opt_allow_stale_mesa=1 ;;
        --list-only)         opt_list_only=1; forwarded+=("$1") ;;
        --pvrgpu-bin|--pvrgpu-bin=*)
                             opt_has_pvrgpu_bin=1; forwarded+=("$1") ;;
        *)                   forwarded+=("$1") ;;
    esac
    shift
done

# Listing patterns runs nothing, so it needs nothing built.
if ((opt_list_only)); then
    opt_build=0
fi

# Default pattern root
PATTERNS_DIR="${PATTERNS_DIR:-/Users/linwanyi/Downloads/Working/GPU_TestPatterns}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/rdc_regression}"

# Ensure python3 exists
if ! command -v python3 &>/dev/null; then
    die "python3 is not found in PATH"
fi

# ------------------------------------------------------------------------------
# config/local.env - a pre-existing environment wins over the file
# ------------------------------------------------------------------------------
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    saved_build_dir="${PVRGPU_BUILD_DIR:-}"
    saved_mesa_prefix="${PVRGPU_MESA_PVRGPU_PREFIX:-}"
    set -a
    # shellcheck disable=SC1091
    source "${REPO_DIR}/config/local.env"
    set +a
    if [[ -n "${saved_build_dir}" ]]; then
        PVRGPU_BUILD_DIR="${saved_build_dir}"
    fi
    if [[ -n "${saved_mesa_prefix}" ]]; then
        PVRGPU_MESA_PVRGPU_PREFIX="${saved_mesa_prefix}"
    fi
fi
PVRGPU_BUILD_DIR="${PVRGPU_BUILD_DIR:-${REPO_DIR}/build}"

# The newest modification time under the given files/directories, or 0.
newest_mtime() {
    python3 - "$@" <<'PY'
import os
import sys

newest = 0
for root in sys.argv[1:]:
    if os.path.isfile(root):
        try:
            newest = max(newest, int(os.path.getmtime(root)))
        except OSError:
            pass
        continue
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                newest = max(newest, int(os.path.getmtime(os.path.join(dirpath, name))))
            except OSError:
                pass
print(newest)
PY
}

# ------------------------------------------------------------------------------
# The CMake half: the SystemC bridge and the pvrgpu runner
# ------------------------------------------------------------------------------
if ((opt_build)); then
    if [[ ! -f "${PVRGPU_BUILD_DIR}/CMakeCache.txt" ]]; then
        die "No CMake build at ${PVRGPU_BUILD_DIR}." \
            "Configure it first, or set PVRGPU_BUILD_DIR, or pass --no-build:" \
            "  cmake -S \"${REPO_DIR}\" -B \"${PVRGPU_BUILD_DIR}\" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    fi

    # The build directory lives outside the repository, so it is worth being
    # sure it was configured against *this* source tree and not another
    # checkout that happens to sit next to it.
    build_source_dir="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' \
        "${PVRGPU_BUILD_DIR}/CMakeCache.txt" | head -n 1)"
    if [[ -n "${build_source_dir}" && "${build_source_dir}" != "${REPO_DIR}" ]]; then
        die "${PVRGPU_BUILD_DIR} was configured for a different source tree." \
            "  build:  ${build_source_dir}" \
            "  script: ${REPO_DIR}" \
            "Point PVRGPU_BUILD_DIR at the matching build, or pass --no-build."
    fi

    if ! command -v cmake &>/dev/null; then
        die "cmake is not found in PATH; pass --no-build to run the existing build"
    fi

    note "Building ${PVRGPU_BUILD_DIR}"
    if ! cmake --build "${PVRGPU_BUILD_DIR}"; then
        die "build failed; the regression would have reported on the previous one"
    fi
fi

# ------------------------------------------------------------------------------
# The Mesa half: the Gallium driver, which this script cannot build
# ------------------------------------------------------------------------------
if ((!opt_list_only)) && [[ -n "${PVRGPU_MESA_PVRGPU_PREFIX:-}" ]] &&
   [[ -d "${PVRGPU_MESA_PVRGPU_PREFIX}" ]]; then
    driver_mtime="$(newest_mtime "${REPO_DIR}/src/gallium/drivers/pvrgpu")"
    mesa_mtime="$(newest_mtime "${PVRGPU_MESA_PVRGPU_PREFIX}")"
    if ((driver_mtime > mesa_mtime)); then
        stale_message=(
            "The Gallium driver sources are newer than the installed Mesa prefix."
            "  sources: ${REPO_DIR}/src/gallium/drivers/pvrgpu"
            "  prefix:  ${PVRGPU_MESA_PVRGPU_PREFIX}"
            "Rebuild Mesa into that prefix, or pass --allow-stale-mesa to run anyway."
        )
        if ((opt_allow_stale_mesa)); then
            note "WARNING: ${stale_message[0]}"
            note "         The driver being tested is the one already installed."
        else
            die "${stale_message[@]}"
        fi
    fi
fi

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
# Name the runner explicitly.  Left to itself the discovery picks the first
# executable off a hard-coded candidate list, which need not be the build this
# script just brought up to date.
if ((!opt_has_pvrgpu_bin)) && [[ -x "${PVRGPU_BUILD_DIR}/bin/pvrgpu" ]]; then
    forwarded+=(--pvrgpu-bin "${PVRGPU_BUILD_DIR}/bin/pvrgpu")
fi

# `${array[@]}` on an empty array is an unbound variable under `set -u` in the
# bash macOS still ships, so expand it only when it has something in it.
exec python3 "${PYTHON_SCRIPT}" \
    --pattern-dir "${PATTERNS_DIR}" \
    --out-dir "${OUTPUT_DIR}" \
    ${forwarded[@]+"${forwarded[@]}"}
