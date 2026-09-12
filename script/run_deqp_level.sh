#!/usr/bin/env bash
# Run the shared dEQP L1/L2/L3/L4 plan; no argument means --1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Help must work without a configured GPU runtime.
for argument in "$@"; do
    case "${argument}" in
        -h|--help)
            exec "${PVRGPU_DEQP_LEVEL_PYTHON:-python3}" \
                "${SCRIPT_DIR}/deqp_level_runner.py" --help ;;
    esac
done

# Match the dynamic runner's precedence: flags > environment > local.env.
saved_names=()
saved_values=()
for variable in PVRGPU_WORK_ROOT PVRGPU_BUILD_DIR PVRGPU_OUTPUT_ROOT \
    PVRGPU_MESA_PVRGPU_PREFIX PVRGPU_SYSTEMC_API_LIB PVRGPU_DEQP_BUILD_DIR \
    PVRGPU_DEQP_PROJECT_DIR PVRGPU_DEQP_SOURCE_DIR PVRGPU_MODEL_MEMORY_MODE \
    PVRGPU_TEXTURE_LOD_MODE \
    PVRGPU_DEQP_LEVEL_PYTHON; do
    if declare -p "${variable}" >/dev/null 2>&1; then
        saved_names+=("${variable}")
        saved_values+=("${!variable}")
    fi
done
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_DIR}/config/local.env"
    set +a
fi
for ((index=0; index<${#saved_names[@]}; index++)); do
    export "${saved_names[index]}=${saved_values[index]}"
done

exec "${PVRGPU_DEQP_LEVEL_PYTHON:-python3}" \
    "${SCRIPT_DIR}/deqp_level_runner.py" "$@"
