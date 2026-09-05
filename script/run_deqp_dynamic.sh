#!/usr/bin/env bash
# ==============================================================================
# PvrGPU - dEQP dynamic-link runner
# ------------------------------------------------------------------------------
# Runs a stock upstream dEQP module binary against the ALREADY-BUILT PCO driver
# and the ALREADY-BUILT PvrGPU SystemC model. Nothing is compiled, relinked or
# rebuilt here: the three artifacts are joined only at run time.
#
#   deqp-<module>                (dEQP project build - unmodified)
#     |  DYLD_LIBRARY_PATH
#     v
#   libEGL.1.dylib / libGLESv2.2.dylib   (Mesa prefix that carries the PCO
#     |  GALLIUM_DRIVER=pvrgpu            driver: gallium pvrgpu + PCO compiler)
#     v
#   dlopen(PVRGPU_SYSTEMC_API_LIB)
#     |
#     v
#   libpvrgpu_systemc_bridge.dylib       (PvrGPU build - SystemC model)
#
# This is the dynamic counterpart of `pvrgpu-deqp`, which folds the same three
# pieces into one statically linked executable. Use this script when the dEQP
# binary and the driver were built separately and must stay separate.
#
# Contract (same as pvrgpu-deqp / tools/deqp_live_ui.py):
#   * one exact case per process - the SystemC bridge defers simulation to
#     process exit and keeps only its latest submitted command;
#   * wildcards and comma-separated case lists are rejected;
#   * a --caselist file is expanded here and run one fresh process per case.
#
# Usage:
#   ./script/run_deqp_dynamic.sh --check
#   ./script/run_deqp_dynamic.sh -c dEQP-GLES2.functional.prerequisite.clear_color
#   ./script/run_deqp_dynamic.sh -c dEQP-EGL.functional.create_context.rgb565_no_depth_no_stencil
#   ./script/run_deqp_dynamic.sh --caselist my_cases.txt --output-dir /tmp/run1
#   ./script/run_deqp_dynamic.sh -c <case> --verify-link
#   ./script/run_deqp_dynamic.sh --module egl --discover --caselist-out /tmp/egl.txt
#
# The desktop front end for this script is script/deqp_dynamic_ui.py.
#
# Paths come from config/local.env unless overridden by flags or environment:
#   PVRGPU_MESA_PVRGPU_PREFIX   Mesa prefix containing the PCO/pvrgpu driver
#   PVRGPU_SYSTEMC_API_LIB      libpvrgpu_systemc_bridge.dylib
#   PVRGPU_BUILD_DIR            PvrGPU build directory (bridge fallback)
#   PVRGPU_OUTPUT_ROOT          run output root
#   PVRGPU_DEQP_PROJECT_DIR     dEQP project (default: <repo>/../dEQP)
#   PVRGPU_DEQP_BUILD_DIR       dEQP CMake build dir (default: from that
#                               project's out/deqp-build-<arch>.env)
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() { echo "run_deqp_dynamic: $*" >&2; exit 1; }
warn() { echo "run_deqp_dynamic: warning: $*" >&2; }
info() { echo "$*"; }

# ------------------------------------------------------------------------------
# Defaults and command line
# ------------------------------------------------------------------------------
DEFAULT_CASE="dEQP-GLES2.functional.prerequisite.clear_color"

opt_case=""
opt_caselist=""
opt_module="auto"
opt_deqp_binary="${DEQP_BINARY:-}"
opt_deqp_build_dir="${PVRGPU_DEQP_BUILD_DIR:-}"
opt_deqp_project="${PVRGPU_DEQP_PROJECT_DIR:-}"
opt_archive_dir=""
opt_mesa_prefix="${PVRGPU_MESA_PVRGPU_PREFIX:-}"
opt_systemc_lib="${PVRGPU_SYSTEMC_API_LIB:-}"
opt_output_dir="${OUTPUT_DIR:-}"
opt_gl_config="rgba8888d24s8ms0"
opt_surface_type="pbuffer"
opt_width="256"
opt_height="256"
opt_log_images="disable"
opt_timeout="0"
opt_verify_link=0
opt_check_only=0
opt_print_env=0
opt_keep_going=0
opt_emit_events=0
opt_discover=0
opt_caselist_out=""
extra_args=()

usage() {
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  -c, --case CASE            one exact dEQP case (default: dEQP-GLES2.functional.prerequisite.clear_color)
      --caselist FILE        file of exact cases, one per line ('#' comments allowed)
      --module NAME          auto|egl|gles2|gles3|gles31 (default: auto, from the case prefix)
      --deqp-binary PATH     explicit dEQP module binary
      --deqp-build-dir DIR   dEQP CMake build dir (expects modules/<module>/deqp-<module>)
      --deqp-project DIR     dEQP project dir holding out/deqp-build-<arch>.env
      --archive-dir DIR      dEQP data archive dir (default: the binary's own directory)
      --mesa-prefix DIR      Mesa prefix with the PCO/pvrgpu driver
      --systemc-lib PATH     PvrGPU SystemC bridge dylib
      --output-dir DIR       run output directory
      --gl-config NAME       --deqp-gl-config-name value ('' to omit)
      --surface-type TYPE    pbuffer|fbo|window (default: pbuffer)
      --size WxH             surface size (default: 256x256)
      --log-images MODE      enable|disable (default: disable)
      --timeout SECONDS      per-case wall clock limit (0 = none)
      --verify-link          ask dyld to report every loaded library and record
                             which libEGL/libGLESv2/driver/bridge were used
      --keep-going           with --caselist, continue after a failing case
      --check                preflight only: resolve and validate, run nothing
      --print-env            print the exported runtime environment
      --emit-events          also print machine-readable "PVRGPU_DYN {json}" lines
      --discover             list this module's cases through the same dynamic
                             wiring instead of running anything (needs --module)
      --caselist-out FILE    where --discover writes the discovered case names
  -h, --help                 this help
  -- ARGS...                 extra arguments passed straight to the dEQP binary
EOF
}

while (($# > 0)); do
    case "$1" in
        -c|--case)          [[ $# -ge 2 ]] || die "missing value for $1"; opt_case="$2"; shift 2 ;;
        --caselist)         [[ $# -ge 2 ]] || die "missing value for $1"; opt_caselist="$2"; shift 2 ;;
        --module)           [[ $# -ge 2 ]] || die "missing value for $1"; opt_module="$2"; shift 2 ;;
        --deqp-binary)      [[ $# -ge 2 ]] || die "missing value for $1"; opt_deqp_binary="$2"; shift 2 ;;
        --deqp-build-dir)   [[ $# -ge 2 ]] || die "missing value for $1"; opt_deqp_build_dir="$2"; shift 2 ;;
        --deqp-project)     [[ $# -ge 2 ]] || die "missing value for $1"; opt_deqp_project="$2"; shift 2 ;;
        --archive-dir)      [[ $# -ge 2 ]] || die "missing value for $1"; opt_archive_dir="$2"; shift 2 ;;
        --mesa-prefix)      [[ $# -ge 2 ]] || die "missing value for $1"; opt_mesa_prefix="$2"; shift 2 ;;
        --systemc-lib)      [[ $# -ge 2 ]] || die "missing value for $1"; opt_systemc_lib="$2"; shift 2 ;;
        --output-dir)       [[ $# -ge 2 ]] || die "missing value for $1"; opt_output_dir="$2"; shift 2 ;;
        --gl-config)        [[ $# -ge 2 ]] || die "missing value for $1"; opt_gl_config="$2"; shift 2 ;;
        --surface-type)     [[ $# -ge 2 ]] || die "missing value for $1"; opt_surface_type="$2"; shift 2 ;;
        --log-images)       [[ $# -ge 2 ]] || die "missing value for $1"; opt_log_images="$2"; shift 2 ;;
        --timeout)          [[ $# -ge 2 ]] || die "missing value for $1"; opt_timeout="$2"; shift 2 ;;
        --caselist-out)     [[ $# -ge 2 ]] || die "missing value for $1"; opt_caselist_out="$2"; shift 2 ;;
        --size)
            [[ $# -ge 2 ]] || die "missing value for $1"
            [[ "$2" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]] || die "size must be WIDTHxHEIGHT"
            opt_width="${BASH_REMATCH[1]}"; opt_height="${BASH_REMATCH[2]}"; shift 2 ;;
        --verify-link)      opt_verify_link=1; shift ;;
        --keep-going)       opt_keep_going=1; shift ;;
        --check)            opt_check_only=1; shift ;;
        --print-env)        opt_print_env=1; shift ;;
        --emit-events)      opt_emit_events=1; shift ;;
        --discover)         opt_discover=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; extra_args=("$@"); break ;;
        *)                  usage >&2; die "unknown argument: $1" ;;
    esac
done

# ------------------------------------------------------------------------------
# Machine-readable events for the desktop UI (script/deqp_dynamic_ui.py)
# ------------------------------------------------------------------------------
json_string() {
    local text="$1"
    text="${text//\\/\\\\}"
    text="${text//\"/\\\"}"
    text="${text//$'\n'/ }"
    text="${text//$'\r'/ }"
    text="${text//$'\t'/ }"
    printf '"%s"' "${text}"
}

emit_event() {
    ((opt_emit_events)) || return 0
    local body="$1"
    printf 'PVRGPU_DYN {%s}\n' "${body}"
}

now_ms() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import time; print(int(time.time() * 1000))'
    else
        printf '%s000' "$(date +%s)"
    fi
}

# ------------------------------------------------------------------------------
# config/local.env - flags and pre-existing environment win over the file
# ------------------------------------------------------------------------------
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    saved_mesa_prefix="${opt_mesa_prefix}"
    saved_systemc_lib="${opt_systemc_lib}"
    saved_build_dir="${PVRGPU_BUILD_DIR:-}"
    saved_output_root="${PVRGPU_OUTPUT_ROOT:-}"
    set -a
    # shellcheck disable=SC1091
    source "${REPO_DIR}/config/local.env"
    set +a
    [[ -n "${saved_mesa_prefix}" ]] && opt_mesa_prefix="${saved_mesa_prefix}"
    [[ -n "${saved_systemc_lib}" ]] && opt_systemc_lib="${saved_systemc_lib}"
    [[ -n "${saved_build_dir}" ]] && PVRGPU_BUILD_DIR="${saved_build_dir}"
    [[ -n "${saved_output_root}" ]] && PVRGPU_OUTPUT_ROOT="${saved_output_root}"
    [[ -z "${opt_mesa_prefix}" ]] && opt_mesa_prefix="${PVRGPU_MESA_PVRGPU_PREFIX:-}"
    [[ -z "${opt_systemc_lib}" ]] && opt_systemc_lib="${PVRGPU_SYSTEMC_API_LIB:-}"
fi

PVRGPU_BUILD_DIR="${PVRGPU_BUILD_DIR:-${REPO_DIR}/build}"
[[ -z "${opt_systemc_lib}" ]] && opt_systemc_lib="${PVRGPU_BUILD_DIR}/lib/libpvrgpu_systemc_bridge.dylib"

# ------------------------------------------------------------------------------
# Case list
# ------------------------------------------------------------------------------
validate_case() {
    local name="$1"
    [[ -n "${name}" ]] || die "empty case name"
    case "${name}" in
        dEQP-EGL.*|dEQP-GLES2.*|dEQP-GLES3.*|dEQP-GLES31.*) ;;
        *) die "case must start with dEQP-EGL., dEQP-GLES2., dEQP-GLES3. or dEQP-GLES31.: ${name}" ;;
    esac
    case "${name}" in
        *'*'*|*'?'*|*','*|*'{'*|*'}'*)
            die "wildcards and case lists are rejected; the SystemC bridge keeps only one command per process: ${name}" ;;
    esac
}

module_for_case() {
    case "$1" in
        dEQP-EGL.*)    printf 'egl' ;;
        dEQP-GLES2.*)  printf 'gles2' ;;
        dEQP-GLES3.*)  printf 'gles3' ;;
        dEQP-GLES31.*) printf 'gles31' ;;
    esac
}

suite_for_module() {
    case "$1" in
        egl)    printf 'dEQP-EGL' ;;
        gles2)  printf 'dEQP-GLES2' ;;
        gles3)  printf 'dEQP-GLES3' ;;
        gles31) printf 'dEQP-GLES31' ;;
    esac
}

cases=()
if ((opt_discover)); then
    [[ "${opt_module}" != "auto" ]] || die "--discover needs an explicit --module (egl|gles2|gles3|gles31)"
    [[ -n "${opt_caselist_out}" ]] || die "--discover needs --caselist-out FILE"
elif [[ -n "${opt_caselist}" ]]; then
    [[ -n "${opt_case}" ]] && die "use either --case or --caselist, not both"
    [[ -f "${opt_caselist}" ]] || die "caselist not found: ${opt_caselist}"
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(printf '%s' "${line}" | tr -d '[:space:]')"
        [[ -z "${line}" ]] && continue
        validate_case "${line}"
        cases+=("${line}")
    done < "${opt_caselist}"
    ((${#cases[@]} > 0)) || die "caselist contains no cases: ${opt_caselist}"
else
    opt_case="${opt_case:-${DEQP_CASE:-${DEFAULT_CASE}}}"
    validate_case "${opt_case}"
    cases=("${opt_case}")
fi

if [[ "${opt_module}" == "auto" ]]; then
    first_module="$(module_for_case "${cases[0]}")"
    for one_case in "${cases[@]}"; do
        [[ "$(module_for_case "${one_case}")" == "${first_module}" ]] ||
            die "a caselist must stay inside one dEQP module; found ${first_module} and $(module_for_case "${one_case}")"
    done
    module="${first_module}"
else
    case "${opt_module}" in
        egl|gles2|gles3|gles31) module="${opt_module}" ;;
        *) die "unknown module: ${opt_module}" ;;
    esac
fi

# ------------------------------------------------------------------------------
# Locate the already-built dEQP module binary
# ------------------------------------------------------------------------------
host_arch="$(uname -m)"

if [[ -z "${opt_deqp_build_dir}" ]]; then
    if [[ -z "${opt_deqp_project}" ]]; then
        for candidate in "${REPO_DIR}/../dEQP" "${REPO_DIR}/../deqp" "${HOME}/Codex/dEQP"; do
            if [[ -d "${candidate}/scripts" ]]; then
                opt_deqp_project="$(cd "${candidate}" && pwd)"
                break
            fi
        done
    fi
    if [[ -n "${opt_deqp_project}" ]]; then
        metadata="${opt_deqp_project}/out/deqp-build-${host_arch}.env"
        if [[ -f "${metadata}" ]]; then
            # shellcheck disable=SC1090
            source "${metadata}"
            opt_deqp_build_dir="${DEQP_METADATA_BUILD_DIR:-}"
        else
            warn "no dEQP build metadata at ${metadata}"
        fi
    fi
fi

deqp_search_roots=()
[[ -n "${opt_deqp_build_dir}" ]] && deqp_search_roots+=("${opt_deqp_build_dir}")
if [[ -n "${PVRGPU_DEQP_SOURCE_DIR:-}" ]]; then
    deqp_src="${PVRGPU_DEQP_SOURCE_DIR%/}"
    deqp_search_roots+=("${deqp_src}/build" "$(dirname "${deqp_src}")/build" "$(dirname "${deqp_src}")")
fi
[[ -n "${PVRGPU_WORK_ROOT:-}" ]] && deqp_search_roots+=(
    "${PVRGPU_WORK_ROOT}/build/deqp-mesa/build"
    "${PVRGPU_WORK_ROOT}/build/deqp/build"
)

if [[ -z "${opt_deqp_binary}" ]]; then
    for root in "${deqp_search_roots[@]}"; do
        [[ -d "${root}" ]] || continue
        for candidate in "${root}/modules/${module}/deqp-${module}" \
                         "${root}/external/openglcts/modules/deqp-${module}"; do
            if [[ -x "${candidate}" ]]; then
                opt_deqp_binary="${candidate}"
                break 2
            fi
        done
    done
fi

if [[ -z "${opt_deqp_binary}" ]]; then
    echo "run_deqp_dynamic: no deqp-${module} binary found. Looked for:" >&2
    for root in "${deqp_search_roots[@]}"; do
        echo "  ${root}/modules/${module}/deqp-${module}" >&2
    done
    existing="$(for root in "${deqp_search_roots[@]}"; do
                    ls -d "${root}"/modules/*/deqp-* 2>/dev/null
                done)"
    if [[ -n "${existing}" ]]; then
        echo "  module binaries that do exist:" >&2
        printf '%s\n' "${existing}" | sed 's/^/    /' >&2
    fi
    die "pass --deqp-binary PATH, or --deqp-build-dir DIR, or build that module in the dEQP project (SELECTED_BUILD_TARGETS=deqp-${module})"
fi

[[ -x "${opt_deqp_binary}" ]] || die "dEQP binary is not executable: ${opt_deqp_binary}"
deqp_binary="$(cd "$(dirname "${opt_deqp_binary}")" && pwd)/$(basename "${opt_deqp_binary}")"
archive_dir="${opt_archive_dir:-$(dirname "${deqp_binary}")}"
[[ -d "${archive_dir}" ]] || die "dEQP archive dir not found: ${archive_dir}"

# ------------------------------------------------------------------------------
# Validate the two already-built PvrGPU artifacts
# ------------------------------------------------------------------------------
[[ -n "${opt_mesa_prefix}" ]] || die "Mesa PCO/pvrgpu prefix is unset. Set PVRGPU_MESA_PVRGPU_PREFIX in config/local.env or pass --mesa-prefix."
[[ -d "${opt_mesa_prefix}" ]] || die "Mesa PCO/pvrgpu prefix not found: ${opt_mesa_prefix}"
mesa_prefix="$(cd "${opt_mesa_prefix}" && pwd)"
mesa_egl="${mesa_prefix}/lib/libEGL.1.dylib"
mesa_gles="${mesa_prefix}/lib/libGLESv2.2.dylib"
[[ -f "${mesa_egl}" ]]  || die "PCO driver build has no libEGL.1.dylib: ${mesa_egl}"
[[ -f "${mesa_gles}" ]] || die "PCO driver build has no libGLESv2.2.dylib: ${mesa_gles}"

[[ -f "${opt_systemc_lib}" ]] || die "PvrGPU SystemC bridge not found: ${opt_systemc_lib}
Build it first:  cmake --build \"${PVRGPU_BUILD_DIR}\" --target pvrgpu_systemc_bridge"
systemc_lib="$(cd "$(dirname "${opt_systemc_lib}")" && pwd)/$(basename "${opt_systemc_lib}")"

# The whole point of this script: the runner must reach EGL/GLES dynamically.
if command -v otool >/dev/null 2>&1; then
    linked_gl="$(otool -L "${deqp_binary}" 2>/dev/null | grep -E 'libEGL|libGLESv2' || true)"
    if [[ -z "${linked_gl}" ]]; then
        warn "${deqp_binary} does not list libEGL/libGLESv2 as dynamic dependencies.
         A statically linked runner (such as pvrgpu-deqp) cannot be redirected by
         DYLD_LIBRARY_PATH; use the dEQP project's own deqp-<module> binary."
    fi
fi

# Architecture agreement across all three artifacts.
if command -v lipo >/dev/null 2>&1; then
    arch_of() { lipo -archs "$1" 2>/dev/null | tr ' ' '\n' | sort -u | tr '\n' ' '; }
    runner_archs="$(arch_of "${deqp_binary}")"
    mesa_archs="$(arch_of "${mesa_egl}")"
    bridge_archs="$(arch_of "${systemc_lib}")"
    for pair in "dEQP runner:${runner_archs}" "PCO driver:${mesa_archs}" "SystemC bridge:${bridge_archs}"; do
        label="${pair%%:*}"; archs=" ${pair#*:}"
        [[ "${archs}" == *" ${host_arch} "* ]] ||
            die "${label} is [${archs# }] but this host is ${host_arch}. Rebuild that artifact for ${host_arch} or run from the matching shell."
    done
fi

# Soft check: is the pvrgpu gallium driver actually inside this Mesa build?
if command -v strings >/dev/null 2>&1; then
    driver_hit=0
    for gallium_lib in "${mesa_prefix}"/lib/libgallium*.dylib "${mesa_prefix}"/lib/dri/*.dylib "${mesa_gles}"; do
        [[ -f "${gallium_lib}" ]] || continue
        if strings -a "${gallium_lib}" 2>/dev/null | grep -q 'pvrgpu'; then
            driver_hit=1
            break
        fi
    done
    ((driver_hit)) || warn "no 'pvrgpu' symbol text found under ${mesa_prefix}/lib.
         This Mesa prefix may not carry the PCO/pvrgpu Gallium driver; GALLIUM_DRIVER=pvrgpu would then fall back or fail."
fi

# ------------------------------------------------------------------------------
# Output layout
# ------------------------------------------------------------------------------
run_stamp="$(date +%Y%m%d_%H%M%S)"
output_root="${opt_output_dir:-${PVRGPU_OUTPUT_ROOT:-${REPO_DIR}/outputs}/deqp_dynamic/${run_stamp}}"
mkdir -p "${output_root}" || die "cannot create output directory: ${output_root}"
output_root="$(cd "${output_root}" && pwd)"

case_slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# ------------------------------------------------------------------------------
# Runtime environment - the dynamic wiring itself
# ------------------------------------------------------------------------------
export DYLD_LIBRARY_PATH="${mesa_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export DYLD_FALLBACK_LIBRARY_PATH="${mesa_prefix}/lib${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"
export LIBGL_DRIVERS_PATH="${mesa_prefix}/lib/dri"
export EGL_PLATFORM="surfaceless"
export GALLIUM_DRIVER="pvrgpu"
export MESA_LOADER_DRIVER_OVERRIDE="swrast"
export LIBGL_ALWAYS_SOFTWARE="true"
export MESA_SHADER_CACHE_DISABLE="${MESA_SHADER_CACHE_DISABLE:-true}"
export PVRGPU_DEQP_LIVE="1"
export PVRGPU_DEQP_OUTPUT_ROOT="${output_root}"
export PVRGPU_SYSTEMC_API_LIB="${systemc_lib}"

info "dEQP module      : ${module}"
info "dEQP runner      : ${deqp_binary}"
info "Archive dir      : ${archive_dir}"
info "PCO driver (Mesa): ${mesa_prefix}"
info "PvrGPU bridge    : ${systemc_lib}"
info "Host arch        : ${host_arch}"
info "Cases            : ${#cases[@]}"
info "Output           : ${output_root}"

if ((opt_print_env)); then
    info ""
    info "Exported runtime environment:"
    for name in DYLD_LIBRARY_PATH LIBGL_DRIVERS_PATH EGL_PLATFORM GALLIUM_DRIVER \
                MESA_LOADER_DRIVER_OVERRIDE LIBGL_ALWAYS_SOFTWARE \
                MESA_SHADER_CACHE_DISABLE PVRGPU_DEQP_LIVE PVRGPU_SYSTEMC_API_LIB; do
        info "  ${name}=${!name}"
    done
fi

emit_event "$(printf '"event":"resolved","module":%s,"runner":%s,"archive_dir":%s,"mesa_prefix":%s,"systemc_lib":%s,"host_arch":%s,"output_root":%s' \
    "$(json_string "${module}")" "$(json_string "${deqp_binary}")" \
    "$(json_string "${archive_dir}")" "$(json_string "${mesa_prefix}")" \
    "$(json_string "${systemc_lib}")" "$(json_string "${host_arch}")" \
    "$(json_string "${output_root}")")"

if ((opt_check_only)); then
    info ""
    info "Preflight passed. Nothing was executed (--check)."
    emit_event '"event":"check_ok"'
    exit 0
fi

# ------------------------------------------------------------------------------
# Discovery - list the module's cases through the same dynamic wiring
# ------------------------------------------------------------------------------
if ((opt_discover)); then
    suite="$(suite_for_module "${module}")"
    discovery_dir="${output_root}/discovery"
    mkdir -p "${discovery_dir}"
    export_file="${discovery_dir}/all-cases.txt"
    rm -f -- "${export_file}"

    # Listing builds the case hierarchy; it does not execute GL tests. The
    # override only lets the ES3/ES3.1 packages enumerate while the live EGL
    # config is still ES2-only.
    case "${module}" in
        gles3)  export MESA_GLES_VERSION_OVERRIDE="3.0" ;;
        gles31) export MESA_GLES_VERSION_OVERRIDE="3.1" ;;
        *)      unset MESA_GLES_VERSION_OVERRIDE ;;
    esac

    info ""
    info "Discovering ${suite} cases ..."
    emit_event "$(printf '"event":"discovery_started","module":%s,"suite":%s' \
        "$(json_string "${module}")" "$(json_string "${suite}")")"

    (
        cd "${archive_dir}" || exit 127
        "${deqp_binary}" \
            "--deqp-case=${suite}.*" \
            "--deqp-archive-dir=${archive_dir}" \
            "--deqp-runmode=txt-caselist" \
            "--deqp-caselist-export-file=${export_file}" \
            "--deqp-log-filename=${discovery_dir}/discovery.qpa" \
            "--deqp-surface-type=${opt_surface_type}" \
            --deqp-log-images=disable
    ) > "${discovery_dir}/discovery.log" 2>&1
    discovery_exit=$?

    if [[ ! -f "${export_file}" ]]; then
        emit_event "$(printf '"event":"discovery_failed","exit_code":%d,"log":%s' \
            "${discovery_exit}" "$(json_string "${discovery_dir}/discovery.log")")"
        die "caselist discovery produced no export (exit ${discovery_exit}); see ${discovery_dir}/discovery.log"
    fi

    mkdir -p "$(dirname "${opt_caselist_out}")"
    grep '^TEST:' "${export_file}" | sed 's/^TEST:[[:space:]]*//' > "${opt_caselist_out}"
    discovered="$(wc -l < "${opt_caselist_out}" | tr -d '[:space:]')"

    info "Discovered ${discovered} case(s) -> ${opt_caselist_out}"
    emit_event "$(printf '"event":"discovery_finished","discovered":%s,"caselist":%s,"exit_code":%d' \
        "${discovered:-0}" "$(json_string "${opt_caselist_out}")" "${discovery_exit}")"
    exit 0
fi

# ------------------------------------------------------------------------------
# Per-case execution - one fresh process per case
# ------------------------------------------------------------------------------
timeout_cmd=()
if [[ "${opt_timeout}" != "0" ]]; then
    if command -v timeout >/dev/null 2>&1; then
        timeout_cmd=(timeout "${opt_timeout}")
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_cmd=(gtimeout "${opt_timeout}")
    else
        warn "neither timeout nor gtimeout is available; --timeout is ignored"
    fi
fi

# One fresh process per case, from the module's own archive directory.
run_one_case() {
    (
        cd "${archive_dir}" || exit 127
        if ((${#timeout_cmd[@]} > 0)); then
            exec "${timeout_cmd[@]}" "${deqp_binary}" "${deqp_args[@]}"
        else
            exec "${deqp_binary}" "${deqp_args[@]}"
        fi
    )
}

summary_tsv="${output_root}/summary.tsv"
printf 'case\tstatus\texit_code\tduration_ms\tcase_dir\n' > "${summary_tsv}"

emit_event "$(printf '"event":"run_start","total":%d,"module":%s,"output_root":%s,"summary":%s' \
    "${#cases[@]}" "$(json_string "${module}")" "$(json_string "${output_root}")" \
    "$(json_string "${summary_tsv}")")"

failures=0
completed=0
run_start_ms="$(now_ms)"
for one_case in "${cases[@]}"; do
    index=$((completed + 1))
    slug="$(case_slug "${one_case}")"
    case_dir="${output_root}/cases/${slug}"
    mkdir -p "${case_dir}/systemc"

    qpa_path="${case_dir}/results.qpa"
    log_path="${case_dir}/run.log"
    rm -f -- "${qpa_path}" "${log_path}"

    # Same artifact contract as pvrgpu-deqp's SetArtifactEnvironment().
    export PVRGPU_RDC_CASE_NAME="${one_case}"
    export PVRGPU_DRIVER_COMMAND_OUT="${case_dir}/driver-command.txt"
    export PVRGPU_DRIVER_COUNTER_OUT="${case_dir}/driver-counter.txt"
    export PVRGPU_SYSTEMC_JSONL_OUT="${case_dir}/systemc.jsonl"
    export PVRGPU_SYSTEMC_STDERR_OUT="${case_dir}/systemc.stderr.log"
    export PVRGPU_SYSTEMC_OUTDIR="${case_dir}/systemc"

    deqp_args=(
        "--deqp-case=${one_case}"
        "--deqp-archive-dir=${archive_dir}"
        "--deqp-log-filename=${qpa_path}"
        "--deqp-surface-type=${opt_surface_type}"
        "--deqp-surface-width=${opt_width}"
        "--deqp-surface-height=${opt_height}"
        "--deqp-log-images=${opt_log_images}"
    )
    [[ -n "${opt_gl_config}" ]] && deqp_args+=("--deqp-gl-config-name=${opt_gl_config}")
    ((${#extra_args[@]} > 0)) && deqp_args+=("${extra_args[@]}")

    {
        echo "schema=pvrgpu.deqp-dynamic-run.v1"
        echo "backend=pvrgpu"
        echo "link=dynamic"
        echo "case=${one_case}"
        echo "module=${module}"
        echo "runner=${deqp_binary}"
        echo "archive_dir=${archive_dir}"
        echo "mesa_pco_prefix=${mesa_prefix}"
        echo "systemc_api_lib=${systemc_lib}"
        echo "host_arch=${host_arch}"
        echo "output_dir=${case_dir}"
    } > "${case_dir}/run.txt"

    info ""
    info "==> [${index}/${#cases[@]}] ${one_case}"
    emit_event "$(printf '"event":"case_start","index":%d,"total":%d,"case":%s,"case_dir":%s' \
        "${index}" "${#cases[@]}" "$(json_string "${one_case}")" "$(json_string "${case_dir}")")"

    if ((opt_verify_link)); then
        export DYLD_PRINT_LIBRARIES=1
    else
        unset DYLD_PRINT_LIBRARIES
    fi

    case_start_ms="$(now_ms)"
    if ((opt_verify_link)); then
        run_one_case 2>&1 | tee "${log_path}" | grep -v '^dyld\[' || true
    else
        run_one_case 2>&1 | tee "${log_path}"
    fi
    exit_code=${PIPESTATUS[0]}
    case_end_ms="$(now_ms)"
    duration_ms=$((case_end_ms - case_start_ms))

    if ((opt_verify_link)); then
        grep -E '^dyld\[' "${log_path}" 2>/dev/null \
            | grep -E 'libEGL|libGLESv2|libgallium|/dri/|pvrgpu' \
            > "${case_dir}/link.txt" || true
        if [[ -s "${case_dir}/link.txt" ]]; then
            info "Loaded GL/driver libraries:"
            sed 's/^/  /' "${case_dir}/link.txt"
        fi
    fi

    if [[ -f "${qpa_path}" ]]; then
        status="$(grep -o 'StatusCode="[^"]*"' "${qpa_path}" | head -1 | sed 's/StatusCode="//; s/"//')"
        [[ -n "${status}" ]] || status="NoResult"
    else
        status="NoQpa"
    fi

    printf '%s\t%s\t%d\t%d\t%s\n' "${one_case}" "${status}" "${exit_code}" "${duration_ms}" "${case_dir}" >> "${summary_tsv}"
    info "Result           : ${status} (exit ${exit_code}, ${duration_ms} ms)"
    info "QPA              : ${qpa_path}"
    info "Log              : ${log_path}"

    emit_event "$(printf '"event":"case_end","index":%d,"total":%d,"case":%s,"status":%s,"exit_code":%d,"duration_ms":%d,"case_dir":%s,"qpa":%s,"log":%s' \
        "${index}" "${#cases[@]}" "$(json_string "${one_case}")" "$(json_string "${status}")" \
        "${exit_code}" "${duration_ms}" "$(json_string "${case_dir}")" \
        "$(json_string "${qpa_path}")" "$(json_string "${log_path}")")"

    completed=$((completed + 1))

    case "$(printf '%s' "${status}" | tr '[:upper:]' '[:lower:]')" in
        pass|notsupported|waiver|qualitywarning|compatibilitywarning) case_failed=0 ;;
        *) case_failed=1 ;;
    esac
    ((exit_code != 0)) && case_failed=1

    if ((case_failed)); then
        failures=$((failures + 1))
        if ((opt_keep_going == 0)) && ((${#cases[@]} > 1)); then
            warn "stopping after the first failing case; pass --keep-going to continue"
            break
        fi
    fi
done

run_end_ms="$(now_ms)"
info ""
info "Summary          : ${summary_tsv}"
emit_event "$(printf '"event":"run_end","total":%d,"completed":%d,"failures":%d,"duration_ms":%d,"summary":%s,"output_root":%s' \
    "${#cases[@]}" "${completed}" "${failures}" "$((run_end_ms - run_start_ms))" \
    "$(json_string "${summary_tsv}")" "$(json_string "${output_root}")")"

if ((failures == 0)); then
    info "All ${completed} case(s) passed."
    exit 0
fi
info "${failures} of ${completed} case(s) did not pass."
exit 1
