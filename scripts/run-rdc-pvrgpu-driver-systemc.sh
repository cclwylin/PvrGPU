#!/usr/bin/env bash
# Replay one RDC through RenderDoc + Mesa/Gallium pvrgpu, then run the emitted
# driver command through the PvrGPU SystemC model.
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
Usage: ./scripts/run-rdc-pvrgpu-driver-systemc.sh \
         --rdc FILE --case CASE --width W --height H --outdir DIR

Replay one RDC through:

  RenderDoc player -> Mesa/Gallium pvrgpu -> driver-command -> PvrGPU SystemC

The runner emits pvrgpu.counter.v1 JSONL to stdout for the directory counter
comparer.  Player logs, driver-command.txt, and driver-counter.txt are stored
beside DIR.

Environment overrides:
  PVRGPU_MESA_PVRGPU_PREFIX        Mesa install prefix with pvrgpu Gallium
  PVRGPU_RENDERDOC_MESA_ROOT       RenderDoc player/runtime root
  PVRGPU_MODEL_STUB                pvrgpu-model-stub executable
  PVRGPU_MESA_GLES_VERSION_OVERRIDE  GLES replay gate, default 3.0
EOF
}

rdc_path=""
case_name=""
width=""
height=""
outdir=""
while (($# > 0)); do
    case "$1" in
        --rdc)
            (($# >= 2)) || { echo "--rdc requires a value" >&2; exit 2; }
            rdc_path="$2"
            shift 2
            ;;
        --case)
            (($# >= 2)) || { echo "--case requires a value" >&2; exit 2; }
            case_name="$2"
            shift 2
            ;;
        --width)
            (($# >= 2)) || { echo "--width requires a value" >&2; exit 2; }
            width="$2"
            shift 2
            ;;
        --height)
            (($# >= 2)) || { echo "--height requires a value" >&2; exit 2; }
            height="$2"
            shift 2
            ;;
        --outdir)
            (($# >= 2)) || { echo "--outdir requires a value" >&2; exit 2; }
            outdir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${rdc_path}" || -z "${case_name}" || -z "${width}" ||
      -z "${height}" || -z "${outdir}" ]]; then
    echo "--rdc, --case, --width, --height, and --outdir are all required" >&2
    usage >&2
    exit 2
fi
if [[ ! "${case_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Unsafe case name: ${case_name}" >&2
    exit 2
fi
if [[ ! "${width}" =~ ^[1-9][0-9]*$ || ! "${height}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--width and --height must be positive integers" >&2
    exit 2
fi
if [[ ! -f "${rdc_path}" ]]; then
    echo "RDC is missing or is not a regular file: ${rdc_path}" >&2
    exit 1
fi
if [[ -e "${outdir}" && ! -d "${outdir}" ]]; then
    echo "PvrGPU output path is not a directory: ${outdir}" >&2
    exit 1
fi

pvrgpu_work_root="${PVRGPU_WORK_ROOT%/}"
working_root="$(dirname "${pvrgpu_work_root}")"
mesa_prefix="${PVRGPU_MESA_PVRGPU_PREFIX:-${pvrgpu_work_root}/tmp/pvrgpu-mesa-install}"
renderdoc_root="${PVRGPU_RENDERDOC_MESA_ROOT:-${working_root}/build/renderdoc-mesa}"
player="${renderdoc_root}/bin/renderdoc-mesa-player"
model="${PVRGPU_MODEL_STUB}"
gles_override="${PVRGPU_MESA_GLES_VERSION_OVERRIDE:-3.0}"

for required_runtime in \
    "${mesa_prefix}/lib/libEGL.dylib" \
    "${mesa_prefix}/lib/libGLESv2.dylib" \
    "${player}"
do
    if [[ ! -f "${required_runtime}" ]]; then
        echo "Required runtime is missing: ${required_runtime}" >&2
        exit 1
    fi
done
if [[ ! -x "${player}" ]]; then
    echo "RenderDoc player is not executable: ${player}" >&2
    exit 1
fi
if [[ ! -x "${model}" ]]; then
    echo "PvrGPU model is missing or not executable: ${model}" >&2
    exit 1
fi

rdc_parent="$(cd "$(dirname "${rdc_path}")" && pwd -P)"
rdc_path="${rdc_parent}/$(basename "${rdc_path}")"
mkdir -p "${outdir}" "${PVRGPU_TMP_ROOT%/}"
outdir="$(cd "${outdir}" && pwd -P)"

if [[ "$(basename "${outdir}")" == "png" ]]; then
    artifact_root="$(cd "${outdir}/.." && pwd -P)"
    model_png_dir="${outdir}"
else
    artifact_root="${outdir}"
    model_png_dir="${outdir}/png"
    mkdir -p "${model_png_dir}"
fi

driver_dri_dir="${artifact_root}/dri"
player_png_dir="${artifact_root}/player-png"
cache_dir="${artifact_root}/xdg-cache"
tmp_dir="${artifact_root}/tmp"
mkdir -p "${driver_dri_dir}" "${player_png_dir}" "${cache_dir}" "${tmp_dir}"

if [[ -f "${mesa_prefix}/lib/dri/swrast_dri.dylib" ]]; then
    driver_search_path="${mesa_prefix}/lib/dri"
else
    gallium_driver=""
    while IFS= read -r candidate; do
        gallium_driver="${candidate}"
        break
    done < <(find "${mesa_prefix}/lib" -maxdepth 1 -name 'libgallium-*.dylib' -type f | sort)
    if [[ -z "${gallium_driver}" ]]; then
        echo "Required Mesa DRI driver is missing under: ${mesa_prefix}/lib" >&2
        echo "Expected lib/dri/swrast_dri.dylib or lib/libgallium-*.dylib." >&2
        exit 1
    fi
    ln -sf "${gallium_driver}" "${driver_dri_dir}/swrast_dri.dylib"
    driver_search_path="${driver_dri_dir}"
fi

rdc_sha256="$(shasum -a 256 "${rdc_path}" | awk '{print $1}')"
trace_stem="$(basename "${rdc_path}" .rdc)"
player_png="${player_png_dir}/${trace_stem}_replay.png"
command_out="${artifact_root}/driver-command.txt"
counter_out="${artifact_root}/driver-counter.txt"
player_stdout="${artifact_root}/player.stdout.log"
player_stderr="${artifact_root}/player.stderr.log"
model_stdout="${artifact_root}/model.stdout.jsonl"
model_stderr="${artifact_root}/model.stderr.log"
rm -f -- "${player_png}" "${command_out}" "${counter_out}" \
    "${player_stdout}" "${player_stderr}" "${model_stdout}" "${model_stderr}"

{
    printf 'schema=pvrgpu.rdc-pvrgpu-driver-systemc.v1\n'
    printf 'rdc=%s\n' "${rdc_path}"
    printf 'rdc_sha256=%s\n' "${rdc_sha256}"
    printf 'case=%s\n' "${case_name}"
    printf 'mesa_prefix=%s\n' "${mesa_prefix}"
    printf 'renderdoc_root=%s\n' "${renderdoc_root}"
    printf 'gles_version_override=%s\n' "${gles_override}"
    printf 'uses_gallium_trace=false\n'
    printf 'uses_command_capsule=false\n'
} >"${artifact_root}/runner.txt"

set +e
env \
    LC_ALL=C \
    LANG=C \
    TZ=UTC \
    TMPDIR="${tmp_dir}" \
    XDG_CACHE_HOME="${cache_dir}" \
    EGL_PLATFORM=surfaceless \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    GALLIUM_DRIVER=pvrgpu \
    LIBGL_DRIVERS_PATH="${driver_search_path}" \
    MESA_SHADER_CACHE_DISABLE=true \
    MESA_GLES_VERSION_OVERRIDE="${gles_override}" \
    RENDERDOC_OUTPUT_FROM_DRAW_FBO=1 \
    RENDERDOC_MESA_EGL_PATH="${mesa_prefix}/lib/libEGL.dylib" \
    RENDERDOC_MESA_GLES_PATH="${mesa_prefix}/lib/libGLESv2.dylib" \
    DYLD_LIBRARY_PATH="${mesa_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    PVRGPU_DRIVER_COMMAND_OUT="${command_out}" \
    PVRGPU_DRIVER_COUNTER_OUT="${counter_out}" \
    PVRGPU_RDC_CASE_NAME="${case_name}" \
    "${player}" "${rdc_path}" "${player_png}" \
    >"${player_stdout}" \
    2>"${player_stderr}"
player_rc=$?
set -e
if [[ "${player_rc}" != 0 ]]; then
    echo "RenderDoc Mesa/Gallium pvrgpu replay failed with code ${player_rc}" >&2
    echo "stdout: ${player_stdout}" >&2
    echo "stderr: ${player_stderr}" >&2
    exit "${player_rc}"
fi
if [[ ! -s "${command_out}" ]]; then
    echo "PvrGPU Gallium driver did not emit a driver command: ${command_out}" >&2
    echo "driver_counter: ${counter_out}" >&2
    exit 1
fi

set +e
"${model}" \
    --driver-command "${command_out}" \
    --outdir "${model_png_dir}" \
    >"${model_stdout}" \
    2>"${model_stderr}"
model_rc=$?
set -e

PYTHONDONTWRITEBYTECODE=1 python3 -c '
from __future__ import annotations

import json
import sys

digest = sys.argv[1]
for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    stripped = line.lstrip()
    if stripped.startswith("{"):
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            print(line)
            continue
        if message.get("type") == "hello" and message.get("backend") == "pvrgpu":
            existing = message.get("rdc_sha256")
            if existing is not None and existing != digest:
                raise SystemExit("PvrGPU hello.rdc_sha256 conflicts with input RDC")
            message["rdc_sha256"] = digest
            print(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
            continue
    print(line)
' "${rdc_sha256}" <"${model_stdout}"

if [[ "${model_rc}" != 0 ]]; then
    echo "PvrGPU SystemC model failed with code ${model_rc}; stderr: ${model_stderr}" >&2
fi
exit "${model_rc}"
