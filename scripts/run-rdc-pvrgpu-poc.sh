#!/usr/bin/env bash
# RenderDoc + Mesa/POC command ingestion runner for one frozen RDC frame.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
caller_model="${PVRGPU_MODEL_STUB:-}"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

usage() {
    cat <<'EOF'
Usage: ./scripts/run-rdc-pvrgpu-poc.sh \
         --rdc FILE --case CASE --width W --height H --outdir DIR

Replay one frozen RDC through RenderDoc + Mesa/POC, translate the RenderDoc
API trace and Mesa Gallium trace into a strict command capsule, then run the
PvrGPU SystemC model with that capsule.

Only the PvrGPU model JSONL stream is written to stdout. Replay/translation
diagnostics and intermediate evidence are kept under a sibling mesa-poc
artifact directory.

Environment overrides:
  PVRGPU_MESA_POC_PREFIX          Mesa/POC install prefix
  PVRGPU_RENDERDOC_MESA_ROOT      RenderDoc player/runtime root
  PVRGPU_RENDERDOC_MESA_PLAYER    RenderDoc player executable
  PVRGPU_RDC_POC_ARTIFACT_DIR     Intermediate artifact directory
  PVRGPU_MODEL_STUB               PvrGPU SystemC model executable
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
            [[ -z "${rdc_path}" ]] || { echo "--rdc may be specified only once" >&2; exit 2; }
            rdc_path="$2"
            shift 2
            ;;
        --case)
            (($# >= 2)) || { echo "--case requires a value" >&2; exit 2; }
            [[ -z "${case_name}" ]] || { echo "--case may be specified only once" >&2; exit 2; }
            case_name="$2"
            shift 2
            ;;
        --width)
            (($# >= 2)) || { echo "--width requires a value" >&2; exit 2; }
            [[ -z "${width}" ]] || { echo "--width may be specified only once" >&2; exit 2; }
            width="$2"
            shift 2
            ;;
        --height)
            (($# >= 2)) || { echo "--height requires a value" >&2; exit 2; }
            [[ -z "${height}" ]] || { echo "--height may be specified only once" >&2; exit 2; }
            height="$2"
            shift 2
            ;;
        --outdir)
            (($# >= 2)) || { echo "--outdir requires a value" >&2; exit 2; }
            [[ -z "${outdir}" ]] || { echo "--outdir may be specified only once" >&2; exit 2; }
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
if [[ ! "${case_name}" =~ ^[a-z0-9_]+$ ]]; then
    echo "Unsafe case name: ${case_name}" >&2
    exit 2
fi
if [[ ! "${width}" =~ ^[1-9][0-9]*$ || ! "${height}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--width and --height must be positive integers" >&2
    exit 2
fi

manifest="${PVRGPU_RDC_MANIFEST:-${project_dir}/config/rdc-glbench-v1.tsv}"
translator="${project_dir}/tools/rdc/build_mesa_poc_command.py"
pvrgpu_work_root="${PVRGPU_WORK_ROOT%/}"
working_root="$(dirname "${pvrgpu_work_root}")"
mesa_prefix="${PVRGPU_MESA_POC_PREFIX:-${pvrgpu_work_root}/mesa-poc/install}"
renderdoc_root="${PVRGPU_RENDERDOC_MESA_ROOT:-${working_root}/build/renderdoc-mesa}"
player="${PVRGPU_RENDERDOC_MESA_PLAYER:-${renderdoc_root}/bin/renderdoc-mesa-player}"
real_dir="${renderdoc_root}/real"
model="${caller_model:-${PVRGPU_MODEL_STUB}}"
artifact_dir="${PVRGPU_RDC_POC_ARTIFACT_DIR:-$(dirname "${outdir}")/mesa-poc}"
tmp_dir="${artifact_dir}/tmp"
cache_dir="${artifact_dir}/xdg-cache"
replay_png="${artifact_dir}/replay.png"
api_trace="${artifact_dir}/renderdoc-api.md"
gallium_trace="${artifact_dir}/gallium.xml"
command_capsule="${artifact_dir}/command.txt"

for required_file in "${rdc_path}" "${manifest}" "${translator}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required input is missing: ${required_file}" >&2
        exit 1
    fi
done
for required_executable in "${player}" "${model}"; do
    if [[ ! -x "${required_executable}" ]]; then
        echo "Required executable is missing: ${required_executable}" >&2
        exit 1
    fi
done
for required_runtime in \
    "${mesa_prefix}/lib/libEGL.dylib" \
    "${mesa_prefix}/lib/libGLESv2.dylib" \
    "${mesa_prefix}/lib/dri/swrast_dri.dylib" \
    "${real_dir}/libMesaEGL.dylib" \
    "${real_dir}/libMesaGLESv2.dylib"
do
    if [[ ! -f "${required_runtime}" ]]; then
        echo "Required Mesa/RenderDoc runtime is missing: ${required_runtime}" >&2
        exit 1
    fi
done

mkdir -p "${outdir}" "${artifact_dir}" "${tmp_dir}" "${cache_dir}"
rm -f -- "${replay_png}" "${api_trace}" "${gallium_trace}" "${command_capsule}"
{
    printf 'schema=pvrgpu.rdc-poc-runner.v1\n'
    printf 'case=%s\n' "${case_name}"
    printf 'width=%s\n' "${width}"
    printf 'height=%s\n' "${height}"
    printf 'rdc=%s\n' "${rdc_path}"
    printf 'mesa_prefix=%s\n' "${mesa_prefix}"
    printf 'renderdoc_root=%s\n' "${renderdoc_root}"
    printf 'png_compare=false\n'
} >"${artifact_dir}/runner.txt"

echo "PvrGPU POC replay: ${case_name} (${width}x${height})" >&2
echo "POC artifacts: ${artifact_dir}" >&2
env \
    LC_ALL=C \
    LANG=C \
    TZ=UTC \
    TMPDIR="${tmp_dir}" \
    XDG_CACHE_HOME="${cache_dir}" \
    EGL_PLATFORM=surfaceless \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    LIBGL_DRIVERS_PATH="${mesa_prefix}/lib/dri" \
    MESA_SHADER_CACHE_DISABLE=true \
    RENDERDOC_MESA_EGL_PATH="${real_dir}/libMesaEGL.dylib" \
    RENDERDOC_MESA_GLES_PATH="${real_dir}/libMesaGLESv2.dylib" \
    DYLD_LIBRARY_PATH="${mesa_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    GALLIUM_TRACE="${gallium_trace}" \
    "${player}" "${rdc_path}" "${replay_png}" "${api_trace}" \
    >"${artifact_dir}/player.stdout.log" \
    2>"${artifact_dir}/player.stderr.log"

for replay_artifact in "${replay_png}" "${api_trace}" "${gallium_trace}"; do
    if [[ ! -s "${replay_artifact}" ]]; then
        echo "RenderDoc/Mesa replay did not produce: ${replay_artifact}" >&2
        exit 1
    fi
done

PYTHONDONTWRITEBYTECODE=1 python3 "${translator}" \
    --rdc "${rdc_path}" \
    --api-trace "${api_trace}" \
    --gallium-trace "${gallium_trace}" \
    --manifest "${manifest}" \
    --case "${case_name}" \
    --width "${width}" \
    --height "${height}" \
    --output "${command_capsule}" \
    >"${artifact_dir}/translator.stdout.log" \
    2>"${artifact_dir}/translator.stderr.log"
if [[ ! -s "${command_capsule}" ]]; then
    echo "Mesa/POC translator did not produce: ${command_capsule}" >&2
    exit 1
fi

echo "Starting PvrGPU SystemC with Mesa command capsule" >&2
exec "${model}" \
    --frames 1 \
    --mesa-command "${command_capsule}" \
    --outdir "${outdir}"
