#!/usr/bin/env bash
# Replay one manifest-mapped RDC with the counter-enabled llvmpipe Golden.
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
Usage: ./scripts/run-rdc-golden-counter.sh \
         --rdc FILE --case CASE --width W --height H --outdir DIR

Replay one RDC through BenchScope RenderDoc + the counter-enabled Mesa
llvmpipe runtime.  The generated one-frame Report.md is copied to:

  DIR/Report.md

This adapter does not run PvrGPU, compare PNGs, or perform a second replay.

Environment overrides:
  PVRGPU_BENCHSCOPE_ROOT       BenchScope workspace
  PVRGPU_RENDERDOC_MESA_ROOT  RenderDoc player/runtime root
  PVRGPU_LLVMPIPE_MESA_PREFIX Counter-enabled Mesa install prefix
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
if [[ ! -f "${rdc_path}" ]]; then
    echo "RDC is missing or is not a regular file: ${rdc_path}" >&2
    exit 1
fi
if [[ -e "${outdir}" && ! -d "${outdir}" ]]; then
    echo "Golden output path is not a directory: ${outdir}" >&2
    exit 1
fi

pvrgpu_work_root="${PVRGPU_WORK_ROOT%/}"
working_root="$(dirname "${pvrgpu_work_root}")"
benchscope_root="${PVRGPU_BENCHSCOPE_ROOT:-$(dirname "${project_dir}")/BenchScope}"
golden_runner="${benchscope_root}/scripts/play_glbench_rdc.sh"
mesa_prefix="${PVRGPU_LLVMPIPE_MESA_PREFIX:-${pvrgpu_work_root}/mesa-counter/install}"
renderdoc_root="${PVRGPU_RENDERDOC_MESA_ROOT:-${working_root}/build/renderdoc-mesa}"

if [[ ! -f "${golden_runner}" ]]; then
    echo "BenchScope Golden runner is missing: ${golden_runner}" >&2
    exit 1
fi

rdc_parent="$(cd "$(dirname "${rdc_path}")" && pwd -P)"
rdc_path="${rdc_parent}/$(basename "${rdc_path}")"
mkdir -p "${outdir}"
outdir="$(cd "${outdir}" && pwd -P)"

player_output_root="${outdir}/player-output"
report_output="${outdir}/Report.md"
if [[ -e "${player_output_root}" || -e "${report_output}" ]]; then
    echo "Golden outdir already contains player-output or Report.md: ${outdir}" >&2
    exit 1
fi

tmp_dir="${outdir}/tmp"
cache_dir="${outdir}/xdg-cache"
mkdir -p "${player_output_root}" "${tmp_dir}" "${cache_dir}"

echo "Golden llvmpipe replay: ${case_name} (${width}x${height})" >&2
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
    MESA_SHADER_CACHE_DISABLE=true \
    RENDERDOC_OUTPUT_FROM_DRAW_FBO=1 \
    DYLD_LIBRARY_PATH= \
    MESA_PREFIX="${mesa_prefix}" \
    RENDERDOC_MESA_ROOT="${renderdoc_root}" \
    PLAYER_OUTPUT_ROOT="${player_output_root}" \
    bash "${golden_runner}" "${rdc_path}"

trace_stem="$(basename "${rdc_path}" .rdc)"
generated_report="${player_output_root}/${trace_stem}/Report.md"
if [[ ! -s "${generated_report}" ]]; then
    echo "Golden replay did not produce a non-empty Report.md: ${generated_report}" >&2
    exit 1
fi
if ! grep -Fqi 'llvmpipe' "${generated_report}"; then
    echo "Golden Report.md does not identify an llvmpipe renderer: ${generated_report}" >&2
    exit 1
fi

cp "${generated_report}" "${report_output}"
echo "GOLDEN_COUNTER_READY case=${case_name} report=${report_output}"
