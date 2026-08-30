#!/usr/bin/env bash
# Replay one already-captured dEQP RDC through RenderDoc + Mesa llvmpipe.
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
Usage: ./scripts/run-deqp-capture-golden.sh --rdc FILE.rdc --case DEQP_CASE --outdir DIR

Replay one pre-recorded dEQP RDC capture through the existing RenderDoc + Mesa
llvmpipe player path. This command never invokes a dEQP binary.

Outputs:
  DIR/Report.md
  DIR/counter_golden.txt
  DIR/player-output/
  DIR/stdout.log
  DIR/stderr.log
EOF
}

rdc_path=""
case_name=""
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

if [[ -z "${rdc_path}" || -z "${case_name}" || -z "${outdir}" ]]; then
    echo "--rdc, --case, and --outdir are required" >&2
    usage >&2
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
golden_runner="${PVRGPU_DEQP_CAPTURE_PLAYER:-${benchscope_root}/scripts/play_glbench_rdc.sh}"
mesa_prefix="${PVRGPU_LLVMPIPE_MESA_PREFIX:-${pvrgpu_work_root}/mesa-counter/install}"
renderdoc_root="${PVRGPU_RENDERDOC_MESA_ROOT:-${working_root}/build/renderdoc-mesa}"
counter_tool="${project_dir}/tools/rdc/write_counter_txt.py"

for required_file in "${golden_runner}" "${counter_tool}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file is missing: ${required_file}" >&2
        exit 1
    fi
done
for required_runtime in \
    "${mesa_prefix}/lib/libEGL.dylib" \
    "${mesa_prefix}/lib/libGLESv2.dylib" \
    "${renderdoc_root}/bin/renderdoc-mesa-player"
do
    if [[ ! -f "${required_runtime}" ]]; then
        echo "Required runtime is missing: ${required_runtime}" >&2
        exit 1
    fi
done

rdc_parent="$(cd "$(dirname "${rdc_path}")" && pwd -P)"
rdc_path="${rdc_parent}/$(basename "${rdc_path}")"
mkdir -p "${outdir}"
outdir="$(cd "${outdir}" && pwd -P)"

player_output_root="${outdir}/player-output"
report_output="${outdir}/Report.md"
counter_output="${outdir}/counter_golden.txt"
if [[ -e "${player_output_root}" || -e "${report_output}" || -e "${counter_output}" ]]; then
    echo "Golden outdir already contains prior output: ${outdir}" >&2
    exit 1
fi

tmp_dir="${outdir}/tmp"
cache_dir="${outdir}/xdg-cache"
mkdir -p "${player_output_root}" "${tmp_dir}" "${cache_dir}"

{
    printf 'schema=pvrgpu.deqp-capture-golden.v1\n'
    printf 'case=%s\n' "${case_name}"
    printf 'rdc=%s\n' "${rdc_path}"
    printf 'mesa_prefix=%s\n' "${mesa_prefix}"
    printf 'renderdoc_root=%s\n' "${renderdoc_root}"
    printf 'uses_deqp_binary=false\n'
} >"${outdir}/runner.txt"

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
    bash "${golden_runner}" "${rdc_path}" \
    >"${outdir}/stdout.log" \
    2>"${outdir}/stderr.log"

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
PYTHONDONTWRITEBYTECODE=1 python3 "${counter_tool}" \
    --golden-report "${report_output}" \
    --output "${counter_output}"

echo "DEQP_CAPTURE_GOLDEN_READY case=${case_name} report=${report_output}"
