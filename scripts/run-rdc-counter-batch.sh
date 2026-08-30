#!/usr/bin/env bash
# Counter-only, stop-on-first-difference RDC gate.
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
Usage: ./scripts/run-rdc-counter-batch.sh [--case CASE]
       ./scripts/run-rdc-counter-batch.sh --all
       ./scripts/run-rdc-counter-batch.sh --start INDEX

For each selected RDC, in frozen manifest order:
  1. RenderDoc + Mesa llvmpipe -> counter_golden.txt
  2. PvrGPU DUT              -> counter_pvrgpu.txt
  3. exact text comparison; stop at the first difference

PNG files may be emitted by the replay/model executables, but this script
never compares them and they do not affect PASS/FAIL.

By default the current PvrGPU --case adapter is used for step 2.  A separate
explicit PvrGPU runner can replace it without changing this batch:

  PVRGPU_RDC_DUT_RUNNER=/path/to/runner ./scripts/run-rdc-counter-batch.sh --all

The external runner is invoked as:
  runner --rdc FILE --case CASE --width W --height H --outdir DIR
and must write one pvrgpu.counter.v1 JSONL stream to stdout.
EOF
}

selection="case"
selected_case="fill_solid"
start_index=1
while (($# > 0)); do
    case "$1" in
        --case)
            (($# >= 2)) || { echo "--case requires a value" >&2; exit 2; }
            selection="case"
            selected_case="$2"
            shift 2
            ;;
        --all)
            selection="all"
            start_index=1
            shift
            ;;
        --start)
            (($# >= 2)) || { echo "--start requires an index" >&2; exit 2; }
            selection="all"
            start_index="$2"
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

if [[ ! "${selected_case}" =~ ^[a-z0-9_]+$ ]]; then
    echo "Unsafe case name: ${selected_case}" >&2
    exit 2
fi
if [[ ! "${start_index}" =~ ^[1-9][0-9]*$ ]] || ((start_index > 20)); then
    echo "--start must be an integer from 1 through 20" >&2
    exit 2
fi

manifest="${project_dir}/config/rdc-glbench-v1.tsv"
normalizer="${project_dir}/tools/rdc/write_counter_txt.py"
benchscope_root="${PVRGPU_BENCHSCOPE_ROOT:-$(dirname "${project_dir}")/BenchScope}"
golden_runner="${benchscope_root}/scripts/play_glbench_rdc.sh"
pvrgpu_work_root="${PVRGPU_WORK_ROOT%/}"
working_root="$(dirname "${pvrgpu_work_root}")"
rdc_root="${PVRGPU_RDC_ROOT:-${working_root}/drc_patterns/1.GLBench}"
mesa_prefix="${PVRGPU_LLVMPIPE_MESA_PREFIX:-${pvrgpu_work_root}/mesa-counter/install}"
renderdoc_root="${PVRGPU_RENDERDOC_MESA_ROOT:-${working_root}/build/renderdoc-mesa}"
model="${PVRGPU_MODEL_STUB}"
dut_runner="${PVRGPU_RDC_DUT_RUNNER:-}"
output_base="${PVRGPU_RDC_COUNTER_OUTPUT:-${pvrgpu_work_root}/out/rdc-counter-batch}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_root="${output_base}/${run_stamp}"

for required_file in "${manifest}" "${normalizer}" "${golden_runner}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file is missing: ${required_file}" >&2
        exit 1
    fi
done
if [[ -n "${dut_runner}" ]]; then
    if [[ ! -x "${dut_runner}" ]]; then
        echo "PVRGPU_RDC_DUT_RUNNER is not executable: ${dut_runner}" >&2
        exit 1
    fi
    dut_mode="external-pvrgpu-runner"
else
    if [[ ! -x "${model}" ]]; then
        echo "PvrGPU model is missing: ${model}" >&2
        exit 1
    fi
    dut_mode="builtin-case-adapter"
fi

mkdir -p "${run_root}"
cp "${manifest}" "${run_root}/manifest.tsv"
printf 'dut_mode=%s\n' "${dut_mode}" >"${run_root}/run.txt"
printf 'png_compare=false\n' >>"${run_root}/run.txt"
printf 'counter_compare=17-fields-exact\n' >>"${run_root}/run.txt"

matched=0
passed=0
while IFS=$'\t' read -r \
    frame_index case_name rdc_relative rdc_expected_sha256 \
    recorder_png_relative recorder_png_expected_sha256 \
    recorder_report_relative recorder_report_expected_sha256 \
    target_event_policy output_attachment_policy width height \
    format samples color_space compare_policy counter_policy \
    determinism_policy feature_gate
do
    [[ "${frame_index}" == "index" ]] && continue
    if [[ "${selection}" == "case" && "${case_name}" != "${selected_case}" ]]; then
        continue
    fi
    if [[ "${selection}" == "all" ]] && ((frame_index < start_index)); then
        continue
    fi
    matched=$((matched + 1))
    printf -v padded_index '%03d' "${frame_index}"
    case_root="${run_root}/${padded_index}-${case_name}"
    golden_root="${case_root}/golden"
    dut_root="${case_root}/pvrgpu"
    mkdir -p "${golden_root}/output" "${golden_root}/tmp" \
        "${golden_root}/xdg-cache" "${dut_root}/png"

    rdc_path="${rdc_root}/${rdc_relative}"
    if [[ ! -f "${rdc_path}" ]]; then
        echo "RDC is missing: ${rdc_path}" >&2
        exit 1
    fi
    actual_rdc_sha256="$(shasum -a 256 "${rdc_path}" | awk '{print $1}')"
    if [[ "${actual_rdc_sha256}" != "${rdc_expected_sha256}" ]]; then
        echo "RDC hash mismatch: ${rdc_path}" >&2
        exit 1
    fi

    echo "[${padded_index}/020] ${case_name}: Golden"
    trace_stem="$(basename "${rdc_path}" .rdc)"
    env \
        LC_ALL=C \
        LANG=C \
        TZ=UTC \
        TMPDIR="${golden_root}/tmp" \
        XDG_CACHE_HOME="${golden_root}/xdg-cache" \
        MESA_SHADER_CACHE_DISABLE=true \
        MESA_PREFIX="${mesa_prefix}" \
        RENDERDOC_MESA_ROOT="${renderdoc_root}" \
        PLAYER_OUTPUT_ROOT="${golden_root}/output" \
        bash "${golden_runner}" "${rdc_path}" \
        >"${golden_root}/stdout.log" \
        2>"${golden_root}/stderr.log"
    golden_report="${golden_root}/output/${trace_stem}/Report.md"
    if [[ ! -s "${golden_report}" ]] || ! grep -Fqi llvmpipe "${golden_report}"; then
        echo "Golden did not produce an llvmpipe Report.md: ${golden_report}" >&2
        exit 1
    fi
    PYTHONDONTWRITEBYTECODE=1 python3 "${normalizer}" \
        --golden-report "${golden_report}" \
        --output "${case_root}/counter_golden.txt"

    echo "[${padded_index}/020] ${case_name}: PvrGPU"
    if [[ -n "${dut_runner}" ]]; then
        "${dut_runner}" \
            --rdc "${rdc_path}" \
            --case "${case_name}" \
            --width "${width}" \
            --height "${height}" \
            --outdir "${dut_root}/png" \
            >"${dut_root}/stdout.jsonl" \
            2>"${dut_root}/stderr.log"
    else
        "${model}" \
            --frames 1 \
            --width "${width}" \
            --height "${height}" \
            --case "${case_name}" \
            --outdir "${dut_root}/png" \
            >"${dut_root}/stdout.jsonl" \
            2>"${dut_root}/stderr.log"
    fi
    PYTHONDONTWRITEBYTECODE=1 python3 "${normalizer}" \
        --pvrgpu-jsonl "${dut_root}/stdout.jsonl" \
        --output "${case_root}/counter_pvrgpu.txt"

    echo "[${padded_index}/020] ${case_name}: compare"
    if ! cmp -s "${case_root}/counter_golden.txt" \
        "${case_root}/counter_pvrgpu.txt"; then
        diff -u "${case_root}/counter_golden.txt" \
            "${case_root}/counter_pvrgpu.txt" \
            >"${case_root}/counter_diff.txt" || true
        printf 'status=FAIL\nstage=counter-compare\n' >"${case_root}/result.txt"
        echo "RDC_COUNTER_FAIL index=${frame_index} case=${case_name}" >&2
        cat "${case_root}/counter_diff.txt" >&2
        echo "Artifacts: ${case_root}" >&2
        exit 1
    fi

    printf 'status=PASS\n' >"${case_root}/result.txt"
    passed=$((passed + 1))
    echo "RDC_COUNTER_PASS index=${frame_index} case=${case_name} counters=17-exact"
done <"${manifest}"

if ((matched == 0)); then
    echo "No manifest row matched the requested selection." >&2
    exit 2
fi

printf 'status=PASS\nframes=%s\n' "${passed}" >>"${run_root}/run.txt"
echo "RDC_COUNTER_BATCH_PASS frames=${passed} dut_mode=${dut_mode}"
echo "Artifacts: ${run_root}"
