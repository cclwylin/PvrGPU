#!/usr/bin/env bash
# Frozen RDC replay gate: RenderDoc + Mesa llvmpipe Golden only.
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
Usage: ./scripts/run-rdc-golden.sh [--case CASE]
       ./scripts/run-rdc-golden.sh [CASE]

Replay one frozen GLBench RDC twice in independent cold output/cache
directories through BenchScope RenderDoc + Mesa llvmpipe.  With no argument,
only Frame 1 (fill_solid) runs.  This command never runs the DUT or advances
suite progress.

Environment overrides:
  PVRGPU_RDC_ROOT              Input root (default: sibling drc_patterns/1.GLBench)
  PVRGPU_RDC_GOLDEN_OUTPUT     Artifact root (default: PVRGPU_WORK_ROOT/out/rdc-differential)
  PVRGPU_BENCHSCOPE_ROOT       BenchScope workspace
  PVRGPU_RENDERDOC_MESA_ROOT   Pinned RenderDoc/Mesa runtime
  PVRGPU_LLVMPIPE_MESA_PREFIX  Pinned counter-enabled Mesa prefix
EOF
}

case_name="fill_solid"
case_was_set=0
while (($# > 0)); do
    case "$1" in
        --case)
            if (($# < 2)); then
                echo "--case requires a value" >&2
                exit 2
            fi
            if ((case_was_set)); then
                echo "Specify exactly one case." >&2
                exit 2
            fi
            case_name="$2"
            case_was_set=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if ((case_was_set)); then
                echo "Specify exactly one case." >&2
                exit 2
            fi
            case_name="$1"
            case_was_set=1
            shift
            ;;
    esac
done

if [[ ! "${case_name}" =~ ^[a-z0-9_]+$ ]]; then
    echo "Unsafe case name: ${case_name}" >&2
    exit 2
fi

manifest="${project_dir}/config/rdc-glbench-v1.tsv"
provenance_tool="${project_dir}/tools/rdc/check_golden_provenance.py"
rgba_compare_tool="${project_dir}/tools/compare_png_rgba.py"
pvrgpu_work_root="${PVRGPU_WORK_ROOT%/}"
working_root="$(dirname "${pvrgpu_work_root}")"
rdc_root="${PVRGPU_RDC_ROOT:-${working_root}/drc_patterns/1.GLBench}"
artifact_base="${PVRGPU_RDC_GOLDEN_OUTPUT:-${pvrgpu_work_root}/out/rdc-differential}"
benchscope_root="${PVRGPU_BENCHSCOPE_ROOT:-$(dirname "${project_dir}")/BenchScope}"
benchscope_script="${benchscope_root}/scripts/play_glbench_rdc.sh"
gfx_player_script="${benchscope_root}/scripts/play-gfxbench-renderdoc-mesa.sh"
renderdoc_root="${PVRGPU_RENDERDOC_MESA_ROOT:-${working_root}/build/renderdoc-mesa}"
mesa_prefix="${PVRGPU_LLVMPIPE_MESA_PREFIX:-${pvrgpu_work_root}/mesa-counter/install}"

for required_file in \
    "${manifest}" \
    "${provenance_tool}" \
    "${rgba_compare_tool}" \
    "${benchscope_script}" \
    "${gfx_player_script}"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required Golden harness file is missing: ${required_file}" >&2
        exit 1
    fi
done

selected_row="$({
    awk -F '\t' -v selected_case="${case_name}" '
        NR > 1 && $2 == selected_case { print; matches += 1 }
        END { if (matches != 1) exit 1 }
    ' "${manifest}"
} || true)"
if [[ -z "${selected_row}" ]]; then
    echo "Case is not in the frozen 20-frame manifest: ${case_name}" >&2
    exit 2
fi

IFS=$'\t' read -r \
    frame_index manifest_case rdc_relative rdc_expected_sha256 \
    recorder_png_relative recorder_png_expected_sha256 \
    recorder_report_relative recorder_report_expected_sha256 \
    manifest_policy_fields <<<"${selected_row}"
if [[ "${manifest_case}" != "${case_name}" ]]; then
    echo "Internal manifest selection error for ${case_name}" >&2
    exit 1
fi
printf -v padded_index '%03d' "${frame_index}"

rdc_path="${rdc_root}/${rdc_relative}"
recorder_png="${rdc_root}/${recorder_png_relative}"
recorder_report="${rdc_root}/${recorder_report_relative}"
trace_stem="$(basename "${rdc_path}" .rdc)"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
case_artifact_parent="${artifact_base}/golden/${padded_index}-${case_name}"
artifact_root="${case_artifact_parent}/${run_stamp}"
mkdir -p "${case_artifact_parent}"
if ! mkdir "${artifact_root}"; then
    echo "Golden artifact directory already exists: ${artifact_root}" >&2
    exit 1
fi
mkdir -p "${artifact_root}/compare"
cp "${manifest}" "${artifact_root}/suite-manifest.tsv"

current_stage="setup"
seal_hashes() {
    (
        cd "${artifact_root}"
        : >.SHA256SUMS.tmp
        while IFS= read -r artifact_file; do
            shasum -a 256 "${artifact_file}" >>.SHA256SUMS.tmp
        done < <(
            find . -type f \
                ! -name SHA256SUMS \
                ! -name .SHA256SUMS.tmp \
                -print | LC_ALL=C sort
        )
        mv .SHA256SUMS.tmp SHA256SUMS
    )
}

report_failure() {
    exit_code=$?
    trap - EXIT
    if [[ ${exit_code} -ne 0 ]]; then
        rm -f -- "${artifact_root}/GOLDEN_PASS"
        {
            printf 'schema=pvrgpu.rdc.golden-result.v1\n'
            printf 'suite=rdc-glbench-v1\n'
            printf 'index=%s\n' "${frame_index}"
            printf 'case=%s\n' "${case_name}"
            printf 'status=FAIL\n'
            printf 'stage=%s\n' "${current_stage}"
            printf 'exit_code=%s\n' "${exit_code}"
        } >"${artifact_root}/result.txt"
        seal_hashes || true
        echo "GOLDEN_FAIL index=${frame_index} case=${case_name} stage=${current_stage}" >&2
        echo "Artifacts: ${artifact_root}" >&2
    fi
    exit "${exit_code}"
}
trap report_failure EXIT

current_stage="input-provenance"
PYTHONDONTWRITEBYTECODE=1 python3 "${provenance_tool}" manifest \
    --manifest "${manifest}" \
    --rdc-root "${rdc_root}" \
    --case "${case_name}" \
    --output "${artifact_root}/input-provenance.json" \
    2>&1 | tee "${artifact_root}/input-provenance.log"

current_stage="runtime-provenance"
PYTHONDONTWRITEBYTECODE=1 python3 "${provenance_tool}" runtime \
    --manifest "${manifest}" \
    --benchscope-script "${benchscope_script}" \
    --gfx-player-script "${gfx_player_script}" \
    --renderdoc-root "${renderdoc_root}" \
    --mesa-prefix "${mesa_prefix}" \
    --output "${artifact_root}/runtime-provenance.json" \
    2>&1 | tee "${artifact_root}/runtime-provenance.log"

replay_reports=()
replay_pngs=()
for replay_number in 1 2; do
    printf -v run_label 'run-%02d' "${replay_number}"
    replay_root="${artifact_root}/golden/${run_label}"
    player_output_root="${replay_root}/output"
    mkdir -p \
        "${replay_root}/tmp" \
        "${replay_root}/xdg-cache" \
        "${player_output_root}"

    current_stage="golden-${run_label}"
    env \
        LC_ALL=C \
        LANG=C \
        TZ=UTC \
        TMPDIR="${replay_root}/tmp" \
        XDG_CACHE_HOME="${replay_root}/xdg-cache" \
        MESA_SHADER_CACHE_DISABLE=true \
        DYLD_LIBRARY_PATH= \
        MESA_PREFIX="${mesa_prefix}" \
        RENDERDOC_MESA_ROOT="${renderdoc_root}" \
        PLAYER_OUTPUT_ROOT="${player_output_root}" \
        bash "${benchscope_script}" "${rdc_path}" \
        >"${replay_root}/stdout.log" \
        2>"${replay_root}/stderr.log"

    replay_output="${player_output_root}/${trace_stem}"
    replay_report="${replay_output}/Report.md"
    replay_png="${replay_output}/png/${trace_stem}_replay.png"
    if [[ ! -s "${replay_report}" || ! -s "${replay_png}" ]]; then
        echo "${run_label} did not produce both Report.md and replay PNG." >&2
        exit 1
    fi
    if ! grep -Fqi 'llvmpipe' "${replay_report}"; then
        echo "${run_label} did not prove an llvmpipe renderer." >&2
        exit 1
    fi
    if ! grep -Fq 'Player complete.' "${replay_root}/stdout.log"; then
        echo "${run_label} did not reach the BenchScope completion marker." >&2
        exit 1
    fi
    replay_reports+=("${replay_report}")
    replay_pngs+=("${replay_png}")
done

current_stage="rgba-recorder-run-01"
PYTHONDONTWRITEBYTECODE=1 python3 "${rgba_compare_tool}" \
    "${recorder_png}" "${replay_pngs[0]}" \
    2>&1 | tee "${artifact_root}/compare/rgba-recorder-run-01.log"

current_stage="rgba-recorder-run-02"
PYTHONDONTWRITEBYTECODE=1 python3 "${rgba_compare_tool}" \
    "${recorder_png}" "${replay_pngs[1]}" \
    2>&1 | tee "${artifact_root}/compare/rgba-recorder-run-02.log"

current_stage="rgba-cold-determinism"
PYTHONDONTWRITEBYTECODE=1 python3 "${rgba_compare_tool}" \
    "${replay_pngs[0]}" "${replay_pngs[1]}" \
    2>&1 | tee "${artifact_root}/compare/rgba-run-01-run-02.log"

current_stage="counter-self-check"
PYTHONDONTWRITEBYTECODE=1 python3 "${provenance_tool}" counters \
    --case "${case_name}" \
    --recorder "${recorder_report}" \
    --replay "${replay_reports[0]}" \
    --replay "${replay_reports[1]}" \
    --output "${artifact_root}/compare/counter-self-check.json" \
    2>&1 | tee "${artifact_root}/compare/counter-self-check.log"

current_stage="seal"
{
    printf 'schema=pvrgpu.rdc.golden-result.v1\n'
    printf 'suite=rdc-glbench-v1\n'
    printf 'index=%s\n' "${frame_index}"
    printf 'case=%s\n' "${case_name}"
    printf 'status=PASS\n'
    printf 'cold_replays=2\n'
    printf 'rdc_sha256=%s\n' "${rdc_expected_sha256}"
    printf 'recorder_png_sha256=%s\n' "${recorder_png_expected_sha256}"
    printf 'recorder_report_sha256=%s\n' "${recorder_report_expected_sha256}"
    printf 'rgba_policy=rgba8-exact-regression-v1\n'
    printf 'counter_policy=llvmpipe-17-exact-selection-normalized-v1\n'
    printf 'determinism_policy=two-cold-replays-exact-v1\n'
} >"${artifact_root}/result.txt"
printf 'GOLDEN_PASS index=%s case=%s\n' "${frame_index}" "${case_name}" \
    >"${artifact_root}/GOLDEN_PASS"
seal_hashes

trap - EXIT
echo "GOLDEN_PASS index=${frame_index} case=${case_name} cold_replays=2 rgba=exact counters=17-exact"
echo "Artifacts: ${artifact_root}"
