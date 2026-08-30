#!/usr/bin/env bash
# Stop-on-first-error differential gate: llvmpipe golden versus PvrGPU model.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

case_name="${1:-}"
size="${2:-64x64}"
case "${case_name}" in
    fill_solid|fill_solid_blended|fill_solid_depth_neq|fill_solid_depth_never|fill_tex_nearest|fill_tex_bilinear|fill_tex_trilinear_linear_01|fill_tex_trilinear_linear_04|fill_tex_trilinear_linear_05)
        test_group="fill_rate"
        ;;
    triangle_setup|triangle_setup_all_culled|triangle_setup_half_culled)
        test_group="triangle_setup"
        ;;
    attribute_fetch_shader|attribute_fetch_shader_2_attr|attribute_fetch_shader_4_attr|attribute_fetch_shader_8_attr)
        test_group="attribute_fetch_shader"
        ;;
    varyings_shader_1|varyings_shader_2|varyings_shader_4|varyings_shader_8)
        test_group="varyings_ddx_shader"
        ;;
    *)
        echo "Unsupported differential case: ${case_name:-<missing>}" >&2
        echo "Supported: fill_solid fill_solid_blended fill_solid_depth_neq fill_solid_depth_never fill_tex_nearest fill_tex_bilinear fill_tex_trilinear_linear_01 fill_tex_trilinear_linear_04 fill_tex_trilinear_linear_05 triangle_setup triangle_setup_all_culled triangle_setup_half_culled attribute_fetch_shader attribute_fetch_shader_2_attr attribute_fetch_shader_4_attr attribute_fetch_shader_8_attr varyings_shader_1 varyings_shader_2 varyings_shader_4 varyings_shader_8" >&2
        exit 2
        ;;
esac
if [[ ! "${size}" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]]; then
    echo "Size must be WIDTHxHEIGHT: ${size}" >&2
    exit 2
fi
width="${size%x*}"
height="${size#*x}"

runner="${PVRGPU_GLBENCH_RUNNER}"
mesa_prefix="${PVRGPU_LLVMPIPE_MESA_PREFIX}"
model="${PVRGPU_MODEL_STUB}"
for executable in "${runner}" "${model}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Required executable is missing: ${executable}" >&2
        exit 1
    fi
done
if ! "${runner}" --list | grep -Fqx "${test_group}:${case_name}"; then
    echo "Pinned GLBench runner does not advertise ${test_group}:${case_name}" >&2
    exit 1
fi

run_stamp="$(date +%Y%m%d-%H%M%S)-$$"
run_root="${PVRGPU_WORK_ROOT}/differential/glbench/${case_name}-${size}-${run_stamp}"
golden_dir="${run_root}/llvmpipe"
model_dir="${run_root}/pvrgpu"
mkdir -p "${golden_dir}/png" "${golden_dir}/tmp" "${model_dir}/png"

current_stage="llvmpipe"
report_failure() {
    exit_code=$?
    trap - EXIT
    if [[ ${exit_code} -ne 0 ]]; then
        echo "DIFFERENTIAL_FAIL case=${case_name} size=${size} stage=${current_stage}" >&2
        echo "Artifacts: ${run_root}" >&2
    fi
    exit "${exit_code}"
}
trap report_failure EXIT

env \
    EGL_PLATFORM=surfaceless \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    GALLIUM_DRIVER=llvmpipe \
    LIBGL_DRIVERS_PATH="${mesa_prefix}/lib/dri" \
    DYLD_LIBRARY_PATH="${mesa_prefix}/lib" \
    MESA_SHADER_CACHE_DISABLE=true \
    MESA_COUNTER_REPORT_PATH="${golden_dir}/Report.md" \
    MESA_COUNTER_FRAME_SELECTION_MS="PvrGPU differential ${case_name}" \
    TMPDIR="${golden_dir}/tmp" \
    "${runner}" \
        --test "${test_group}" \
        --case "${case_name}" \
        --sample 1 \
        --size "${size}" \
        --outdir "${golden_dir}/png" \
        >"${golden_dir}/stdout.log" \
        2>"${golden_dir}/stderr.log"

grep -q 'llvmpipe' "${golden_dir}/Report.md"
grep -q '@TEST_END' "${golden_dir}/stdout.log"

current_stage="pvrgpu"
"${model}" \
    --frames 1 \
    --width "${width}" \
    --height "${height}" \
    --case "${case_name}" \
    --outdir "${model_dir}/png" \
    >"${model_dir}/stdout.log" \
    2>"${model_dir}/stderr.log"

if grep -q '"type":"error"' "${model_dir}/stdout.log"; then
    echo "PvrGPU emitted a protocol error; stopping at ${case_name}." >&2
    exit 1
fi
grep -q '"type":"done".*"pool_leaks":0' "${model_dir}/stdout.log"

current_stage="counter-compare"
PYTHONDONTWRITEBYTECODE=1 python3 \
    "${project_dir}/tools/check_glbench_differential_counters.py" \
    --case "${case_name}" \
    --golden-report "${golden_dir}/Report.md" \
    --model-log "${model_dir}/stdout.log" \
    2>&1 | tee "${run_root}/counter-compare.log"

golden_png="${golden_dir}/png/${case_name}_sample_000001.png"
model_png="${model_dir}/png/${case_name}_sample_000001.png"
current_stage="rgba-compare"
PYTHONDONTWRITEBYTECODE=1 python3 "${project_dir}/tools/compare_png_rgba.py" \
    "${golden_png}" "${model_png}" \
    2>&1 | tee "${run_root}/rgba-compare.log"

current_stage="artifact-manifest"
{
    printf 'case=%s\n' "${case_name}"
    printf 'size=%s\n' "${size}"
    printf 'golden_renderer=llvmpipe\n'
    printf 'model_renderer=pvrgpu-systemc\n'
    printf 'result=PASS\n'
} >"${run_root}/manifest.txt"

(
    cd "${run_root}"
    shasum -a 256 \
        llvmpipe/Report.md \
        llvmpipe/stdout.log \
        llvmpipe/stderr.log \
        "llvmpipe/png/${case_name}_sample_000001.png" \
        pvrgpu/stdout.log \
        pvrgpu/stderr.log \
        "pvrgpu/png/${case_name}_sample_000001.png" \
        counter-compare.log \
        rgba-compare.log \
        manifest.txt \
        >SHA256SUMS
)

trap - EXIT
echo "DIFFERENTIAL_PASS case=${case_name} size=${size}"
echo "Artifacts: ${run_root}"
