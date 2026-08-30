#!/usr/bin/env bash
# End-to-end Mesa frontend -> pvrgpu Gallium driver -> driver command ->
# PvrGPU SystemC model smoke test for the first draw vertical slice.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

mesa_prefix="${PVRGPU_MESA_PVRGPU_PREFIX:-${PVRGPU_WORK_ROOT%/}/tmp/pvrgpu-mesa-install}"
model="${PVRGPU_MODEL_STUB}"
output_root="${PVRGPU_MESA_DRIVER_SMOKE_OUTPUT:-${PVRGPU_WORK_ROOT%/}/out/pvrgpu-mesa-driver-smoke}"
size="16x16"

usage() {
    cat <<'EOF'
Usage: ./scripts/run-pvrgpu-mesa-driver-triangle-systemc-smoke.sh [options]

Options:
  --mesa-prefix DIR  Mesa prefix containing the pvrgpu Gallium runtime.
                     Default: PVRGPU_WORK_ROOT/tmp/pvrgpu-mesa-install
  --model PATH       pvrgpu-model-stub executable.
                     Default: PVRGPU_MODEL_STUB
  --outdir DIR       Artifact directory. Default: timestamped dir under
                     PVRGPU_WORK_ROOT/out/pvrgpu-mesa-driver-smoke
  --size WxH         Pbuffer/model size. Default: 16x16

Expected result:
  The smoke creates driver/driver-command.txt with command=draw_triangle,
  runs pvrgpu-model-stub --driver-command, and verifies the model emitted
  driver_triangle_solid counters plus a red/black PNG.
EOF
}

while (($# > 0)); do
    case "$1" in
        --mesa-prefix)
            (($# >= 2)) || { echo "--mesa-prefix requires a directory" >&2; exit 2; }
            mesa_prefix="$2"
            shift 2
            ;;
        --model)
            (($# >= 2)) || { echo "--model requires an executable path" >&2; exit 2; }
            model="$2"
            shift 2
            ;;
        --outdir)
            (($# >= 2)) || { echo "--outdir requires a directory" >&2; exit 2; }
            output_root="$2"
            shift 2
            ;;
        --size)
            (($# >= 2)) || { echo "--size requires WIDTHxHEIGHT" >&2; exit 2; }
            size="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${size}" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
    echo "--size must be WIDTHxHEIGHT with positive integers" >&2
    exit 2
fi

if [[ ! -x "${model}" ]]; then
    echo "pvrgpu-model-stub is missing or not executable: ${model}" >&2
    echo "Hint: build the model first, or pass --model PATH." >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${output_root}" == "${PVRGPU_WORK_ROOT%/}/out/pvrgpu-mesa-driver-smoke" ]]; then
    output_dir="${output_root}/triangle-systemc-smoke-${timestamp}-$$"
else
    output_dir="${output_root}"
fi
driver_dir="${output_dir}/driver"
model_dir="${output_dir}/model-systemc"
mkdir -p "${driver_dir}" "${model_dir}"

"${project_dir}/scripts/run-pvrgpu-mesa-driver-triangle-smoke.sh" \
    --mesa-prefix "${mesa_prefix}" \
    --size "${size}" \
    --outdir "${driver_dir}" \
    >"${output_dir}/driver-smoke.stdout" \
    2>"${output_dir}/driver-smoke.stderr"

command_out="${driver_dir}/driver-command.txt"
if [[ ! -s "${command_out}" ]]; then
    echo "Driver triangle smoke did not produce command: ${command_out}" >&2
    exit 1
fi

"${model}" \
    --driver-command "${command_out}" \
    --outdir "${model_dir}" \
    --cache-bypass on \
    >"${model_dir}/stdout.jsonl" \
    2>"${model_dir}/stderr.log"

PYTHONDONTWRITEBYTECODE=1 python3 - "${project_dir}" "${model_dir}" <<'PY'
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

project_dir = Path(sys.argv[1])
model_dir = Path(sys.argv[2])

messages = [
    json.loads(line)
    for line in (model_dir / "stdout.jsonl").read_text(encoding="utf-8").splitlines()
    if line.startswith("{")
]
hello = [message for message in messages if message.get("type") == "hello"]
counters = [message for message in messages if message.get("type") == "counter"]
done = [message for message in messages if message.get("type") == "done"]
if len(hello) != 1 or len(counters) != 1 or not done:
    raise SystemExit("model JSONL did not contain one hello/counter and a done record")
if hello[0].get("mode") != "pvrgpu-driver-draw-triangle-phase2":
    raise SystemExit("model did not enter driver triangle mode")
if hello[0].get("driver_command") != "draw_triangle":
    raise SystemExit("model did not ingest draw_triangle command")
if counters[0].get("functional_scope") != "driver_triangle_solid-pco-iss-v1":
    raise SystemExit("model counter did not report driver_triangle_solid scope")
values = counters[0].get("counters")
if not isinstance(values, dict):
    raise SystemExit("model counter payload is missing")
expected = {
    "ia_vertices": 3,
    "ia_primitives": 1,
    "vs_invocations": 3,
    "c_primitives": 1,
    "setup_triangles": 1,
}
for field, value in expected.items():
    if values.get(field) != value:
        raise SystemExit(f"{field}={values.get(field)!r}, expected {value}")
if values.get("ps_invocations", 0) <= 0 or values.get("pbe_pixels_written", 0) <= 0:
    raise SystemExit("model did not shade/write triangle pixels")
if done[-1].get("pool_leaks") != 0:
    raise SystemExit("model reported memory-pool leaks")

spec = importlib.util.spec_from_file_location(
    "check_model_pipeline", project_dir / "tests" / "check_model_pipeline.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
png = model_dir / "driver_triangle_solid_sample_000001.png"
width, height, pixels = module.decode_rgba8_png(png)
red = sum(
    1
    for offset in range(0, len(pixels), 4)
    if pixels[offset : offset + 4] == b"\xff\x00\x00\xff"
)
black = sum(
    1
    for offset in range(0, len(pixels), 4)
    if pixels[offset : offset + 4] == b"\x00\x00\x00\xff"
)
if red <= 0 or black <= 0 or red + black != width * height:
    raise SystemExit(
        f"unexpected PNG pixels: size={width}x{height} red={red} black={black}"
    )
print(f"model_pixels={width}x{height} red={red} black={black}")
PY

echo "PvrGPU Mesa driver triangle SystemC smoke PASS"
echo "artifacts=${output_dir}"
echo "driver_command=${command_out}"
echo "model_jsonl=${model_dir}/stdout.jsonl"
echo "model_png=${model_dir}/driver_triangle_solid_sample_000001.png"
