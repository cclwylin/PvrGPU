#!/usr/bin/env bash
# Single-RDC PvrGPU counter runner for the current SystemC case adapter.
#
# This runner intentionally does not replay RDC files through RenderDoc and
# does not build or ingest trace capsules.  The RDC file is used only as the
# provenance object whose SHA-256 is attached to the PvrGPU hello record so the
# directory regression worker can bind the output to the exact input file.
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
Usage: ./scripts/run-rdc-pvrgpu-case-counter.sh \
         --rdc FILE --case CASE --width W --height H --outdir DIR

Run one manifest-mapped workload through the current PvrGPU SystemC case
adapter and emit pvrgpu.counter.v1 JSONL to stdout.

This is a regression adapter, not an RDC trace replay path.  It does not read
Gallium traces and does not create command capsules.
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
if [[ ! -x "${PVRGPU_MODEL_STUB}" ]]; then
    echo "PvrGPU model is missing or not executable: ${PVRGPU_MODEL_STUB}" >&2
    exit 1
fi

mkdir -p "${outdir}" "${PVRGPU_TMP_ROOT%/}"
rdc_sha256="$(shasum -a 256 "${rdc_path}" | awk '{print $1}')"
tmp_dir="$(mktemp -d "${PVRGPU_TMP_ROOT%/}/pvrgpu-rdc-case-counter.XXXXXX")"
cleanup() {
    if [[ -n "${tmp_dir:-}" &&
          "${tmp_dir}" == "${PVRGPU_TMP_ROOT%/}"/pvrgpu-rdc-case-counter.* ]]; then
        rm -rf -- "${tmp_dir}"
    fi
}
trap cleanup EXIT

model_stdout="${tmp_dir}/model.stdout.jsonl"
set +e
"${PVRGPU_MODEL_STUB}" \
    --frames 1 \
    --width "${width}" \
    --height "${height}" \
    --case "${case_name}" \
    --outdir "${outdir}" \
    >"${model_stdout}"
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

exit "${model_rc}"
