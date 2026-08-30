#!/usr/bin/env bash
# Fail-closed PvrGPU hook for pre-recorded dEQP RDC captures.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/run-deqp-capture-pvrgpu-probe.sh \
         --rdc FILE.rdc --case DEQP_CASE --phase N --phase-key KEY --outdir DIR

This Phase 0-6 hook intentionally does not invoke dEQP binaries and does not
pretend that arbitrary dEQP captures can already lower to PrvGPU commands.
It writes a probe artifact and exits 3, which the capture report classifies as
UNSUPPORTED. Replace this hook with a real lowering runner phase by phase.
EOF
}

rdc_path=""
case_name=""
phase=""
phase_key=""
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
        --phase)
            (($# >= 2)) || { echo "--phase requires a value" >&2; exit 2; }
            phase="$2"
            shift 2
            ;;
        --phase-key)
            (($# >= 2)) || { echo "--phase-key requires a value" >&2; exit 2; }
            phase_key="$2"
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

if [[ -z "${rdc_path}" || -z "${case_name}" || -z "${phase}" ||
      -z "${phase_key}" || -z "${outdir}" ]]; then
    echo "--rdc, --case, --phase, --phase-key, and --outdir are required" >&2
    usage >&2
    exit 2
fi
if [[ ! -f "${rdc_path}" ]]; then
    echo "RDC is missing or is not a regular file: ${rdc_path}" >&2
    exit 1
fi
if [[ ! "${phase}" =~ ^[0-6]$ ]]; then
    echo "--phase must be in the range 0..6" >&2
    exit 2
fi
if [[ -e "${outdir}" && ! -d "${outdir}" ]]; then
    echo "PvrGPU output path is not a directory: ${outdir}" >&2
    exit 1
fi

mkdir -p "${outdir}"
{
    printf 'schema=pvrgpu.deqp-capture-pvrgpu-probe.v1\n'
    printf 'rdc=%s\n' "${rdc_path}"
    printf 'case=%s\n' "${case_name}"
    printf 'phase=%s\n' "${phase}"
    printf 'phase_key=%s\n' "${phase_key}"
    printf 'uses_deqp_binary=false\n'
    printf 'counter_pvrgpu=false\n'
    printf 'status=UNSUPPORTED\n'
} >"${outdir}/probe.txt"

{
    printf 'PrvGPU dEQP capture lowering is not implemented yet for %s.\n' "${phase_key}"
    printf 'RDC: %s\n' "${rdc_path}"
    printf 'Case: %s\n' "${case_name}"
} >"${outdir}/unsupported.txt"

echo "PVRGPU_DEQP_CAPTURE_UNSUPPORTED phase=${phase} case=${case_name}" >&2
exit 3
