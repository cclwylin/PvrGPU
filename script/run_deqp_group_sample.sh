#!/usr/bin/env bash
# ==============================================================================
# PvrGPU - dEQP group sampler
# ------------------------------------------------------------------------------
# Runs a sample of one group from tools/deqp_groups.py against the ALREADY-BUILT
# PCO driver and PvrGPU model, then prints a pass/fail/skip tally and the QPA
# reason behind every failure. Nothing is compiled here; this is a thin front end
# over script/run_deqp_dynamic.sh, which resolves the driver and bridge paths
# from config/local.env.
#
# Usage:
#   ./script/run_deqp_group_sample.sh <group-index> [n-cases] [tag]
#   ./script/run_deqp_group_sample.sh --list
#
#   group-index   1-based index into tools/deqp_groups.py GROUP_SPECS
#   n-cases       how many of the group's cases to sample (default 20)
#   tag           output subdirectory name (default g<index>)
#
# Case discovery is cached per module under <output root>/deqp_groups/discovery,
# because enumerating a module takes far longer than running a 20-case sample.
# Delete that file to re-discover.
#
# Paths come from config/local.env:
#   PVRGPU_UI_PYTHON     python used to import tools/deqp_groups.py
#   PVRGPU_OUTPUT_ROOT   run output root
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() { echo "run_deqp_group_sample: $*" >&2; exit 2; }

if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_DIR}/config/local.env"
    set +a
fi

PY="${PVRGPU_UI_PYTHON:-python3}"
command -v "${PY}" >/dev/null 2>&1 || die "python not found: ${PY} (set PVRGPU_UI_PYTHON in config/local.env)"

OUT_ROOT="${PVRGPU_OUTPUT_ROOT:-${REPO_DIR}/outputs}/deqp_groups"
CACHE="${OUT_ROOT}/discovery"

if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then
    "${PY}" -c "
import sys; sys.path.insert(0, '${REPO_DIR}/tools')
from deqp_groups import GROUP_SPECS
for i, g in enumerate(GROUP_SPECS, 1):
    print(f'{i:3d}  {g.suite:<12} {g.label}')
"
    exit $?
fi

IDX="${1:?usage: run_deqp_group_sample.sh <group-index> [n-cases] [tag]  (--list to enumerate)}"
N="${2:-20}"
TAG="${3:-g${IDX}}"
OUT="${OUT_ROOT}/${TAG}"

mkdir -p "${CACHE}" || die "cannot create ${CACHE}"

# ------------------------------------------------------------------------------
# Which suite does this group live in
# ------------------------------------------------------------------------------
read -r SUITE LABEL <<<"$("${PY}" -c "
import sys; sys.path.insert(0, '${REPO_DIR}/tools')
from deqp_groups import GROUP_SPECS
g = GROUP_SPECS[${IDX} - 1]
print(g.suite, g.label)
")" || die "no group at index ${IDX} (try --list)"

case "${SUITE}" in
    dEQP-EGL)    MOD=egl ;;
    dEQP-GLES2)  MOD=gles2 ;;
    dEQP-GLES3)  MOD=gles3 ;;
    dEQP-GLES31) MOD=gles31 ;;
    *) die "unknown suite ${SUITE}" ;;
esac

DISC="${CACHE}/${MOD}.txt"
if [[ ! -s "${DISC}" ]]; then
    echo "== discovering ${MOD} (one time, cached in ${DISC}) =="
    "${REPO_DIR}/script/run_deqp_dynamic.sh" --module "${MOD}" --discover \
        --caselist-out "${DISC}" >/dev/null 2>&1 \
        || die "discovery failed for ${MOD}"
fi

# ------------------------------------------------------------------------------
# Sample the group.  The driver APPENDS to its counter file, so re-running into
# a directory that already holds a run interleaves this run's events with the
# previous one's -- and reading those stale lines is a reliable way to reach a
# wrong diagnosis.  Start from an empty directory every time.
# ------------------------------------------------------------------------------
LIST="${OUT}/caselist.txt"
rm -rf "${OUT}"
mkdir -p "${OUT}" || die "cannot create ${OUT}"

"${PY}" -c "
import sys; sys.path.insert(0, '${REPO_DIR}/tools')
from deqp_groups import GROUP_SPECS, filter_exact_cases
g = GROUP_SPECS[${IDX} - 1]
all_cases = [l.strip() for l in open('${DISC}') if l.strip()]
sel = list(filter_exact_cases(g, all_cases))[:${N}]
if not sel:
    sys.exit(f'group ${IDX} matched none of the {len(all_cases)} discovered cases')
open('${LIST}', 'w').write('\n'.join(sel) + '\n')
print(f'group ${IDX}: {g.label}  ({len(all_cases)} discovered -> {len(sel)} sampled)')
" || die "case selection failed"

"${REPO_DIR}/script/run_deqp_dynamic.sh" \
    --surface-type pbuffer --size 256x256 --gl-config rgba8888d24s8ms0 \
    --log-images disable --keep-going \
    --caselist "${LIST}" --output-dir "${OUT}" >"${OUT}/run.log" 2>&1

[[ -s "${OUT}/summary.tsv" ]] || die "no summary.tsv; see ${OUT}/run.log"

# ------------------------------------------------------------------------------
# Report.  Statuses are dEQP's own, read back from the summary it wrote.
# ------------------------------------------------------------------------------
echo
echo "=== ${LABEL} ==="
awk -F'\t' 'NR>1 { c[$2]++ } END { for (k in c) printf "%-16s %d\n", k, c[k] }' \
    "${OUT}/summary.tsv" | sort

echo
echo "--- failures ---"
awk -F'\t' 'NR>1 && $2 != "Pass" && $2 != "NotSupported" { print $2"\t"$1"\t"$5 }' \
    "${OUT}/summary.tsv" |
while IFS=$'\t' read -r status case dir; do
    reason="$(grep -o '<Result StatusCode="[^"]*">[^<]*' "${dir}/results.qpa" 2>/dev/null |
              sed 's/.*">//' | head -1)"
    [[ -z "${reason}" ]] && reason="$(tail -20 "${dir}/run.log" 2>/dev/null |
              grep -iE 'error|fail|assert|abort' | head -1)"
    printf '%-14s %s\n    %s\n' "${status}" "${case#dEQP-}" "${reason:-<no reason>}"
done

echo
echo "run dir: ${OUT}"
