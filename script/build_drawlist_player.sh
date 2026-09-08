#!/usr/bin/env bash
# Build the official checkpoint player privately; no installed runtime mutation.
set -euo pipefail
if [[ $# != 2 || "$1" == --help ]]; then
  echo "Usage: $0 RENDERDOC_SOURCE_ROOT SNAPSHOT_RENDERDOC_LIBRARY"
  [[ $# == 1 && "$1" == --help ]] && exit 0
  exit 2
fi
DRAWLIST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAWLIST_REPO_DIR="$(cd "${DRAWLIST_SCRIPT_DIR}/.." && pwd)"
DRAWLIST_SOURCE="$(cd "$1" && pwd)"
DRAWLIST_LIBRARY="$2"
[[ -f "${DRAWLIST_SOURCE}/renderdoc/api/replay/renderdoc_replay.h" && -f "$DRAWLIST_LIBRARY" ]]
DRAWLIST_BUILD_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-drawlist-player.XXXXXX")"
DRAWLIST_PLATFORM_FLAGS=()
case "$(uname -s)" in
  Darwin) DRAWLIST_PLATFORM_FLAGS+=(-DRENDERDOC_PLATFORM_APPLE) ;;
  Linux) DRAWLIST_PLATFORM_FLAGS+=(-DRENDERDOC_PLATFORM_LINUX -ldl) ;;
  *) echo "Unsupported build platform" >&2; exit 2 ;;
esac
"${CXX:-c++}" -std=c++17 -O2 -g -Wall -Wextra -Wpedantic \
  "${DRAWLIST_PLATFORM_FLAGS[@]}" \
  -isystem "${DRAWLIST_SOURCE}/renderdoc" -I "${DRAWLIST_REPO_DIR}/src" \
  "${DRAWLIST_REPO_DIR}/tools/renderdoc-drawlist-player.cpp" \
  "${DRAWLIST_REPO_DIR}/src/rdc_runner/native_report.cpp" \
  "${DRAWLIST_REPO_DIR}/src/rdc_runner/replay_range_audit.cpp" \
  "${DRAWLIST_REPO_DIR}/src/rdc_runner/sha256.cpp" \
  "$DRAWLIST_LIBRARY" -Wl,-rpath,"$(dirname "$DRAWLIST_LIBRARY")" \
  -o "${DRAWLIST_BUILD_OUT}/pvrgpu-drawlist-player"
echo "${DRAWLIST_BUILD_OUT}/pvrgpu-drawlist-player"
