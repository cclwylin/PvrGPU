#!/usr/bin/env bash
# Build only the repository formal helper into a fresh private directory.
# Never changes installed RenderDoc, Mesa, the bridge, or local configuration.
set -euo pipefail
REPLAY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLAY_REPO_DIR="$(cd "${REPLAY_SCRIPT_DIR}/.." && pwd)"
if [[ $# != 2 ]]; then
  echo "Usage: $0 RENDERDOC_SOURCE_ROOT RENDERDOC_LIBRARY" >&2
  exit 2
fi
REPLAY_SOURCE="$(cd "$1" && pwd)"
REPLAY_LIBRARY="$2"
[[ -f "${REPLAY_SOURCE}/renderdoc/api/replay/renderdoc_replay.h" && -f "$REPLAY_LIBRARY" ]]
REPLAY_BUILD_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-rdc-player.XXXXXX")"
REPLAY_PNG_FLAGS=()
read -r -a REPLAY_PNG_FLAGS <<< "$(pkg-config --cflags --libs libpng)"
REPLAY_PLATFORM_FLAGS=()
case "$(uname -s)" in
  Darwin) REPLAY_PLATFORM_FLAGS+=(-DRENDERDOC_PLATFORM_APPLE) ;;
  Linux) REPLAY_PLATFORM_FLAGS+=(-DRENDERDOC_PLATFORM_LINUX -ldl) ;;
  *) echo "Unsupported build platform" >&2; exit 2 ;;
esac
"${CXX:-c++}" -std=c++17 -O2 -g -Wall -Wextra -Wpedantic \
  "${REPLAY_PLATFORM_FLAGS[@]}" \
  -isystem "${REPLAY_SOURCE}/renderdoc" -I "${REPLAY_REPO_DIR}/src" \
  "${REPLAY_REPO_DIR}/tools/renderdoc-mesa-player.cpp" \
  "${REPLAY_REPO_DIR}/src/support/png_writer.cpp" \
  "$REPLAY_LIBRARY" "${REPLAY_PNG_FLAGS[@]}" \
  -Wl,-rpath,"$(dirname "$REPLAY_LIBRARY")" \
  -o "${REPLAY_BUILD_OUT}/pvrgpu-rdc-player"
echo "${REPLAY_BUILD_OUT}/pvrgpu-rdc-player"
