#!/usr/bin/env bash
# Private standalone checks; no Mesa, RenderDoc, bridge or shared build mutation.
set -euo pipefail
DEBUG_SCOPE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# != 0 && $# != 2 ]]; then
  echo "Usage: $0 [MESA_EGL_LIBRARY MESA_GLES_LIBRARY]" >&2
  exit 2
fi
DEBUG_SCOPE_BUILD="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/rdc-gl-debug-scope.XXXXXX")"
DEBUG_SCOPE_FLAGS=(-std=c++17)
case "$(uname -s)" in Linux) DEBUG_SCOPE_FLAGS+=(-ldl) ;; esac
if [[ "${PVRGPU_TEST_SANITIZERS:-0}" == 1 ]]; then
  DEBUG_SCOPE_FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"${CXX:-c++}" -O1 -g -Wall -Wextra -Wpedantic -Werror \
  -I "$DEBUG_SCOPE_REPO/src" "$DEBUG_SCOPE_REPO/tests/rdc_gl_debug_scope_test.cpp" \
  "${DEBUG_SCOPE_FLAGS[@]}" -o "$DEBUG_SCOPE_BUILD/rdc-gl-debug-scope-test"
"$DEBUG_SCOPE_BUILD/rdc-gl-debug-scope-test" "$@"
echo "Private test executable: $DEBUG_SCOPE_BUILD/rdc-gl-debug-scope-test"
