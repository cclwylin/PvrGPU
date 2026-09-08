#!/usr/bin/env bash
# Isolated snapshot-extension build; never invokes the shared Ninja build.
set -euo pipefail
SNAPSHOT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SNAPSHOT_SCRIPT_DIR}/build_renderdoc_snapshot.py" "$@"
