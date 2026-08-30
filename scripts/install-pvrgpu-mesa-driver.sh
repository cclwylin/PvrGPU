#!/usr/bin/env bash
# Install the checked-in PvrGPU Gallium skeleton into a Mesa source tree and
# patch Mesa's Meson plumbing so GALLIUM_DRIVER=pvrgpu can select it.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

default_mesa_src="${PVRGPU_MESA_POC_SOURCE_DIR:-${PVRGPU_MESA_POC_ROOT:-${PVRGPU_WORK_ROOT%/}/mesa-poc}/src}"
mesa_src="${default_mesa_src}"
check_only=false

usage() {
    cat <<'EOF'
Usage: ./scripts/install-pvrgpu-mesa-driver.sh [--mesa-src DIR] [--check]

Copy src/gallium/drivers/pvrgpu into a Mesa source tree and patch Mesa Meson
files so the driver can be built with:

  -Dgallium-drivers=llvmpipe,softpipe,pvrgpu

At runtime, select it explicitly with:

  GALLIUM_DRIVER=pvrgpu

The default Mesa source tree is PVRGPU_MESA_POC_ROOT/src or
PVRGPU_WORK_ROOT/mesa-poc/src.
EOF
}

while (($# > 0)); do
    case "$1" in
        --mesa-src)
            if (($# < 2)); then
                echo "--mesa-src requires a directory" >&2
                exit 2
            fi
            mesa_src="$2"
            shift 2
            ;;
        --check)
            check_only=true
            shift
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

python3 - "$project_dir" "$mesa_src" "$check_only" <<'PY'
from __future__ import annotations

from pathlib import Path
import shutil
import sys

project_dir = Path(sys.argv[1]).resolve()
mesa_src = Path(sys.argv[2]).resolve()
check_only = sys.argv[3] == "true"

driver_src = project_dir / "src" / "gallium" / "drivers" / "pvrgpu"
driver_dst = mesa_src / "src" / "gallium" / "drivers" / "pvrgpu"

required_files = [
    mesa_src / "meson.options",
    mesa_src / "meson.build",
    mesa_src / "src" / "gallium" / "meson.build",
    mesa_src / "src" / "gallium" / "targets" / "dri" / "meson.build",
    mesa_src / "src" / "gallium" / "auxiliary" / "target-helpers" / "sw_helper.h",
]

if not driver_src.is_dir():
    raise SystemExit(f"PvrGPU driver source is missing: {driver_src}")
if not mesa_src.is_dir():
    raise SystemExit(f"Mesa source tree is missing: {mesa_src}")
for required in required_files:
    if not required.is_file():
        raise SystemExit(f"Mesa file is missing: {required}")


def apply_replace(
    path: Path,
    old: str,
    new: str,
    label: str,
    changes: dict[Path, str],
    marker: str | None = None,
) -> None:
    text = path.read_text(encoding="utf-8")
    if (marker or new) in text:
        print(f"{label}: already patched")
        return
    if old not in text:
        raise SystemExit(f"{label}: expected Mesa snippet not found in {path}")
    changes[path] = text.replace(old, new, 1)
    print(f"{label}: patch queued")


changes: dict[Path, str] = {}

apply_replace(
    mesa_src / "meson.options",
    "'lima', 'llvmpipe', 'nouveau', 'panfrost', 'r300', 'r600', 'radeonsi',",
    "'lima', 'llvmpipe', 'nouveau', 'panfrost', 'pvrgpu', 'r300', 'r600', 'radeonsi',",
    "meson.options gallium-drivers choice",
    changes,
    marker="'pvrgpu', 'r300'",
)

apply_replace(
    mesa_src / "meson.build",
    "'zink', 'd3d12', 'asahi', 'rocket', 'ethosu'",
    "'zink', 'd3d12', 'asahi', 'rocket', 'ethosu', 'pvrgpu'",
    "meson.build gallium-drivers=all list",
    changes,
    marker="'ethosu', 'pvrgpu'",
)

apply_replace(
    mesa_src / "meson.build",
    "with_gallium_panfrost = gallium_drivers.contains('panfrost')\n"
    "with_gallium_etnaviv = gallium_drivers.contains('etnaviv')",
    "with_gallium_panfrost = gallium_drivers.contains('panfrost')\n"
    "with_gallium_pvrgpu = gallium_drivers.contains('pvrgpu')\n"
    "with_gallium_etnaviv = gallium_drivers.contains('etnaviv')",
    "meson.build with_gallium_pvrgpu flag",
    changes,
    marker="with_gallium_pvrgpu = gallium_drivers.contains('pvrgpu')",
)

apply_replace(
    mesa_src / "src" / "gallium" / "meson.build",
    "if with_any_llvmpipe and with_gallium_softpipe\n"
    "  driver_swrast = declare_dependency(\n"
    "    dependencies : [ driver_softpipe, driver_llvmpipe ],\n"
    "  )\n"
    "elif with_any_llvmpipe\n"
    "  driver_swrast = driver_llvmpipe\n"
    "elif with_gallium_softpipe\n"
    "  driver_swrast = driver_softpipe\n"
    "else\n"
    "  driver_swrast = declare_dependency()\n"
    "endif\n",
    "if with_any_llvmpipe and with_gallium_softpipe\n"
    "  driver_swrast = declare_dependency(\n"
    "    dependencies : [ driver_softpipe, driver_llvmpipe ],\n"
    "  )\n"
    "elif with_any_llvmpipe\n"
    "  driver_swrast = driver_llvmpipe\n"
    "elif with_gallium_softpipe\n"
    "  driver_swrast = driver_softpipe\n"
    "else\n"
    "  driver_swrast = declare_dependency()\n"
    "endif\n"
    "if with_gallium_pvrgpu\n"
    "  subdir('drivers/pvrgpu')\n"
    "else\n"
    "  driver_pvrgpu = declare_dependency()\n"
    "endif\n",
    "gallium/meson.build pvrgpu subdir",
    changes,
    marker="subdir('drivers/pvrgpu')",
)

apply_replace(
    mesa_src / "src" / "gallium" / "targets" / "dri" / "meson.build",
    "driver_swrast, driver_r300, driver_r600, driver_radeonsi, driver_nouveau,",
    "driver_swrast, driver_pvrgpu, driver_r300, driver_r600, driver_radeonsi, driver_nouveau,",
    "targets/dri/meson.build pvrgpu dependency",
    changes,
    marker="driver_swrast, driver_pvrgpu",
)

apply_replace(
    mesa_src / "src" / "gallium" / "auxiliary" / "target-helpers" / "sw_helper.h",
    "#ifdef GALLIUM_D3D12\n"
    "#include \"d3d12/d3d12_public.h\"\n"
    "#endif\n",
    "#ifdef GALLIUM_D3D12\n"
    "#include \"d3d12/d3d12_public.h\"\n"
    "#endif\n"
    "\n"
    "#ifdef GALLIUM_PVRGPU\n"
    "#include \"pvrgpu/pvrgpu_public.h\"\n"
    "#endif\n",
    "sw_helper.h pvrgpu include",
    changes,
    marker='#include "pvrgpu/pvrgpu_public.h"',
)

apply_replace(
    mesa_src / "src" / "gallium" / "auxiliary" / "target-helpers" / "sw_helper.h",
    "#if defined(GALLIUM_LLVMPIPE)\n"
    "   if (screen == NULL && (strcmp(driver, \"llvmpipe\") == 0 || !driver[0]))\n"
    "      screen = llvmpipe_create_screen(winsys);\n"
    "#endif\n",
    "#if defined(GALLIUM_PVRGPU)\n"
    "   if (screen == NULL && strcmp(driver, \"pvrgpu\") == 0)\n"
    "      screen = pvrgpu_create_screen(winsys, config);\n"
    "#endif\n"
    "\n"
    "#if defined(GALLIUM_LLVMPIPE)\n"
    "   if (screen == NULL && (strcmp(driver, \"llvmpipe\") == 0 || !driver[0]))\n"
    "      screen = llvmpipe_create_screen(winsys);\n"
    "#endif\n",
    "sw_helper.h GALLIUM_DRIVER=pvrgpu selection",
    changes,
    marker='strcmp(driver, "pvrgpu") == 0',
)

if check_only:
    print("CHECK PASS: Mesa PvrGPU integration can be applied")
    raise SystemExit(0)

if driver_dst.exists():
    if driver_dst.parent != mesa_src / "src" / "gallium" / "drivers":
        raise SystemExit(f"Refusing to replace unexpected driver directory: {driver_dst}")
    shutil.rmtree(driver_dst)
shutil.copytree(driver_src, driver_dst)
print(f"copied driver: {driver_src} -> {driver_dst}")

for path, new_text in changes.items():
    path.write_text(new_text, encoding="utf-8")

print(f"PvrGPU Mesa driver integration installed in: {mesa_src}")
PY
