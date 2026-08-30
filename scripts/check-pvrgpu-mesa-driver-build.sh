#!/usr/bin/env bash
# Configure a Mesa build with the PvrGPU Gallium driver enabled and compile the
# smallest targets that prove Mesa can select and compile the driver seam.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

pvrgpu_mesa_root="${PVRGPU_MESA_POC_ROOT:-${PVRGPU_WORK_ROOT%/}/mesa-poc}"
mesa_src="${PVRGPU_MESA_POC_SOURCE_DIR:-${pvrgpu_mesa_root}/src}"
build_dir="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_WORK_ROOT%/}/build/mesa-pvrgpu}"
prefix_dir="${PVRGPU_MESA_PVRGPU_PREFIX:-${PVRGPU_WORK_ROOT%/}/mesa-pvrgpu/install}"
native_file="${PVRGPU_MESA_NATIVE_FILE:-$(cd "${project_dir}/.." && pwd)/BenchScope/scripts/mesa-arch-native.ini}"
mesa_platforms="${PVRGPU_MESA_PVRGPU_PLATFORMS:-x11,macos}"
full_dri=false
install_prefix=false

usage() {
    cat <<'EOF'
Usage: ./scripts/check-pvrgpu-mesa-driver-build.sh [options]

Options:
  --mesa-src DIR      Mesa source tree. Default: PVRGPU_WORK_ROOT/mesa-poc/src
  --build-dir DIR     Mesa build dir. Default: PVRGPU_WORK_ROOT/build/mesa-pvrgpu
  --prefix DIR        Mesa install prefix. Default: PVRGPU_WORK_ROOT/mesa-pvrgpu/install
  --native-file FILE  Meson native file. Default: ../BenchScope/scripts/mesa-arch-native.ini
  --platforms LIST    Mesa -Dplatforms value. Default: x11,macos
  --full-dri          Also attempt the full gallium DRI shared-library build.
  --install           Install the configured Mesa prefix after successful build.

Default verification builds:
  1. pvrgpu static library
  2. DRI target object that includes sw_helper.h
  3. pipe_loader_static

The default avoids Mesa's full GLSL parser generation path. Use --full-dri for
a stronger check when a modern bison is available.

For direct surfaceless driver smoke on macOS, use:

  ./scripts/check-pvrgpu-mesa-driver-build.sh --platforms macos --full-dri --install
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
        --build-dir)
            if (($# < 2)); then
                echo "--build-dir requires a directory" >&2
                exit 2
            fi
            build_dir="$2"
            shift 2
            ;;
        --prefix)
            if (($# < 2)); then
                echo "--prefix requires a directory" >&2
                exit 2
            fi
            prefix_dir="$2"
            shift 2
            ;;
        --native-file)
            if (($# < 2)); then
                echo "--native-file requires a file" >&2
                exit 2
            fi
            native_file="$2"
            shift 2
            ;;
        --platforms)
            if (($# < 2)); then
                echo "--platforms requires a comma-separated Mesa platform list" >&2
                exit 2
            fi
            mesa_platforms="$2"
            shift 2
            ;;
        --full-dri)
            full_dri=true
            shift
            ;;
        --install)
            install_prefix=true
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

"${project_dir}/scripts/install-pvrgpu-mesa-driver.sh" --mesa-src "${mesa_src}"

mkdir -p "$(dirname "${build_dir}")" "$(dirname "${prefix_dir}")"

pvrgpu_extra_tool_paths=()
for candidate in /usr/local/opt/bison/bin /opt/homebrew/opt/bison/bin; do
    if [[ -x "${candidate}/bison" ]]; then
        pvrgpu_extra_tool_paths+=("${candidate}")
    fi
done

pvrgpu_tool_path="${pvrgpu_mesa_root}/venv/bin:/usr/local/opt/llvm/bin:${pvrgpu_extra_tool_paths[*]:-}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
pvrgpu_tool_path="${pvrgpu_tool_path// /:}"
export PATH="${pvrgpu_tool_path}"
export CC="${CC:-/usr/local/opt/llvm/bin/clang}"
export CXX="${CXX:-/usr/local/opt/llvm/bin/clang++}"

meson_common_args=(
    "-Dbuildtype=release"
    "-Dprefix=${prefix_dir}"
    "-Dlibdir=lib"
    "-Db_ndebug=true"
    "-Dplatforms=${mesa_platforms}"
    "-Dxlib-lease=disabled"
    "-Degl=enabled"
    "-Degl-native-platform=surfaceless"
    "-Dgles1=disabled"
    "-Dgles2=enabled"
    "-Dglx=disabled"
    "-Dgbm=disabled"
    "-Dgallium-drivers=llvmpipe,zink,pvrgpu"
    "-Dmoltenvk-dir=/usr/local/opt/molten-vk"
    "-Dgallium-rusticl=false"
    "-Dllvm=enabled"
    "-Dopengl=true"
    "-Dvideo-codecs="
    "-Dtools=dlclose-skip"
    "-Dvulkan-drivers="
    "-Dc_link_args=/usr/local/opt/libxcb/lib/libxcb-present.dylib /usr/local/opt/libxcb/lib/libxcb-shm.dylib"
    "-Dcpp_link_args=/usr/local/opt/libxcb/lib/libxcb-present.dylib /usr/local/opt/libxcb/lib/libxcb-shm.dylib"
    "-Dglvnd=disabled"
    "-Ddisplay-info=disabled"
)

if [[ -f "${native_file}" ]]; then
    meson_common_args+=("--native-file=${native_file}")
fi

if [[ -f "${build_dir}/meson-private/coredata.dat" ]]; then
    meson setup --reconfigure "${build_dir}" "${mesa_src}" "${meson_common_args[@]}"
else
    meson setup "${build_dir}" "${mesa_src}" "${meson_common_args[@]}"
fi

meson compile -C "${build_dir}" pvrgpu
ninja -C "${build_dir}" src/gallium/targets/dri/libgallium-26.2.1.dylib.p/dri_target.c.o
meson compile -C "${build_dir}" pipe_loader_static

if [[ "${full_dri}" == true ]]; then
    ninja -C "${build_dir}" src/gallium/targets/dri/libgallium-26.2.1.dylib
fi

if [[ "${install_prefix}" == true ]]; then
    meson install -C "${build_dir}"
fi

echo "PvrGPU Mesa driver build seam PASS: ${build_dir}"
