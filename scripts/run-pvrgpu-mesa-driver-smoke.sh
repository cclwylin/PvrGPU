#!/usr/bin/env bash
# Direct Mesa frontend -> PvrGPU Gallium driver smoke test.
#
# This intentionally avoids RenderDoc, RDC translation, and the SystemC model.
# It proves that Mesa can load the pvrgpu Gallium driver through swrast/DRI and
# that a plain GLES2 glClear() reaches src/gallium/drivers/pvrgpu.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_config="${project_dir}/config/local.env"
if [[ -f "${local_config}" ]]; then
    # shellcheck source=/dev/null
    source "${local_config}"
fi
# shellcheck source=scripts/lib/runtime-paths.sh
source "${project_dir}/scripts/lib/runtime-paths.sh"

mesa_prefix="${PVRGPU_MESA_PVRGPU_PREFIX:-${PVRGPU_WORK_ROOT%/}/mesa-pvrgpu/install}"
output_root="${PVRGPU_MESA_DRIVER_SMOKE_OUTPUT:-${PVRGPU_WORK_ROOT%/}/out/pvrgpu-mesa-driver-smoke}"
size="16x16"
keep_temp=false

usage() {
    cat <<'EOF'
Usage: ./scripts/run-pvrgpu-mesa-driver-smoke.sh [options]

Options:
  --mesa-prefix DIR  Mesa prefix containing libEGL/libGLESv2 and either
                     lib/dri/swrast_dri.dylib or lib/libgallium-*.dylib.
                     Default: PVRGPU_WORK_ROOT/mesa-pvrgpu/install
  --outdir DIR       Artifact directory. Default: timestamped dir under
                     PVRGPU_WORK_ROOT/out/pvrgpu-mesa-driver-smoke
  --size WxH         Pbuffer size for the clear. Default: 16x16
  --keep-temp        Keep the temporary C source and binary.

Expected result:
  The smoke creates driver-command.txt with schema=pvrgpu.driver-command.v1.
EOF
}

while (($# > 0)); do
    case "$1" in
        --mesa-prefix)
            (($# >= 2)) || { echo "--mesa-prefix requires a directory" >&2; exit 2; }
            mesa_prefix="$2"
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
        --keep-temp)
            keep_temp=true
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

if [[ ! "${size}" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
    echo "--size must be WIDTHxHEIGHT with positive integers" >&2
    exit 2
fi
width="${BASH_REMATCH[1]}"
height="${BASH_REMATCH[2]}"

for required_runtime in \
    "${mesa_prefix}/lib/libEGL.dylib" \
    "${mesa_prefix}/lib/libGLESv2.dylib"
do
    if [[ ! -f "${required_runtime}" ]]; then
        echo "Required Mesa runtime is missing: ${required_runtime}" >&2
        echo "Hint: run scripts/check-pvrgpu-mesa-driver-build.sh --platforms macos --full-dri --install." >&2
        exit 1
    fi
done

for command_name in cc; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command was not found: ${command_name}" >&2
        exit 1
    fi
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${output_root}" == "${PVRGPU_WORK_ROOT%/}/out/pvrgpu-mesa-driver-smoke" ]]; then
    output_dir="${output_root}/smoke-${timestamp}-$$"
else
    output_dir="${output_root}"
fi
cache_dir="${output_dir}/xdg-cache"
driver_dir="${output_dir}/dri"
mkdir -p "${output_dir}" "${cache_dir}" "${driver_dir}" "${PVRGPU_TMP_ROOT%/}"

if [[ -f "${mesa_prefix}/lib/dri/swrast_dri.dylib" ]]; then
    driver_search_path="${mesa_prefix}/lib/dri"
else
    gallium_driver=""
    while IFS= read -r candidate; do
        gallium_driver="${candidate}"
        break
    done < <(find "${mesa_prefix}/lib" -maxdepth 1 -name 'libgallium-*.dylib' -type f | sort)
    if [[ -z "${gallium_driver}" ]]; then
        echo "Required Mesa DRI driver is missing under: ${mesa_prefix}/lib" >&2
        echo "Expected lib/dri/swrast_dri.dylib or lib/libgallium-*.dylib." >&2
        exit 1
    fi
    ln -sf "${gallium_driver}" "${driver_dir}/swrast_dri.dylib"
    driver_search_path="${driver_dir}"
fi

smoke_tmp="$(mktemp -d "${PVRGPU_TMP_ROOT%/}/pvrgpu-mesa-driver-smoke.XXXXXX")"
cleanup() {
    if [[ "${keep_temp}" == false && -n "${smoke_tmp:-}" &&
          "${smoke_tmp}" == "${PVRGPU_TMP_ROOT%/}"/pvrgpu-mesa-driver-smoke.* ]]; then
        rm -rf -- "${smoke_tmp}"
    fi
}
trap cleanup EXIT

smoke_source="${smoke_tmp}/pvrgpu_mesa_driver_smoke.c"
smoke_binary="${smoke_tmp}/pvrgpu_mesa_driver_smoke"
command_out="${output_dir}/driver-command.txt"
counter_out="${output_dir}/driver-counter.txt"
stdout_log="${output_dir}/stdout.log"
stderr_log="${output_dir}/stderr.log"
rm -f -- "${command_out}" "${counter_out}" "${stdout_log}" "${stderr_log}"

cat >"${smoke_source}" <<'EOF'
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <stdio.h>
#include <stdlib.h>

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif

static int fail_egl(const char *step) {
   fprintf(stderr, "%s failed: egl_error=0x%04x\n", step, eglGetError());
   return 1;
}

static int fail_gl(const char *step, GLenum error) {
   fprintf(stderr, "%s failed: gl_error=0x%04x\n", step, error);
   return 1;
}

int main(int argc, char **argv) {
   const int width = argc > 1 ? atoi(argv[1]) : 16;
   const int height = argc > 2 ? atoi(argv[2]) : 16;
   if (width <= 0 || height <= 0)
      return 2;

   EGLDisplay display = EGL_NO_DISPLAY;
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   if (get_platform_display) {
      display = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA,
                                     NULL, NULL);
   }
   if (display == EGL_NO_DISPLAY)
      display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (display == EGL_NO_DISPLAY)
      return fail_egl("eglGetDisplay");
   if (eglInitialize(display, NULL, NULL) != EGL_TRUE)
      return fail_egl("eglInitialize");
   if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE)
      return fail_egl("eglBindAPI");

   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE
   };
   EGLConfig config = NULL;
   EGLint config_count = 0;
   if (eglChooseConfig(display, config_attributes, &config, 1,
                       &config_count) != EGL_TRUE || config_count < 1)
      return fail_egl("eglChooseConfig");

   const EGLint surface_attributes[] = {
      EGL_WIDTH, width,
      EGL_HEIGHT, height,
      EGL_NONE
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, surface_attributes);
   if (surface == EGL_NO_SURFACE)
      return fail_egl("eglCreatePbufferSurface");

   const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE
   };
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (context == EGL_NO_CONTEXT)
      return fail_egl("eglCreateContext");
   if (eglMakeCurrent(display, surface, surface, context) != EGL_TRUE)
      return fail_egl("eglMakeCurrent");

   printf("GL_VENDOR=%s\n", glGetString(GL_VENDOR));
   printf("GL_RENDERER=%s\n", glGetString(GL_RENDERER));
   printf("GL_VERSION=%s\n", glGetString(GL_VERSION));

   glViewport(0, 0, width, height);
   glClearColor(0.125f, 0.25f, 0.5f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();

   GLenum error = glGetError();
   if (error != GL_NO_ERROR)
      return fail_gl("glClear", error);

   unsigned char pixel[4] = {0, 0, 0, 0};
   glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
   error = glGetError();
   if (error != GL_NO_ERROR)
      return fail_gl("glReadPixels", error);
   printf("CLEAR_PIXEL=%u,%u,%u,%u\n",
          pixel[0], pixel[1], pixel[2], pixel[3]);
   if (pixel[0] != 32 || pixel[1] != 64 ||
       pixel[2] != 128 || pixel[3] != 255) {
      fprintf(stderr, "unexpected clear pixel: %u,%u,%u,%u\n",
              pixel[0], pixel[1], pixel[2], pixel[3]);
      return 1;
   }

   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
EOF

cc="${CC:-cc}"
"${cc}" \
    -std=c11 -Wall -Wextra \
    -I"${mesa_prefix}/include" \
    "${smoke_source}" \
    -L"${mesa_prefix}/lib" -lEGL -lGLESv2 \
    "-Wl,-rpath,${mesa_prefix}/lib" \
    -o "${smoke_binary}"

env \
    LC_ALL=C \
    LANG=C \
    TZ=UTC \
    TMPDIR="${smoke_tmp}" \
    XDG_CACHE_HOME="${cache_dir}" \
    EGL_PLATFORM=surfaceless \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    GALLIUM_DRIVER=pvrgpu \
    LIBGL_DRIVERS_PATH="${driver_search_path}" \
    MESA_SHADER_CACHE_DISABLE=true \
    PVRGPU_DRIVER_COMMAND_OUT="${command_out}" \
    PVRGPU_DRIVER_COUNTER_OUT="${counter_out}" \
    DYLD_LIBRARY_PATH="${mesa_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${smoke_binary}" "${width}" "${height}" \
    >"${stdout_log}" \
    2>"${stderr_log}"

if [[ ! -s "${command_out}" ]]; then
    echo "PvrGPU driver did not emit: ${command_out}" >&2
    echo "stdout: ${stdout_log}" >&2
    echo "stderr: ${stderr_log}" >&2
    exit 1
fi

grep -Fx "schema=pvrgpu.driver-command.v1" "${command_out}" >/dev/null
grep -Fx "producer=pvrgpu-gallium-driver" "${command_out}" >/dev/null
grep -Fx "command=clear_color" "${command_out}" >/dev/null
grep -Fx "width=${width}" "${command_out}" >/dev/null
grep -Fx "height=${height}" "${command_out}" >/dev/null
grep -Fx "format=PIPE_FORMAT_R8G8B8A8_UNORM" "${command_out}" >/dev/null
grep -Fx "CLEAR_PIXEL=32,64,128,255" "${stdout_log}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=clear_color" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=texture_map" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=texture_unmap" "${counter_out}" >/dev/null

if [[ "${keep_temp}" == true ]]; then
    printf 'temp_dir=%s\n' "${smoke_tmp}" >"${output_dir}/temp.txt"
fi

echo "PvrGPU Mesa driver direct smoke PASS"
echo "artifacts=${output_dir}"
echo "command=${command_out}"
echo "counter=${counter_out}"
