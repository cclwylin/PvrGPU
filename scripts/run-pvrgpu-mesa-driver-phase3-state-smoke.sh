#!/usr/bin/env bash
# Direct Mesa frontend -> PvrGPU Gallium driver Phase 3 state smoke test.
#
# This exercises GLES2 fixed-function state and an indexed triangle draw.
# It is counter-only: no pixels are compared because the driver still does not
# lower draws into a real model command stream.
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
Usage: ./scripts/run-pvrgpu-mesa-driver-phase3-state-smoke.sh [options]

Options:
  --mesa-prefix DIR  Mesa prefix containing libEGL/libGLESv2 and either
                     lib/dri/swrast_dri.dylib or lib/libgallium-*.dylib.
                     Default: PVRGPU_WORK_ROOT/mesa-pvrgpu/install
  --outdir DIR       Artifact directory. Default: timestamped dir under
                     PVRGPU_WORK_ROOT/out/pvrgpu-mesa-driver-smoke
  --size WxH         Pbuffer size for the draw. Default: 16x16
  --keep-temp        Keep the temporary C source and binary.

Expected result:
  driver-counter.txt contains fixed-function state events and
  event=draw_indexed_triangles.
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

if ! command -v cc >/dev/null 2>&1; then
    echo "Required command was not found: cc" >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${output_root}" == "${PVRGPU_WORK_ROOT%/}/out/pvrgpu-mesa-driver-smoke" ]]; then
    output_dir="${output_root}/phase3-state-smoke-${timestamp}-$$"
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

smoke_tmp="$(mktemp -d "${PVRGPU_TMP_ROOT%/}/pvrgpu-mesa-driver-phase3-state-smoke.XXXXXX")"
cleanup() {
    if [[ "${keep_temp}" == false && -n "${smoke_tmp:-}" &&
          "${smoke_tmp}" == "${PVRGPU_TMP_ROOT%/}"/pvrgpu-mesa-driver-phase3-state-smoke.* ]]; then
        rm -rf -- "${smoke_tmp}"
    fi
}
trap cleanup EXIT

smoke_source="${smoke_tmp}/pvrgpu_mesa_driver_phase3_state_smoke.c"
smoke_binary="${smoke_tmp}/pvrgpu_mesa_driver_phase3_state_smoke"
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

static int check_gl(const char *step) {
   GLenum error = glGetError();
   if (error != GL_NO_ERROR)
      return fail_gl(step, error);
   return 0;
}

static GLuint compile_shader(GLenum stage, const char *source) {
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);

   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (ok != GL_TRUE) {
      char log[4096];
      GLsizei written = 0;
      glGetShaderInfoLog(shader, sizeof(log), &written, log);
      fprintf(stderr, "shader compile failed: %.*s\n", written, log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static GLuint link_program(GLuint vs, GLuint fs) {
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glBindAttribLocation(program, 0, "a_position");
   glLinkProgram(program);

   GLint ok = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (ok != GL_TRUE) {
      char log[4096];
      GLsizei written = 0;
      glGetProgramInfoLog(program, sizeof(log), &written, log);
      fprintf(stderr, "program link failed: %.*s\n", written, log);
      glDeleteProgram(program);
      return 0;
   }
   return program;
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

   const char *vs_source =
      "attribute vec2 a_position;\n"
      "void main() {\n"
      "   gl_Position = vec4(a_position, 0.0, 1.0);\n"
      "}\n";
   const char *fs_source =
      "precision mediump float;\n"
      "void main() {\n"
      "   gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
      "}\n";

   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
   if (!vs || !fs)
      return 1;
   GLuint program = link_program(vs, fs);
   if (!program)
      return 1;

   const GLfloat vertices[] = {
      -1.0f, -1.0f,
       1.0f, -1.0f,
       0.0f,  1.0f,
   };
   const GLushort indices[] = {0, 1, 2};

   glViewport(0, 0, width, height);
   glEnable(GL_SCISSOR_TEST);
   glScissor(1, 2, width - 2, height - 3);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);
   glFrontFace(GL_CCW);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LEQUAL);
   glDepthMask(GL_TRUE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glBlendColor(0.1f, 0.2f, 0.3f, 0.4f);
   glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_TRUE);
   glStencilFunc(GL_ALWAYS, 3, 0xff);
   glStencilOp(GL_KEEP, GL_REPLACE, GL_INCR);
   if (check_gl("fixed-function state setup") != 0)
      return 1;

   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   if (check_gl("glClear") != 0)
      return 1;

   glUseProgram(program);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vertices);
   glEnableVertexAttribArray(0);
   glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, indices);
   glFinish();
   if (check_gl("glDrawElements") != 0)
      return 1;

   printf("PHASE3_STATE_DRAW=done\n");

   glDisableVertexAttribArray(0);
   glDeleteProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);
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

if [[ ! -s "${counter_out}" ]]; then
    echo "PvrGPU driver did not emit: ${counter_out}" >&2
    echo "stdout: ${stdout_log}" >&2
    echo "stderr: ${stderr_log}" >&2
    exit 1
fi

grep -Fx "PHASE3_STATE_DRAW=done" "${stdout_log}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=create_blend_state" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=bind_blend_state" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=create_depth_stencil_alpha_state" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=bind_depth_stencil_alpha_state" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=create_rasterizer_state" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=bind_rasterizer_state" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=set_blend_color" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=set_stencil_ref" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=set_scissor" "${counter_out}" >/dev/null
grep -F "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=draw_indexed_triangles" "${counter_out}" >/dev/null

if [[ "${keep_temp}" == true ]]; then
    printf 'temp_dir=%s\n' "${smoke_tmp}" >"${output_dir}/temp.txt"
fi

echo "PvrGPU Mesa driver Phase 3 state smoke PASS"
echo "artifacts=${output_dir}"
echo "counter=${counter_out}"
