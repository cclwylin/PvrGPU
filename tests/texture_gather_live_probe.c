/* SPDX-License-Identifier: MIT */
/* Standalone EGL/GLES 3.1 textureGather(sampler2D, uv, 0) live regression.
 * Build with clang -std=c11 -O2 -Wall -Wextra -Werror and the selected Mesa
 * EGL/GLES include/library paths, then run with an explicit backend/bridge.
 * No version/capability override, capture data or prepared reference output.
 *
 * Source textures are exact Z24 powers of two (plus zero/max); their upload
 * is independently checked by guarded depth readback. RGBA32UI preserves the
 * four gathered binary32 values without a color conversion. Clamp-edge tap
 * indices derive from binary32 coordinate arithmetic, independently of any
 * GPU output. These input values avoid unrelated normalization-rounding
 * differences while distinguishing every texel in each nontrivial texture.
 * 3 sizes x 4 min/mag choices x 28 coordinates = 336 actual draws.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 2, PIXELS = SIDE * SIDE, GUARD = 16, POINTS = 28, MAX_TEXELS = 16 };
static unsigned checks, passes;

static void check(int good, const char *message)
{
   ++checks;
   if (!good) {
      fprintf(stderr, "FAIL: %s\n", message);
      exit(1);
   }
}

static void check_gl(const char *where)
{
   const GLenum error = glGetError();
   if (error != GL_NO_ERROR)
      fprintf(stderr, "GL_ERROR: %s 0x%x\n", where, error);
   check(error == GL_NO_ERROR, where);
}

static uint32_t bits(float value)
{
   uint32_t result;
   memcpy(&result, &value, sizeof(result));
   return result;
}

static GLuint compile(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint ok = 0;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[8192];
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "SHADER: %s\n%s\n", log, source);
   }
   check(ok, "compile shader");
   return shader;
}

static GLuint make_program(void)
{
   const char *vs = "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "void main(){gl_Position=position;}\n";
   const char *fs = "#version 310 es\nprecision highp float; precision highp int;\n"
      "uniform highp sampler2D source_texture; uniform highp vec2 coordinate;\n"
      "layout(location=0) out highp uvec4 color;\n"
      "void main(){color=floatBitsToUint(textureGather(source_texture,coordinate,0));}\n";
   GLuint vertex = compile(GL_VERTEX_SHADER, vs), fragment = compile(GL_FRAGMENT_SHADER, fs);
   GLuint program = glCreateProgram();
   glAttachShader(program, vertex);
   glAttachShader(program, fragment);
   glDeleteShader(vertex);
   glDeleteShader(fragment);
   glLinkProgram(program);
   GLint ok = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[8192];
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "LINK: %s\n", log);
   }
   check(ok, "link component-zero gather program");
   return program;
}

static void indices(float coordinate, unsigned extent, unsigned *low, unsigned *high)
{
   volatile float scaled = coordinate * (float)extent;
   const float clamped = fminf((float)extent, fmaxf(0.f, scaled));
   volatile float minus_half = clamped - .5f, plus_half = clamped + .5f;
   const int first = (int)minus_half, second = (int)plus_half;
   *low = (unsigned)first;
   *high = second >= (int)extent ? extent - 1 : (unsigned)second;
   check(*low < extent && *high < extent, "CPU clamp-edge tap bounds");
}

static void oracle(const uint32_t *depth, unsigned width, unsigned height,
                   const float coordinate[2], uint32_t expected[4])
{
   unsigned x0, x1, y0, y1;
   indices(coordinate[0], width, &x0, &x1);
   indices(coordinate[1], height, &y0, &y1);
   const unsigned offsets[4] = {y1 * width + x0, y1 * width + x1,
                                y0 * width + x1, y0 * width + x0};
   for (unsigned component = 0; component < 4; ++component)
      expected[component] = bits((float)((double)depth[offsets[component]] / 16777215.0));
}

int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint config_attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
   const EGLint surface_attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   const EGLint context_attributes[] = {EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
   EGLConfig config = NULL;
   EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) &&
      eglChooseConfig(display, config_attributes, &config, 1, &count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display, surface, surface, context), "EGL current");
   fprintf(stderr, "RENDERER: %s\nVERSION: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   GLint major = 0, minor = 0;
   glGetIntegerv(GL_MAJOR_VERSION, &major);
   glGetIntegerv(GL_MINOR_VERSION, &minor);
   if (major < 3 || (major == 3 && minor < 1)) {
      fprintf(stderr, "UNSUPPORTED: public GLES 3.1 capability\n");
      return 2;
   }
   check_gl("public context version");
   GLuint program = make_program();
   glUseProgram(program);
   GLint sampler = glGetUniformLocation(program, "source_texture");
   GLint coordinate_location = glGetUniformLocation(program, "coordinate");
   check(sampler >= 0 && coordinate_location >= 0, "active gather uniforms");
   glUniform1i(sampler, 0);

   GLuint output_texture, input_texture, framebuffers[2], vao, vbo;
   glGenTextures(1, &output_texture);
   glBindTexture(GL_TEXTURE_2D, output_texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, SIDE, SIDE);
   glGenFramebuffers(2, framebuffers);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[0]);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output_texture, 0);
   const GLenum color = GL_COLOR_ATTACHMENT0, none = GL_NONE;
   glDrawBuffers(1, &color);
   glReadBuffer(color);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "RGBA32UI output FBO");
   const float vertices[] = {-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glViewport(0, 0, SIDE, SIDE);
   glDisable(GL_BLEND);
   glDisable(GL_CULL_FACE);
   glDisable(GL_DITHER);
   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_DEPTH_TEST);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   float coordinates[POINTS][2] = {
      {-1,-1}, {-1,.5f}, {.5f,-1}, {2,2}, {2,.5f}, {.5f,2},
      {0,0}, {1,1}, {0,1}, {1,0}, {.5f,.5f}, {.125f,.125f},
      {.375f,.625f}, {.875f,.875f}, {.5f,0}, {0,.5f}, {1,.5f}, {.5f,1},
      {-.125f,1.125f}, {1.125f,-.125f}, {.03125f,.96875f}, {.4375f,.5625f}
   };
   for (unsigned axis = 0; axis < 2; ++axis) {
      for (unsigned side = 0; side < 3; ++side) {
         const unsigned point = 22 + axis * 3 + side;
         coordinates[point][0] = coordinates[point][1] = .5f;
         coordinates[point][axis] = side == 0 ? nextafterf(.375f, -INFINITY) :
                                    side == 1 ? .375f : nextafterf(.375f, INFINITY);
      }
   }
   const unsigned sizes[3][2] = {{1,1}, {3,5}, {4,4}};
   const GLenum filters[4][2] = {{GL_NEAREST,GL_NEAREST}, {GL_NEAREST,GL_LINEAR},
                                {GL_LINEAR,GL_NEAREST}, {GL_LINEAR,GL_LINEAR}};
   for (unsigned shape = 0; shape < 3; ++shape) {
      const unsigned width = sizes[shape][0], height = sizes[shape][1], texels = width * height;
      uint32_t depth[MAX_TEXELS], upload[MAX_TEXELS];
      for (unsigned i = 0; i < texels; ++i) {
         depth[i] = UINT32_C(1) << ((i + 3 * shape) % 24);
         if (shape == 2 && i == 0) depth[i] = 0;
         if (shape == 2 && i == texels - 1) depth[i] = UINT32_C(0xffffff);
         /* Midpoint of the desired Z24 bin: GL may unpack UNORM32 through
          * float before reducing to Z24, so the lower bin boundary is not
          * an exact upload oracle. Zero and max remain exact endpoints. */
         upload[i] = depth[i] == 0 ? 0 : depth[i] == UINT32_C(0xffffff) ? UINT32_MAX :
                     (depth[i] << 8) | UINT32_C(0x80);
      }
      glGenTextures(1, &input_texture);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, input_texture);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, (GLsizei)width, (GLsizei)height);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)width, (GLsizei)height,
                      GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, upload);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[1]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, input_texture, 0);
      glDrawBuffers(1, &none);
      glReadBuffer(GL_NONE);
      check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "source Z24 FBO");
      uint32_t raw[GUARD + MAX_TEXELS + GUARD];
      for (unsigned i = 0; i < GUARD + MAX_TEXELS + GUARD; ++i) raw[i] = UINT32_C(0xdeadbeef);
      glReadPixels(0, 0, (GLsizei)width, (GLsizei)height, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, raw + GUARD);
      check_gl("Z24 upload readback");
      for (unsigned i = 0; i < GUARD; ++i)
         check(raw[i] == UINT32_C(0xdeadbeef) && raw[GUARD + texels + i] == UINT32_C(0xdeadbeef), "source readback guards");
      for (unsigned i = 0; i < texels; ++i) {
         const uint32_t expanded = (depth[i] << 8) | (depth[i] >> 16);
         if (raw[GUARD + i] != expanded)
            fprintf(stderr, "UPLOAD_MISMATCH: shape=%u pixel=%u actual=%08x expected=%08x depth24=%06x\n", shape, i, raw[GUARD + i], expanded, depth[i]);
         check(raw[GUARD + i] == expanded, "independent exact Z24 upload oracle");
         printf("UPLOAD %u %u %08x\n", shape, i, raw[GUARD + i]);
      }
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[0]);
      uint32_t first_filter[POINTS][PIXELS * 4];
      for (unsigned filter = 0; filter < 4; ++filter) {
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filters[filter][0]);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)filters[filter][1]);
         for (unsigned point = 0; point < POINTS; ++point) {
            const GLuint clear[4] = {0xdeadbeef,0xdeadbeef,0xdeadbeef,0xdeadbeef};
            glClearBufferuiv(GL_COLOR, 0, clear);
            glUniform2fv(coordinate_location, 1, coordinates[point]);
            check_gl("before gather draw");
            fprintf(stderr, "DRAW: shape=%u filter=%u point=%u\n", shape, filter, point);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            check_gl("gather draw/finish");
            uint32_t actual[GUARD + PIXELS * 4 + GUARD], expected[4];
            for (unsigned i = 0; i < GUARD + PIXELS * 4 + GUARD; ++i) actual[i] = UINT32_C(0xa5a5a5a5);
            glReadPixels(0, 0, SIDE, SIDE, GL_RGBA_INTEGER, GL_UNSIGNED_INT, actual + GUARD);
            check_gl("gather RGBA32UI readback");
            for (unsigned i = 0; i < GUARD; ++i)
               check(actual[i] == UINT32_C(0xa5a5a5a5) && actual[GUARD + PIXELS * 4 + i] == UINT32_C(0xa5a5a5a5), "gather readback guards");
            oracle(depth, width, height, coordinates[point], expected);
            printf("CASE %u %u %u %08x %08x\n", shape, filter, point, bits(coordinates[point][0]), bits(coordinates[point][1]));
            for (unsigned pixel = 0; pixel < PIXELS; ++pixel) {
               for (unsigned component = 0; component < 4; ++component) {
                  const unsigned index = pixel * 4 + component;
                  const uint32_t value = actual[GUARD + index];
                  if (value != expected[component])
                     fprintf(stderr, "MISMATCH: shape=%u filter=%u point=%u pixel=%u component=%u actual=%08x expected=%08x\n", shape, filter, point, pixel, component, value, expected[component]);
                  check(value == expected[component], "input-derived gather bits/order oracle");
                  if (filter == 0) first_filter[point][index] = value;
                  else check(value == first_filter[point][index], "gather min/mag filter invariance");
                  printf("GATHER %u %u %u %u %u %08x\n", shape, filter, point, pixel, component, value);
               }
            }
            ++passes;
            fprintf(stderr, "PASS: shape=%u filter=%u point=%u\n", shape, filter, point);
         }
      }
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[1]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
      glDeleteTextures(1, &input_texture);
   }
   check(passes == 336, "complete gather scenario inventory");
   glDeleteProgram(program);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteFramebuffers(2, framebuffers);
   glDeleteTextures(1, &output_texture);
   check_gl("probe cleanup");
   fprintf(stderr, "PASS: passes=%u checks=%u\n", passes, checks);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
