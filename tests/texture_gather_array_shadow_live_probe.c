/* SPDX-License-Identifier: MIT */
/* Independent 2D-array shadow gather oracle. The four comparison results
 * remain binary32 bits in RGBA32UI. D16/D24 source uploads are checked exactly
 * on every layer before rendering. No capture/reference bytes are inputs.
 * 2 formats x 2 filter pairs x 8 compare functions x 14 cases = 448 draws.
 * Exact half-integer layer ties are intentionally excluded: that separate
 * implementation-specific rounding diagnostic must not define this oracle. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 2, PIXELS = SIDE * SIDE, GUARD = 16, POINTS = 14,
       WIDTH = 4, HEIGHT = 4, LAYERS = 3, TEXELS = WIDTH * HEIGHT };
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
      "uniform highp sampler2DArrayShadow source_texture; uniform highp vec3 coordinate;\n"
      "uniform highp float reference_depth;\n"
      "layout(location=0) out highp uvec4 color;\n"
      "void main(){color=floatBitsToUint(textureGather(source_texture,coordinate,reference_depth));}\n";
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
   check(ok, "link array-shadow gather program");
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

static void oracle(const uint32_t *depth, unsigned depth_bits,
                   const float coordinate[3], float reference,
                   unsigned function, uint32_t expected[4])
{
   unsigned x0, x1, y0, y1;
   indices(coordinate[0], WIDTH, &x0, &x1);
   indices(coordinate[1], HEIGHT, &y0, &y1);
   /* All test layers avoid exact halfway cases, so ordinary nearest rounding
    * and nearest-even have the same result. Clamping is independent of UV. */
   int layer = (int)floorf(coordinate[2] + .5f);
   if (layer < 0) layer = 0;
   if (layer >= LAYERS) layer = LAYERS - 1;
   reference = fminf(1.f, fmaxf(0.f, reference));
   const unsigned offsets[4] = {y1 * WIDTH + x0, y1 * WIDTH + x1,
                                y0 * WIDTH + x1, y0 * WIDTH + x0};
   const double denominator = depth_bits == 16 ? 65535.0 : 16777215.0;
   for (unsigned component = 0; component < 4; ++component) {
      const float value = (float)((double)depth[layer * TEXELS + offsets[component]] / denominator);
      int result = 0;
      switch (function) {
      case 0: result = 0; break;
      case 1: result = reference < value; break;
      case 2: result = reference == value; break;
      case 3: result = reference <= value; break;
      case 4: result = reference > value; break;
      case 5: result = reference != value; break;
      case 6: result = reference >= value; break;
      case 7: result = 1; break;
      }
      expected[component] = result ? UINT32_C(0x3f800000) : 0;
   }
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
   GLint reference_location = glGetUniformLocation(program, "reference_depth");
   check(sampler >= 0 && coordinate_location >= 0 && reference_location >= 0, "active gather uniforms");
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

   const float coordinates[POINTS][3] = {
      {.5f,.5f,0}, {.5f,.5f,1}, {.5f,.5f,2}, {.5f,.5f,-.75f},
      {.5f,.5f,3.25f}, {.5f,.5f,.49f}, {.5f,.5f,.51f},
      {.5f,.5f,1.49f}, {.5f,.5f,1.51f}, {-1,-1,0}, {2,2,2},
      {.375f,.625f,1}, {0,0,0}, {1,1,2}
   };
   const float references[POINTS] = {.5f,.5f,.5f,.5f,.5f,.5f,.5f,
      .5f,.5f,-.25f,1.25f,.25f,0,1};
   const GLenum filters[2] = {GL_NEAREST, GL_LINEAR};
   const GLenum functions[8] = {GL_NEVER,GL_LESS,GL_EQUAL,GL_LEQUAL,
      GL_GREATER,GL_NOTEQUAL,GL_GEQUAL,GL_ALWAYS};
   for (unsigned shape = 0; shape < 2; ++shape) {
      const unsigned depth_bits = shape == 0 ? 16 : 24;
      const uint32_t maximum = shape == 0 ? 65535 : 16777215;
      const uint32_t values[5] = {0, UINT32_C(1) << (depth_bits - 2),
         UINT32_C(1) << (depth_bits - 1), UINT32_C(3) << (depth_bits - 2), maximum};
      uint32_t depth[TEXELS * LAYERS], upload[TEXELS * LAYERS];
      uint16_t upload16[TEXELS * LAYERS];
      for (unsigned layer = 0; layer < LAYERS; ++layer)
         for (unsigned y = 0; y < HEIGHT; ++y) for (unsigned x = 0; x < WIDTH; ++x) {
            const unsigned i = layer * TEXELS + y * WIDTH + x;
            depth[i] = values[(x + 2 * y + layer) % 5];
            if (i == TEXELS * LAYERS - 1) depth[i] = maximum;
            upload16[i] = (uint16_t)depth[i];
            upload[i] = depth[i] == 0 ? 0 : depth[i] == maximum ? UINT32_MAX : (depth[i] << 8) | UINT32_C(0x80);
         }
      glGenTextures(1, &input_texture);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D_ARRAY, input_texture);
      glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, shape == 0 ? GL_DEPTH_COMPONENT16 : GL_DEPTH_COMPONENT24, WIDTH, HEIGHT, LAYERS);
      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, WIDTH, HEIGHT, LAYERS,
         GL_DEPTH_COMPONENT, shape == 0 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
         shape == 0 ? (const void *)upload16 : (const void *)upload);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[1]);
      glDrawBuffers(1, &none);
      glReadBuffer(GL_NONE);
      for (unsigned layer = 0; layer < LAYERS; ++layer) {
         glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, input_texture, 0, layer);
         check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "source layer depth FBO");
         uint32_t raw[GUARD + TEXELS + GUARD];
         for (unsigned i = 0; i < GUARD + TEXELS + GUARD; ++i) raw[i] = UINT32_C(0xdeadbeef);
         glReadPixels(0, 0, WIDTH, HEIGHT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, raw + GUARD);
         check_gl("depth array upload readback");
         for (unsigned i = 0; i < GUARD; ++i)
            check(raw[i] == UINT32_C(0xdeadbeef) && raw[GUARD + TEXELS + i] == UINT32_C(0xdeadbeef), "source readback guards");
         for (unsigned i = 0; i < TEXELS; ++i) {
            const uint32_t code = depth[layer * TEXELS + i];
            const uint32_t expanded = shape == 0 ? (code << 16) | code : (code << 8) | (code >> 16);
            if (raw[GUARD + i] != expanded) fprintf(stderr, "UPLOAD_MISMATCH format%u layer%u pixel%u actual%08x expected%08x\n", depth_bits, layer, i, raw[GUARD + i], expanded);
            check(raw[GUARD + i] == expanded, "independent exact depth upload oracle");
            printf("UPLOAD %u %u %u %08x\n", depth_bits, layer, i, raw[GUARD + i]);
         }
      }
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[0]);
      for (unsigned filter = 0; filter < 2; ++filter) for (unsigned function = 0; function < 8; ++function) {
         glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, (GLint)filters[filter]);
         glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, (GLint)filters[filter]);
         glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, (GLint)functions[function]);
         for (unsigned point = 0; point < POINTS; ++point) {
            const GLuint clear[4] = {0xdeadbeef,0xdeadbeef,0xdeadbeef,0xdeadbeef};
            glClearBufferuiv(GL_COLOR, 0, clear);
            glUniform3fv(coordinate_location, 1, coordinates[point]);
            glUniform1f(reference_location, references[point]);
            check_gl("before gather draw");
            fprintf(stderr, "DRAW: bits=%u filter=%u function=%u point=%u\n", depth_bits, filter, function, point);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            check_gl("gather draw/finish");
            uint32_t actual[GUARD + PIXELS * 4 + GUARD], expected[4];
            for (unsigned i = 0; i < GUARD + PIXELS * 4 + GUARD; ++i) actual[i] = UINT32_C(0xa5a5a5a5);
            glReadPixels(0, 0, SIDE, SIDE, GL_RGBA_INTEGER, GL_UNSIGNED_INT, actual + GUARD);
            check_gl("gather RGBA32UI readback");
            for (unsigned i = 0; i < GUARD; ++i)
               check(actual[i] == UINT32_C(0xa5a5a5a5) && actual[GUARD + PIXELS * 4 + i] == UINT32_C(0xa5a5a5a5), "gather readback guards");
            oracle(depth, depth_bits, coordinates[point], references[point], function, expected);
            for (unsigned pixel = 0; pixel < PIXELS; ++pixel) {
               for (unsigned component = 0; component < 4; ++component) {
                  const unsigned index = pixel * 4 + component;
                  const uint32_t value = actual[GUARD + index];
                  if (value != expected[component])
                     fprintf(stderr, "MISMATCH: bits=%u filter=%u function=%u point=%u pixel=%u component=%u actual=%08x expected=%08x\n", depth_bits, filter, function, point, pixel, component, value, expected[component]);
                  check(value == expected[component], "independent layer/tap/reference/compare oracle");
               }
            }
            printf("DRAW %u bits=%u filter=%u function=%u point=%u expected=%08x,%08x,%08x,%08x PASS\n",
               passes, depth_bits, filter, function, point, expected[0], expected[1], expected[2], expected[3]);
            fflush(stdout);
            ++passes;
         }
      }
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[1]);
      glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 0, 0, 0);
      glDeleteTextures(1, &input_texture);
   }
   check(passes == 448, "complete gather scenario inventory");
   glDeleteProgram(program);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteFramebuffers(2, framebuffers);
   glDeleteTextures(1, &output_texture);
   check_gl("probe cleanup");
   printf("PASS draws=%u checks=%u\n", passes, checks);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
