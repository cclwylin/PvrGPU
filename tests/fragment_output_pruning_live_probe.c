/* SPDX-License-Identifier: MIT */
/* Standalone live EGL/GLES 3.1 fragment-output specialization regression.
 * Build manually against the chosen Mesa EGL/GLES headers and libraries with
 * clang -std=c11 -O2 -Wall -Wextra -Werror; run the same binary with llvmpipe
 * and pvrgpu plus an explicit native bridge. No capability override is used.
 *
 * Each linked four-output program is reused across MRT4 -> MRT1 -> MRT4 ->
 * MRT1 -> depth-only. Initialized output3 feeds active color0 and FragDepth.
 * The first program additionally checks discard; the second has an R32UI
 * image atomic whose return feeds output2, including when output2 is unbound.
 * Fragment SSBO and image+discard are deliberately not required: the current
 * native contracts reject them. All expected values derive from GL inputs,
 * never prepared llvmpipe results. Readbacks have prefix/suffix guards.
 * stdout contains deterministic observed COLOR/DEPTH/IMAGE records only.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 4, PIXELS = SIDE * SIDE, GUARD = 16 };
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

static uint32_t float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

static int extension(const char *wanted)
{
   GLint count = 0;
   glGetIntegerv(GL_NUM_EXTENSIONS, &count);
   for (GLint i = 0; i < count; ++i) {
      const char *name = (const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i);
      if (name && strcmp(name, wanted) == 0)
         return 1;
   }
   return 0;
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

static GLuint make_program(unsigned image)
{
   const char *vs = "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "void main(){gl_Position=position;}\n";
   char fs[4096];
   const int length = snprintf(fs, sizeof(fs),
      "#version 310 es\n%s"
      "precision highp float; precision highp int;\n"
      "layout(location=0) out vec4 output0; layout(location=1) out vec4 output1;\n"
      "layout(location=2) out vec4 output2; layout(location=3) out vec4 output3;\n"
      "uniform int epoch; %s\n"
      "void main(){output3=vec4(0.25+0.25*float(epoch),0.5,0.25,1.0);\n"
      "output1=vec4(0.0,1.0,0.0,1.0); %s\n"
      "output0=vec4(output3.r,output3.g,output3.r+output3.b,1.0);\n"
      "gl_FragDepth=output3.r; %s }\n",
      image ? "#extension GL_OES_shader_image_atomic : require\n" : "",
      image ? "layout(r32ui,binding=0) uniform highp uimage2D writes;" : "uniform int discard_mode;",
      image ? "output2=vec4(float(imageAtomicAdd(writes,ivec2(gl_FragCoord.xy),uint(output3.r*4.0)))*0.0625,0.0,0.0,1.0);" :
              "output2=vec4(0.0,0.0,1.0,1.0);",
      image ? "" : "if(discard_mode!=0 && (int(gl_FragCoord.x)%2)==0)discard;");
   check(length > 0 && (size_t)length < sizeof(fs), "fragment source bound");
   GLuint program = glCreateProgram();
   GLuint vertex = compile(GL_VERTEX_SHADER, vs), fragment = compile(GL_FRAGMENT_SHADER, fs);
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
   check(ok, "link four-output program");
   for (unsigned target = 0; target < 4; ++target) {
      char name[16];
      snprintf(name, sizeof(name), "output%u", target);
      check(glGetFragDataLocation(program, name) == (GLint)target, "explicit fragment output location");
   }
   return program;
}

static void expected_color(unsigned family, unsigned epoch, unsigned target, uint8_t out[4])
{
   const uint8_t expected[4][4] = {
      {epoch ? 128 : 64, 128, epoch ? 191 : 128, 255},
      {0, 255, 0, 255},
      {family ? 64 : 0, 0, family ? 0 : 255, 255},
      {epoch ? 128 : 64, 128, 64, 255}
   };
   memcpy(out, expected[target], 4);
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
   fprintf(stderr, "RENDERER: %s\n", glGetString(GL_RENDERER));
   GLint draw_buffers = 0, color_attachments = 0, image_uniforms = 0;
   glGetIntegerv(GL_MAX_DRAW_BUFFERS, &draw_buffers);
   glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &color_attachments);
   glGetIntegerv(GL_MAX_FRAGMENT_IMAGE_UNIFORMS, &image_uniforms);
   if (draw_buffers < 4 || color_attachments < 4 || image_uniforms < 1 ||
       !extension("GL_OES_shader_image_atomic")) {
      fprintf(stderr, "UNSUPPORTED: public MRT4/fragment-image-atomic capability\n");
      return 2;
   }
   check_gl("public capability queries");
   GLuint programs[2] = {make_program(0), make_program(1)};
   GLuint textures[6], framebuffers[4], vao, vbo;
   glGenTextures(6, textures);
   for (unsigned i = 0; i < 6; ++i) {
      glBindTexture(GL_TEXTURE_2D, textures[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1,
         i < 4 ? GL_RGBA8 : i == 4 ? GL_DEPTH_COMPONENT32F : GL_R32UI, SIDE, SIDE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   }
   glGenFramebuffers(4, framebuffers);
   const GLenum mrt[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
   const GLenum none = GL_NONE;
   for (unsigned i = 0; i < 4; ++i) {
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[i]);
      if (i == 0) {
         for (unsigned target = 0; target < 4; ++target)
            glFramebufferTexture2D(GL_FRAMEBUFFER, mrt[target], GL_TEXTURE_2D, textures[target], 0);
         glDrawBuffers(4, mrt);
      } else if (i == 1 || i == 3) {
         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i == 1 ? 0 : 5], 0);
         glDrawBuffers(1, mrt);
      } else {
         glDrawBuffers(1, &none);
         glReadBuffer(GL_NONE);
      }
      if (i < 3)
         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, textures[4], 0);
      check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "complete exact-format FBO");
   }
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
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_ALWAYS);
   glDepthMask(GL_TRUE);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   check_gl("FBO/geometry setup");
   const unsigned target_counts[] = {4, 1, 4, 1, 0, 1, 4};
   const uint8_t clear_bytes[] = {255, 0, 255, 255};
   const GLfloat clear[] = {1, 0, 1, 1}, clear_depth = 1;
   for (unsigned family = 0; family < 2; ++family) {
      const unsigned pass_count = family ? 5 : 7;
      for (unsigned pass = 0; pass < pass_count; ++pass) {
         const unsigned targets = target_counts[pass], epoch = pass & 1U;
         const unsigned discard = !family && pass >= 5;
         glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[0]);
         for (unsigned target = 0; target < 4; ++target)
            glClearBufferfv(GL_COLOR, (GLint)target, clear);
         glClearBufferfv(GL_DEPTH, 0, &clear_depth);
         if (family) {
            uint32_t initial[PIXELS];
            for (unsigned i = 0; i < PIXELS; ++i)
               initial[i] = 4;
            glBindTexture(GL_TEXTURE_2D, textures[5]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SIDE, SIDE, GL_RED_INTEGER, GL_UNSIGNED_INT, initial);
            glBindImageTexture(0, textures[5], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
         }
         glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[targets == 4 ? 0 : targets == 1 ? 1 : 2]);
         glUseProgram(programs[family]);
         const GLint epoch_location = glGetUniformLocation(programs[family], "epoch");
         check(epoch_location >= 0, "active epoch uniform");
         glUniform1i(epoch_location, (GLint)epoch);
         if (!family) {
            const GLint discard_location = glGetUniformLocation(programs[family], "discard_mode");
            check(discard_location >= 0, "active discard uniform");
            glUniform1i(discard_location, (GLint)discard);
         }
         check_gl("before live draw");
         fprintf(stderr, "DRAW: family=%u pass=%u targets=%u discard=%u\n", family, pass, targets, discard);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);
         glFinish();
         check_gl("live draw/finish");
         glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[0]);
         for (unsigned target = 0; target < 4; ++target) {
            uint8_t actual[GUARD + PIXELS * 4 + GUARD];
            memset(actual, 0xa5, sizeof(actual));
            glReadBuffer(mrt[target]);
            glReadPixels(0, 0, SIDE, SIDE, GL_RGBA, GL_UNSIGNED_BYTE, actual + GUARD);
            check_gl("read color");
            for (unsigned i = 0; i < GUARD; ++i)
               check(actual[i] == 0xa5 && actual[GUARD + PIXELS * 4 + i] == 0xa5, "color readback guards");
            uint8_t expected[4];
            expected_color(family, epoch, target, expected);
            for (unsigned pixel = 0; pixel < PIXELS; ++pixel) {
               const int kept = target < targets && (!discard || (pixel % SIDE) % 2 != 0);
               check(!memcmp(actual + GUARD + pixel * 4, kept ? expected : clear_bytes, 4), "bound/unbound/discard color oracle");
               const uint8_t *value = actual + GUARD + pixel * 4;
               printf("COLOR %u %u %u %u %02x%02x%02x%02x\n", family, pass, target, pixel,
                  value[0], value[1], value[2], value[3]);
            }
         }
         uint32_t depth[GUARD + PIXELS + GUARD];
         for (unsigned i = 0; i < GUARD + PIXELS + GUARD; ++i)
            depth[i] = UINT32_C(0xdeadbeef);
         glReadPixels(0, 0, SIDE, SIDE, GL_DEPTH_COMPONENT, GL_FLOAT, depth + GUARD);
         check_gl("read D32F depth");
         for (unsigned i = 0; i < GUARD; ++i)
            check(depth[i] == UINT32_C(0xdeadbeef) && depth[GUARD + PIXELS + i] == UINT32_C(0xdeadbeef), "depth readback guards");
         for (unsigned pixel = 0; pixel < PIXELS; ++pixel) {
            const float expected = discard && (pixel % SIDE) % 2 == 0 ? 1.f : .25f + .25f * (float)epoch;
            check(depth[GUARD + pixel] == float_bits(expected), "output3-dependent exact depth/discard oracle");
            printf("DEPTH %u %u %u %08x\n", family, pass, pixel, depth[GUARD + pixel]);
         }
         if (family) {
            uint32_t image[GUARD + PIXELS + GUARD];
            for (unsigned i = 0; i < GUARD + PIXELS + GUARD; ++i)
               image[i] = UINT32_C(0xfedcba98);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[3]);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glReadPixels(0, 0, SIDE, SIDE, GL_RED_INTEGER, GL_UNSIGNED_INT, image + GUARD);
            check_gl("read atomic image");
            for (unsigned i = 0; i < GUARD; ++i)
               check(image[i] == UINT32_C(0xfedcba98) && image[GUARD + PIXELS + i] == UINT32_C(0xfedcba98), "image readback guards");
            for (unsigned pixel = 0; pixel < PIXELS; ++pixel) {
               check(image[GUARD + pixel] == 5 + epoch, "unbound-output atomic side effect retained exactly once");
               printf("IMAGE %u %u %u %08x\n", family, pass, pixel, image[GUARD + pixel]);
            }
         }
         ++passes;
         fprintf(stderr, "PASS: family=%u pass=%u targets=%u discard=%u\n", family, pass, targets, discard);
      }
   }
   check(passes == 12, "complete live scenario inventory");
   glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
   glDeleteProgram(programs[0]);
   glDeleteProgram(programs[1]);
   glDeleteBuffers(1, &vbo);
   glDeleteVertexArrays(1, &vao);
   glDeleteFramebuffers(4, framebuffers);
   glDeleteTextures(6, textures);
   check_gl("probe cleanup");
   fprintf(stderr, "PASS: passes=%u checks=%u\n", passes, checks);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
