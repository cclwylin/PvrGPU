/* SPDX-License-Identifier: MIT */
/* Standalone EGL/GLES 3.1 multisample texelFetch A/B probe.
 *
 * Every observed DWORD is produced by a real draw and shader texelFetch.
 * Per-sample values are written by ordinary fragment shaders with the GL
 * sample mask, not gl_SampleID, a CPU texture upload, or a resolve. The host
 * computes test oracles only. Border pixels and a second partial update test
 * preservation of other samples/layers. No out-of-bounds GL fetch is used:
 * unspecified results cannot be compared between backends.
 *
 * Usage: multisample-texture-live-probe [samples=4] [2d|array|both=both]
 * An unsupported requested format/count/extension is reported and returns 2;
 * it is never counted as a Pass. stdout is raw, per-channel observed bits.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 6, HEIGHT = 5, ARRAY_LAYERS = 3, FORMAT_COUNT = 6 };
enum kind { UNORM8, SINT8, UINT8, DEPTH32, SINT32, UINT32 };
static const GLenum formats[] = {
   GL_R8, GL_R8I, GL_R8UI, GL_DEPTH_COMPONENT32F, GL_R32I, GL_R32UI
};
static const char *const names[] = {
   "r8", "r8i", "r8ui", "depth32f", "r32i", "r32ui"
};
static unsigned failures, unsupported, scenarios, reads, draws, checks;
static PFNGLTEXSTORAGE3DMULTISAMPLEOESPROC storage3d;

static int
check_gl(const char *where)
{
   GLenum error;
   int ok = 1;
   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "GL_ERROR\t%s\t0x%x\n", where, error);
      ++failures;
      ok = 0;
   }
   return ok;
}

static uint32_t
float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

static float
bits_float(uint32_t bits)
{
   float value;
   memcpy(&value, &bits, sizeof(value));
   return value;
}

static int
is_signed(enum kind kind)
{
   return kind == SINT8 || kind == SINT32;
}

static int
is_integer(enum kind kind)
{
   return is_signed(kind) || kind == UINT8 || kind == UINT32;
}

static uint32_t
value_bits(enum kind kind, unsigned layer, unsigned sample, unsigned epoch,
           int border)
{
   const unsigned index = layer * 16u + sample;
   if (kind == UNORM8)
      return float_bits((float)(border ? 7u : 20u + 3u * index + epoch) / 255.0f);
   if (kind == DEPTH32)
      return float_bits(border ? 0.9375f : (float)(8u + index + epoch) / 64.0f);
   if (kind == SINT8)
      return (uint32_t)(border ? -127 : -100 + (int)(index * 3u + epoch));
   if (kind == UINT8)
      return border ? 251u : 11u + index * 3u + epoch;
   if (kind == SINT32)
      return border ? UINT32_C(0x80000007) :
             ((index & 1u) ? UINT32_C(0x80000000) + index + epoch :
                              UINT32_C(0x7fffffff) - index - epoch);
   return border ? UINT32_C(0xfedcba98) :
          ((index & 1u) ? index + epoch : UINT32_MAX - index - epoch);
}

static GLuint
program_for(const char *fragment)
{
   static const char *vertex =
      "#version 310 es\nlayout(location=0) in highp vec2 position;\n"
      "void main(){ gl_Position=vec4(position,0.0,1.0); }\n";
   const char *sources[] = {vertex, fragment};
   const GLenum stages[] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   GLuint program = glCreateProgram();
   for (unsigned stage = 0; stage < 2; ++stage) {
      GLuint shader = glCreateShader(stages[stage]);
      GLint ok = 0;
      glShaderSource(shader, 1, &sources[stage], NULL);
      glCompileShader(shader);
      glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
      if (!ok) {
         char log[8192];
         glGetShaderInfoLog(shader, sizeof(log), NULL, log);
         fprintf(stderr, "SHADER_ERROR\tstage=%u\n%s\nSOURCE\n%s\n",
                 stage, log, sources[stage]);
         ++failures;
         glDeleteShader(shader);
         glDeleteProgram(program);
         return 0;
      }
      glAttachShader(program, shader);
      glDeleteShader(shader);
   }
   glLinkProgram(program);
   GLint ok = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[8192];
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "LINK_ERROR\n%s\n", log);
      ++failures;
      glDeleteProgram(program);
      return 0;
   }
   return program;
}

static GLuint
writer_program(enum kind kind)
{
   char source[1024];
   const char *type = is_signed(kind) ? "ivec4" : is_integer(kind) ? "uvec4" : "vec4";
   snprintf(source, sizeof(source),
            "#version 310 es\nprecision highp float; precision highp int;\n"
            "uniform highp %s value; layout(location=0) out highp %s color;\n"
            "void main(){ color=value; %s }\n", type, type,
            kind == DEPTH32 ? "gl_FragDepth=value.r;" : "");
   return program_for(source);
}

static GLuint
reader_program(enum kind kind, int array)
{
   char source[2048];
   snprintf(source, sizeof(source),
            "#version 310 es\n%s"
            "precision highp float; precision highp int;\n"
            "uniform highp %ssampler2DMS%s source_texture;\n"
            "uniform highp int sample_index, layer_index;\n"
            "layout(location=0) out highp uvec4 color;\n"
            "void main(){ color=%s(texelFetch(source_texture,%s,sample_index)); }\n",
            array ? "#extension GL_OES_texture_storage_multisample_2d_array : require\n" : "",
            is_signed(kind) ? "i" : is_integer(kind) ? "u" : "",
            array ? "Array" : "", is_integer(kind) ? "uvec4" : "floatBitsToUint",
            array ? "ivec3(ivec2(gl_FragCoord.xy),layer_index)" : "ivec2(gl_FragCoord.xy)");
   return program_for(source);
}

static void
set_value(GLuint program, enum kind kind, uint32_t bits)
{
   GLint location = glGetUniformLocation(program, "value");
   if (is_signed(kind)) {
      GLint value;
      memcpy(&value, &bits, sizeof(value));
      glUniform4i(location, value, 0, 0, 1);
   } else if (is_integer(kind)) {
      glUniform4ui(location, bits, 0, 0, 1);
   } else {
      glUniform4f(location, bits_float(bits), 0.0f, 0.0f, 1.0f);
   }
}

static void
clear_guard(enum kind kind)
{
   const uint32_t bits = value_bits(kind, 0, 0, 0, 1);
   if (is_signed(kind)) {
      GLint value[4] = {0, 0, 0, 1};
      memcpy(value, &bits, sizeof(bits));
      glClearBufferiv(GL_COLOR, 0, value);
   } else if (is_integer(kind)) {
      const GLuint value[4] = {bits, 0, 0, 1};
      glClearBufferuiv(GL_COLOR, 0, value);
   } else {
      const GLfloat value[4] = {bits_float(bits), 0.0f, 0.0f, 1.0f};
      glClearBufferfv(kind == DEPTH32 ? GL_DEPTH : GL_COLOR, 0, value);
   }
}

static int
attach_layer(GLuint framebuffer, GLuint texture, enum kind kind, int array,
             unsigned layer)
{
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   const GLenum attachment = kind == DEPTH32 ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0;
   if (array)
      glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texture, 0, (GLint)layer);
   else
      glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D_MULTISAMPLE, texture, 0);
   const GLenum draw_buffer = kind == DEPTH32 ? GL_NONE : GL_COLOR_ATTACHMENT0;
   glDrawBuffers(1, &draw_buffer);
   glReadBuffer(draw_buffer);
   const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   if (status != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "FRAMEBUFFER_ERROR\tformat=%s\tlayer=%u\tstatus=0x%x\n",
              names[kind], layer, status);
      ++failures;
      return 0;
   }
   return check_gl("attach_layer");
}

static void
run_format(enum kind kind, unsigned samples, int array)
{
   const unsigned requested_samples = samples;
   const GLenum target = array ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY_OES : GL_TEXTURE_2D_MULTISAMPLE;
   const unsigned layers = array ? ARRAY_LAYERS : 1;
   GLuint texture = 0, output = 0, framebuffers[2] = {0}, writer = 0, reader = 0;
   const unsigned before = failures;
   GLint max_samples = 0;
   glGetInternalformativ(target, formats[kind], GL_SAMPLES, 1, &max_samples);
   if (!check_gl("format_samples"))
      return;
   if (samples > (unsigned)max_samples) {
      fprintf(stderr, "UNSUPPORTED\tformat=%s\tarray=%d\trequested=%u\tmaximum=%d\n",
              names[kind], array, samples, max_samples);
      ++unsupported;
      return;
   }
   writer = writer_program(kind);
   reader = reader_program(kind, array);
   if (!writer || !reader)
      goto cleanup;
   glGenTextures(1, &texture);
   glBindTexture(target, texture);
   if (array)
      storage3d(target, (GLsizei)samples, formats[kind], WIDTH, HEIGHT, layers, GL_TRUE);
   else
      glTexStorage2DMultisample(target, (GLsizei)samples, formats[kind], WIDTH, HEIGHT, GL_TRUE);
   GLint actual_samples = 0;
   glGetTexLevelParameteriv(target, 0, GL_TEXTURE_SAMPLES, &actual_samples);
   /* GL permits rounding the request upward. Visit every actual stored
    * sample and report that topology; fixed-count A/B compares only equal
    * actual counts, never mislabels a 4x allocation as 1x or 2x. */
   if (!check_gl("storage") || actual_samples < (GLint)samples || actual_samples > 8) {
      fprintf(stderr, "SAMPLE_COUNT_ERROR\trequested=%u\tactual=%d\n", samples, actual_samples);
      ++failures;
      goto cleanup;
   }
   samples = (unsigned)actual_samples;
   fprintf(stderr, "STORAGE\tformat=%s\tarray=%d\trequested=%u\tactual=%u\n",
           names[kind], array, requested_samples, samples);
   glGenFramebuffers(2, framebuffers);
   glGenTextures(1, &output);
   glBindTexture(GL_TEXTURE_2D, output);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, WIDTH, HEIGHT);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[1]);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "FRAMEBUFFER_ERROR\toutput_rgba32ui\n");
      ++failures;
      goto cleanup;
   }
   glViewport(0, 0, WIDTH, HEIGHT);
   glDisable(GL_BLEND);
   glDisable(GL_DITHER);
   glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
   glDisable(GL_SAMPLE_COVERAGE);
   glDepthMask(GL_TRUE);
   for (unsigned epoch = 0; epoch < 2; ++epoch) {
      /* Epoch 1 modifies only the last sample in the last layer. Every other
       * previously rendered sample, layer, and clear border must survive. */
      for (unsigned layer = 0; layer < layers; ++layer) {
         if (epoch && layer != layers - 1)
            continue;
         if (!attach_layer(framebuffers[0], texture, kind, array, layer))
            goto cleanup;
         glDisable(GL_SCISSOR_TEST);
         glDisable(GL_SAMPLE_MASK);
         if (!epoch)
            clear_guard(kind);
         glEnable(GL_SAMPLE_MASK);
         glEnable(GL_SCISSOR_TEST);
         glScissor(1, 1, WIDTH - 2, HEIGHT - 2);
         if (kind == DEPTH32) {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
         } else {
            glDisable(GL_DEPTH_TEST);
         }
         glUseProgram(writer);
         for (unsigned sample = 0; sample < samples; ++sample) {
            if (epoch && sample != samples - 1)
               continue;
            glSampleMaski(0, UINT32_C(1) << sample);
            set_value(writer, kind, value_bits(kind, layer, sample, epoch, 0));
            glDrawArrays(GL_TRIANGLES, 0, 3);
            ++draws;
         }
      }
      glDisable(GL_SCISSOR_TEST);
      glDisable(GL_SAMPLE_MASK);
      glDisable(GL_DEPTH_TEST);
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[1]);
      glBindTexture(target, texture);
      glUseProgram(reader);
      glUniform1i(glGetUniformLocation(reader, "source_texture"), 0);
      for (unsigned layer = 0; layer < layers; ++layer) {
         glUniform1i(glGetUniformLocation(reader, "layer_index"), (GLint)layer);
         for (unsigned sample = 0; sample < samples; ++sample) {
            const GLuint guard[4] = {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
            glClearBufferuiv(GL_COLOR, 0, guard);
            glUniform1i(glGetUniformLocation(reader, "sample_index"), (GLint)sample);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            ++draws;
            uint32_t observed[WIDTH * HEIGHT * 4];
            memset(observed, 0xa5, sizeof(observed));
            glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA_INTEGER, GL_UNSIGNED_INT, observed);
            ++reads;
            if (!check_gl("sample_fetch_readback"))
               goto cleanup;
            for (unsigned y = 0; y < HEIGHT; ++y) {
               for (unsigned x = 0; x < WIDTH; ++x) {
                  const int border = !x || x == WIDTH - 1 || !y || y == HEIGHT - 1;
                  const unsigned changed = epoch && layer == layers - 1 && sample == samples - 1;
                  uint32_t expected[4] = {value_bits(kind, layer, sample, changed, border),
                                         0, 0, is_integer(kind) ? 1u : float_bits(1.0f)};
                  for (unsigned channel = 0; channel < 4; ++channel) {
                     const uint32_t actual = observed[(y * WIDTH + x) * 4 + channel];
                     printf("%s\t%d\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%08" PRIx32 "\n",
                            names[kind], array, samples, epoch, layer, sample, x, y, channel, actual);
                     ++checks;
                     int equal = actual == expected[channel];
                     /* UNORM conversion may differ by one binary32 ULP; raw
                      * bits remain in stdout for independent backend comparison. */
                     if (kind == UNORM8 && channel == 0)
                        equal = fabsf(bits_float(actual) - bits_float(expected[0])) <= 1.0f / 16777216.0f;
                     if (!equal) {
                        if (failures - before < 12)
                           fprintf(stderr, "VALUE_FAIL\tformat=%s\tarray=%d\tepoch=%u\tlayer=%u\tsample=%u\tx=%u\ty=%u\tchannel=%u\tactual=%08" PRIx32 "\texpected=%08" PRIx32 "\n",
                                   names[kind], array, epoch, layer, sample, x, y, channel, actual, expected[channel]);
                        ++failures;
                     }
                  }
               }
            }
         }
      }
   }
   ++scenarios;
   fprintf(stderr, "SCENARIO\tformat=%s\tarray=%d\trequested=%u\tsamples=%u\tlayers=%u\tchecks=%s\n",
           names[kind], array, requested_samples, samples, layers, before == failures ? "Pass" : "Fail");
cleanup:
   glUseProgram(0);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteProgram(writer);
   glDeleteProgram(reader);
   glDeleteTextures(1, &texture);
   glDeleteTextures(1, &output);
   glDeleteFramebuffers(2, framebuffers);
   check_gl("format_cleanup");
}

int
main(int argc, char **argv)
{
   unsigned samples = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 10) : 4;
   const char *dimension = argc > 2 ? argv[2] : "both";
   const int run_2d = !strcmp(dimension, "2d") || !strcmp(dimension, "both");
   const int run_array = !strcmp(dimension, "array") || !strcmp(dimension, "both");
   if (argc > 3 || (samples != 1 && samples != 2 && samples != 4 && samples != 8) ||
       (!run_2d && !run_array)) {
      fprintf(stderr, "usage: %s [1|2|4|8] [2d|array|both]\n", argv[0]);
      return 2;
   }
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0, count = 0;
   EGLConfig config = NULL;
   EGLSurface surface = EGL_NO_SURFACE;
   EGLContext context = EGL_NO_CONTEXT;
   GLuint vao = 0, vertex_buffer = 0;
   int current = 0, status = 2;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
      fprintf(stderr, "EGL_INITIALIZE_ERROR\t0x%x\n", eglGetError());
      return 2;
   }
   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION_KHR, 3, EGL_CONTEXT_MINOR_VERSION_KHR, 1, EGL_NONE
   };
   const EGLint surface_attributes[] = {EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE};
   if (!eglBindAPI(EGL_OPENGL_ES_API) ||
       !eglChooseConfig(display, config_attributes, &config, 1, &count) || count != 1)
      goto cleanup;
   surface = eglCreatePbufferSurface(display, config, surface_attributes);
   context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      goto cleanup;
   current = 1;
   fprintf(stderr, "GL\tvendor=%s\trenderer=%s\tversion=%s\tglsl=%s\n",
           glGetString(GL_VENDOR), glGetString(GL_RENDERER),
           glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));
   storage3d = (PFNGLTEXSTORAGE3DMULTISAMPLEOESPROC)eglGetProcAddress("glTexStorage3DMultisampleOES");
   if (run_array && !storage3d) {
      fprintf(stderr, "UNSUPPORTED\tglTexStorage3DMultisampleOES\n");
      ++unsupported;
      goto cleanup;
   }
   static const GLfloat triangle[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vertex_buffer);
   glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
   glBufferData(GL_ARRAY_BUFFER, sizeof(triangle), triangle, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glActiveTexture(GL_TEXTURE0);
   puts("format\tarray\tsamples\tepoch\tlayer\tsample\tx\ty\tchannel\tu32_hex");
   for (enum kind kind = UNORM8; kind < FORMAT_COUNT; kind = (enum kind)(kind + 1)) {
      if (run_2d)
         run_format(kind, samples, 0);
      if (run_array)
         run_format(kind, samples, 1);
   }
   fprintf(stderr, "RESULT\tscenarios=%u\texpected_scenarios=%u\tdraws=%u\treads=%u\tchecks=%u\tfailures=%u\tunsupported=%u\n",
           scenarios, FORMAT_COUNT * (unsigned)(run_2d + run_array), draws, reads, checks, failures, unsupported);
   status = failures ? 1 : unsupported ? 2 : 0;
cleanup:
   if (current) {
      glDeleteBuffers(1, &vertex_buffer);
      glDeleteVertexArrays(1, &vao);
      if (!check_gl("cleanup"))
         status = 1;
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   }
   if (context != EGL_NO_CONTEXT)
      eglDestroyContext(display, context);
   if (surface != EGL_NO_SURFACE)
      eglDestroySurface(display, surface);
   if (status == 2)
      fprintf(stderr, "SETUP_OR_SUPPORT_FAILURE\tegl_error=0x%x\n", eglGetError());
   eglTerminate(display);
   return status;
}
