/* SPDX-License-Identifier: MIT */
/* Native indirect vertex-fetch regression. Compile with the installed Mesa
 * EGL/GLES headers/libraries, then run unchanged against pvrgpu and llvmpipe.
 * stdout is a deterministic complete DWORD/query transcript for exact diff.
 *
 * Coverage: legal negative/positive baseVertex with equivalent effective
 * indices; u8/u16/u32; firstIndex and command byte offsets; DrawArraysIndirect;
 * disabled current/default attributes; explicit binding stride zero; bounded
 * EBO reads returning raw index zero before adding baseVertex. The latter is
 * llvmpipe's robustness policy, not a portable pixel oracle for undefined data.
 * Every draw runs a real VS and captures its position, attribute and instance
 * outputs using guarded transform feedback. No gl_VertexID dependency: that
 * intrinsic is outside the current PvrGPU generic VS input contract.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VERTICES = 4, INSTANCES = 2, RECORD_WORDS = 9,
       GUARD_WORDS = 16, TOTAL_WORDS = 160 };

static unsigned checks;

static void check(int condition, const char *message)
{
   ++checks;
   if (!condition) {
      fprintf(stderr, "FAIL: %s (GL error 0x%x)\n", message, glGetError());
      exit(1);
   }
}

static GLuint shader(GLenum stage, const char *source)
{
   GLuint object = glCreateShader(stage);
   glShaderSource(object, 1, &source, NULL);
   glCompileShader(object);
   GLint ok = 0;
   glGetShaderiv(object, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(object, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
   }
   check(ok, "native shader compile");
   return object;
}

static GLuint program(void)
{
   GLuint object = glCreateProgram();
   GLuint vs = shader(GL_VERTEX_SHADER,
      "#version 310 es\n"
      "layout(location=0) in highp vec4 position;\n"
      "layout(location=1) in highp vec4 vertex_attribute;\n"
      "flat out highp uint instance_id; out highp vec4 observed;\n"
      "void main(){gl_Position=position; observed=vertex_attribute;"
      "instance_id=uint(gl_InstanceID);}\n");
   GLuint fs = shader(GL_FRAGMENT_SHADER,
      "#version 310 es\nprecision highp float; out vec4 color;\n"
      "void main(){color=vec4(1.0);}\n");
   glAttachShader(object, vs);
   glAttachShader(object, fs);
   glDeleteShader(vs);
   glDeleteShader(fs);
   const char *varyings[] = {"instance_id", "observed", "gl_Position"};
   glTransformFeedbackVaryings(object, 3, varyings, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(object);
   GLint ok = 0;
   glGetProgramiv(object, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(object, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
   }
   check(ok, "native TF program link");
   return object;
}

static void encode_index(uint8_t *bytes, unsigned width, unsigned i, uint32_t value)
{
   if (width == 1) {
      check(value <= UINT8_MAX, "u8 index representable");
      bytes[i] = (uint8_t)value;
   } else if (width == 2) {
      const uint16_t narrow = (uint16_t)value;
      check(value <= UINT16_MAX, "u16 index representable");
      memcpy(bytes + i * 2, &narrow, 2);
   } else {
      check(width == 4, "u32 index size");
      memcpy(bytes + i * 4, &value, 4);
   }
}

int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
   const EGLint surface_attributes[] = {EGL_WIDTH, 4, EGL_HEIGHT, 4, EGL_NONE};
   const EGLint context_attributes[] = {EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
   EGLConfig config = NULL;
   EGLint config_count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) &&
      eglChooseConfig(display, attributes, &config, 1, &config_count) && config_count == 1,
      "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display, surface, surface, context), "EGL 3.1 context");
   fprintf(stderr, "RENDERER: %s\n", glGetString(GL_RENDERER));

   const float positions[12][4] = {
      {-0.75f,-0.75f,0.00000f,1}, {-0.25f,-0.75f,0.03125f,1},
      { 0.25f,-0.75f,0.06250f,1}, { 0.75f,-0.75f,0.09375f,1},
      {-0.75f,-0.25f,0.12500f,1}, {-0.25f,-0.25f,0.15625f,1},
      { 0.25f,-0.25f,0.18750f,1}, { 0.75f,-0.25f,0.21875f,1},
      {-0.75f, 0.25f,0.25000f,1}, {-0.25f, 0.25f,0.28125f,1},
      { 0.25f, 0.25f,0.31250f,1}, { 0.75f, 0.25f,0.34375f,1}
   };
   float vertex_attributes[12][4];
   for (unsigned i = 0; i < 12; ++i) {
      vertex_attributes[i][0] = (float)i;
      vertex_attributes[i][1] = 0.25f * (float)i;
      vertex_attributes[i][2] = -(float)i;
      vertex_attributes[i][3] = 1.0f;
   }
   const float default_attribute[4] = {0,0,0,1};
   const float set_attribute[4] = {-0.25f,8,42,1};
   const unsigned effective_indices[4] = {4,6,5,7};
   const unsigned widths[] = {0,1,2,4};
   const int biases[] = {0,2,-3};
   GLuint prog = program(), vao, buffers[5], query;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(5, buffers);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   /* In glVertexAttribPointer, zero denotes tightly packed, not a literal
    * constant stride. The mode=3 glBindVertexBuffer call tests literal zero. */
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_attributes), vertex_attributes, GL_STATIC_DRAW);
   glGenQueries(1, &query);
   glUseProgram(prog);
   glViewport(0, 0, 4, 4);
   glDisable(GL_DITHER);
   check(glGetError() == GL_NO_ERROR, "probe setup");

   unsigned scenarios = 0, checked_words = 0;
   for (unsigned mode = 0; mode < 4; ++mode) {
      if (mode == 0) {
         glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
         glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
         glEnableVertexAttribArray(1);
      } else if (mode == 1 || mode == 2) {
         glDisableVertexAttribArray(1);
         if (mode == 2)
            glVertexAttrib4fv(1, set_attribute);
      } else {
         glVertexAttribFormat(1, 4, GL_FLOAT, GL_FALSE, 0);
         glVertexAttribBinding(1, 1);
         glBindVertexBuffer(1, buffers[1], sizeof(float[4]), 0);
         glEnableVertexAttribArray(1);
      }
      check(glGetError() == GL_NO_ERROR, "attribute mode setup");
      uint32_t equivalent[3][TOTAL_WORDS];
      for (unsigned wi = 0; wi < 4; ++wi) {
         const unsigned width = widths[wi];
         /* kind 0: valid. kind 1: first two reads valid, last two OOB.
          * kind 2: firstIndex exactly at end, so all four reads OOB. */
         for (unsigned kind = 0; kind < (width ? 3U : 1U); ++kind) {
            for (unsigned bi = 0; bi < (width ? (kind ? 2U : 3U) : 1U); ++bi) {
               const int bias = biases[bi];
               const unsigned index_count = kind ? 4 : 8;
               const unsigned first_index = kind == 2 ? index_count : 2;
               uint8_t indices[32] = {0};
               if (width) {
                  encode_index(indices, width, 0, 99);
                  encode_index(indices, width, 1, 98);
                  for (unsigned i = 2; i < index_count; ++i) {
                     const unsigned chosen = i < 6 ? effective_indices[i - 2] : 97;
                     encode_index(indices, width, i, (uint32_t)((int)chosen - bias));
                  }
                  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
                  glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_count * width, indices, GL_STATIC_DRAW);
               }
               uint32_t args[8] = {0x12345678,0xabcdef98,VERTICES,INSTANCES,
                  width ? first_index : 4, width ? (uint32_t)bias : 0,0,0x55aa00ff};
               glBindBuffer(GL_DRAW_INDIRECT_BUFFER, buffers[3]);
               glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(args), args, GL_STATIC_DRAW);
               uint32_t expected[TOTAL_WORDS];
               for (unsigned i = 0; i < TOTAL_WORDS; ++i)
                  expected[i] = 0xdead0000u ^ (i * 0x1021u);
               glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffers[4]);
               glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(expected), expected, GL_DYNAMIC_READ);
               glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[4], GUARD_WORDS * 4,
                  (VERTICES * INSTANCES * RECORD_WORDS + GUARD_WORDS) * 4);
               glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
               glBeginTransformFeedback(GL_POINTS);
               if (width)
                  glDrawElementsIndirect(GL_POINTS, width == 1 ? GL_UNSIGNED_BYTE :
                     width == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, (void *)(uintptr_t)8);
               else
                  glDrawArraysIndirect(GL_POINTS, (void *)(uintptr_t)8);
               glEndTransformFeedback();
               glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
               check(glGetError() == GL_NO_ERROR, "native indirect TF draw");
               GLuint written = UINT32_MAX;
               glGetQueryObjectuiv(query, GL_QUERY_RESULT, &written);
               check(written == VERTICES * INSTANCES, "eight native TF point records");
               printf("QUERY %u %u %u %d %u\n", mode, width, kind, bias, written);
               for (unsigned v = 0; v < VERTICES * INSTANCES; ++v) {
                  unsigned vertex = width ? effective_indices[v % VERTICES] : 4 + v % VERTICES;
                  if (kind == 2 || (kind == 1 && v % VERTICES >= 2))
                     vertex = (unsigned)bias;
                  check(vertex < 12, "oracle effective vertex bound");
                  const unsigned offset = GUARD_WORDS + v * RECORD_WORDS;
                  expected[offset] = v / VERTICES;
                  const float *attribute = mode == 0 ? vertex_attributes[vertex] :
                     mode == 1 ? default_attribute : mode == 2 ? set_attribute : vertex_attributes[1];
                  memcpy(expected + offset + 1, attribute, 16);
                  memcpy(expected + offset + 5, positions[vertex], 16);
               }
               const uint32_t *actual = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                  sizeof(expected), GL_MAP_READ_BIT);
               check(actual != NULL, "guarded TF readback map");
               for (unsigned i = 0; i < TOTAL_WORDS; ++i) {
                  if (actual[i] != expected[i])
                     fprintf(stderr, "mode=%u width=%u kind=%u bias=%d word=%u actual=%08x expected=%08x\n",
                        mode, width, kind, bias, i, actual[i], expected[i]);
                  check(actual[i] == expected[i], "exact TF attribute/position and prefix/range/tail guards");
                  printf("WORD %u %u %u %d %u %08x\n", mode, width, kind, bias, i, actual[i]);
                  ++checked_words;
               }
               if (width && kind == 0) {
                  if (bi == 0)
                     memcpy(equivalent[wi - 1], actual, sizeof(expected));
                  else
                     check(!memcmp(equivalent[wi - 1], actual, sizeof(expected)),
                        "positive and negative baseVertex exactly equivalent to zero base");
               }
               check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER), "TF unmap");
               const void *commands = glMapBufferRange(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(args), GL_MAP_READ_BIT);
               check(commands && !memcmp(commands, args, sizeof(args)), "command payload and guards unchanged");
               check(glUnmapBuffer(GL_DRAW_INDIRECT_BUFFER), "command unmap");
               if (width) {
                  const void *elements = glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0,
                     index_count * width, GL_MAP_READ_BIT);
                  check(elements && !memcmp(elements, indices, index_count * width), "EBO payload and guards unchanged");
                  check(glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER), "EBO unmap");
               }
               check(glGetError() == GL_NO_ERROR, "all native query/readback checks");
               fprintf(stderr, "PASS: mode=%u width=%u kind=%u bias=%d written=%u\n",
                  mode, width, kind, bias, written);
               ++scenarios;
            }
         }
      }
   }
   check(scenarios == 88, "complete scenario inventory");
   fprintf(stderr, "PASS: scenarios=%u checked_words=%u checks=%u\n", scenarios, checked_words, checks);
   glDeleteProgram(prog);
   glDeleteQueries(1, &query);
   glDeleteBuffers(5, buffers);
   glDeleteVertexArrays(1, &vao);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
