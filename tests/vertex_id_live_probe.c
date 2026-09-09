/* SPDX-License-Identifier: MIT */
/* Live VertexID/InstanceID transport regression. Build with the installed Mesa
 * EGL/GLES headers/libraries, then run the same executable with llvmpipe and
 * pvrgpu (and its explicit native bridge). stdout is a deterministic DWORD and
 * query transcript suitable for exact diff; expected values come from GL input
 * indices/first/baseVertex, never from prepared llvmpipe output.
 *
 * Example: clang -std=c11 -O2 -I "$PREFIX/include" this_file.c
 *   -L "$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" -lEGL -lGLESv2 -o probe
 * Run with EGL_PLATFORM=surfaceless, MESA_GLES_VERSION_OVERRIDE=3.1,
 * GALLIUM_DRIVER=llvmpipe or pvrgpu and the corresponding Mesa library paths.
 * Follow the existing indirect_draw_vertex_fetch_probe.c manual convention.
 *
 * Four independently linked VS variants read VertexID alone or together with
 * InstanceID, with two real attributes or a zero-attribute procedural layout,
 * checking synthetic input slot order. Real VS outputs (both IDs,
 * vertex attribute and position) are captured with prefix/range/tail guards.
 * Covers u8/u16/u32, repeated/nonzero-minimum indices, legal positive/negative
 * baseVertex, nonindexed first=4, direct/indirect draws and 1/3 instances. No
 * undefined/OOB inputs or renderer-specific robustness assumptions are used.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VERTICES = 6, RECORD_WORDS = 10, GUARD_WORDS = 16,
       TOTAL_WORDS = 256, INPUT_VERTICES = 16 };
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
   check(ok, "VS/FS compile");
   return object;
}

static GLuint program(unsigned variant)
{
   const unsigned joint_ids = variant & 1U;
   const char *prefix = variant < 2 ?
      "#version 310 es\n"
      "layout(location=0) in highp vec4 position;\n"
      "layout(location=1) in highp vec4 vertex_attribute;\n"
      "flat out highp uvec2 observed_ids; out highp vec4 observed;\n"
      "void main(){gl_Position=position; observed=vertex_attribute;\n" :
      "#version 310 es\n"
      "flat out highp uvec2 observed_ids; out highp vec4 observed;\n"
      "void main(){highp float id=float(gl_VertexID);\n"
      "gl_Position=vec4(-0.75+0.125*float(gl_VertexID%4),\n"
      "-0.75+0.25*float(gl_VertexID/4),0.03125*id,1.0);\n"
      "observed=vec4(id,0.25*id,-id,1.0);\n";
   char source[1024];
   const int length = snprintf(source, sizeof(source), "%s%s", prefix,
      joint_ids ? "observed_ids=uvec2(uint(gl_VertexID),uint(gl_InstanceID));}\n" :
                  "observed_ids=uvec2(uint(gl_VertexID),0x1357u);}\n");
   check(length > 0 && (size_t)length < sizeof(source), "shader source bound");
   GLuint object = glCreateProgram();
   GLuint vs = shader(GL_VERTEX_SHADER, source);
   GLuint fs = shader(GL_FRAGMENT_SHADER,
      "#version 310 es\nprecision highp float; out vec4 color;\n"
      "void main(){color=vec4(1.0);}\n");
   glAttachShader(object, vs);
   glAttachShader(object, fs);
   glDeleteShader(vs);
   glDeleteShader(fs);
   const char *varyings[] = {"observed_ids", "observed", "gl_Position"};
   glTransformFeedbackVaryings(object, 3, varyings, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(object);
   GLint ok = 0;
   glGetProgramiv(object, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(object, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
   }
   check(ok, "VertexID TF program link");
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
      check(width == 4, "u32 index width");
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

   float positions[INPUT_VERTICES][4], vertex_attributes[INPUT_VERTICES][4];
   for (unsigned i = 0; i < INPUT_VERTICES; ++i) {
      positions[i][0] = -0.75f + 0.125f * (float)(i % 4);
      positions[i][1] = -0.75f + 0.25f * (float)(i / 4);
      positions[i][2] = 0.03125f * (float)i;
      positions[i][3] = 1.0f;
      vertex_attributes[i][0] = (float)i;
      vertex_attributes[i][1] = 0.25f * (float)i;
      vertex_attributes[i][2] = -(float)i;
      vertex_attributes[i][3] = 1.0f;
   }
   const unsigned effective_indices[VERTICES] = {4, 7, 4, 5, 7, 6};
   const unsigned widths[] = {0, 1, 2, 4};
   const int biases[] = {0, 2, -3};
   const unsigned instance_counts[] = {1, 3};
   GLuint programs[4] = {program(0), program(1), program(2), program(3)};
   GLuint vao[2], buffers[5], query;
   glGenVertexArrays(2, vao);
   glBindVertexArray(vao[0]);
   glGenBuffers(5, buffers);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_attributes), vertex_attributes, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(1);
   glGenQueries(1, &query);
   glViewport(0, 0, 4, 4);
   glDisable(GL_DITHER);
   check(glGetError() == GL_NO_ERROR, "probe setup");

   unsigned scenarios = 0, checked_words = 0;
   for (unsigned joint = 0; joint < 4; ++joint) {
      glBindVertexArray(vao[joint >= 2]);
      glUseProgram(programs[joint]);
      for (unsigned ii = 0; ii < 2; ++ii) {
         const unsigned instances = instance_counts[ii];
         uint32_t equivalent[TOTAL_WORDS];
         int have_equivalent = 0;
         for (unsigned wi = 0; wi < 4; ++wi) {
            const unsigned width = widths[wi];
            for (unsigned bi = 0; bi < (width ? 3U : 1U); ++bi) {
               const int bias = biases[bi];
               for (unsigned indirect = 0; indirect < 2; ++indirect) {
                  if (!indirect && bias)
                     continue; /* GLES 3.1 core direct draw has no baseVertex argument. */
                  uint8_t indices[40] = {0};
                  if (width) {
                     encode_index(indices, width, 0, 99);
                     encode_index(indices, width, 1, 98);
                     for (unsigned i = 0; i < VERTICES; ++i) {
                        const int raw = (int)effective_indices[i] - bias;
                        check(raw > 0 && raw + bias == (int)effective_indices[i],
                           "oracle nonzero raw index and original effective ID");
                        encode_index(indices, width, i + 2, (uint32_t)raw);
                     }
                     encode_index(indices, width, 8, 97);
                     encode_index(indices, width, 9, 96);
                     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
                     glBufferData(GL_ELEMENT_ARRAY_BUFFER, 10 * width, indices, GL_STATIC_DRAW);
                  }
                  uint32_t args[8] = {0x12345678, 0xabcdef98, VERTICES, instances,
                     width ? 2U : 4U, width ? (uint32_t)bias : 0U, 0, 0x55aa00ff};
                  glBindBuffer(GL_DRAW_INDIRECT_BUFFER, buffers[3]);
                  glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(args), args, GL_STATIC_DRAW);
                  uint32_t expected[TOTAL_WORDS];
                  for (unsigned i = 0; i < TOTAL_WORDS; ++i)
                     expected[i] = 0xdead0000u ^ (i * 0x1021u);
                  glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffers[4]);
                  glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(expected), expected, GL_DYNAMIC_READ);
                  glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[4], GUARD_WORDS * 4,
                     (VERTICES * instances * RECORD_WORDS + GUARD_WORDS) * 4);
                  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
                  glBeginTransformFeedback(GL_POINTS);
                  const GLenum type = width == 1 ? GL_UNSIGNED_BYTE :
                     width == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT;
                  if (indirect) {
                     if (width)
                        glDrawElementsIndirect(GL_POINTS, type, (void *)(uintptr_t)8);
                     else
                        glDrawArraysIndirect(GL_POINTS, (void *)(uintptr_t)8);
                  } else if (width) {
                     glDrawElementsInstanced(GL_POINTS, VERTICES, type,
                        (void *)(uintptr_t)(2 * width), instances);
                  } else {
                     glDrawArraysInstanced(GL_POINTS, 4, VERTICES, instances);
                  }
                  glEndTransformFeedback();
                  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
                  check(glGetError() == GL_NO_ERROR, "live VertexID TF draw");
                  GLuint written = UINT32_MAX;
                  glGetQueryObjectuiv(query, GL_QUERY_RESULT, &written);
                  check(written == VERTICES * instances, "exact native TF primitive count");
                  printf("QUERY %u %u %u %d %u %u\n", joint, instances, width, bias, indirect, written);
                  for (unsigned v = 0; v < VERTICES * instances; ++v) {
                     const unsigned vertex = width ? effective_indices[v % VERTICES] : 4 + v % VERTICES;
                     check(vertex < INPUT_VERTICES, "oracle original effective ID in vertex buffer");
                     const unsigned offset = GUARD_WORDS + v * RECORD_WORDS;
                     expected[offset] = vertex;
                     expected[offset + 1] = (joint & 1U) ? v / VERTICES : 0x1357;
                     memcpy(expected + offset + 2, vertex_attributes[vertex], 16);
                     memcpy(expected + offset + 6, positions[vertex], 16);
                  }
                  const uint32_t *actual = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                     sizeof(expected), GL_MAP_READ_BIT);
                  check(actual != NULL, "guarded TF readback map");
                  for (unsigned i = 0; i < TOTAL_WORDS; ++i) {
                     if (actual[i] != expected[i])
                        fprintf(stderr, "joint=%u instances=%u width=%u bias=%d indirect=%u word=%u actual=%08x expected=%08x\n",
                           joint, instances, width, bias, indirect, i, actual[i], expected[i]);
                     check(actual[i] == expected[i], "exact IDs/attributes/position and prefix/range/tail guards");
                     printf("WORD %u %u %u %d %u %u %08x\n", joint, instances, width, bias, indirect, i, actual[i]);
                     ++checked_words;
                  }
                  if (width) {
                     if (!have_equivalent) {
                        memcpy(equivalent, actual, sizeof(expected));
                        have_equivalent = 1;
                     } else {
                        check(!memcmp(equivalent, actual, sizeof(expected)),
                           "all widths/baseVertex/direct modes preserve original effective IDs");
                     }
                  }
                  check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER), "TF unmap");
                  const void *commands = glMapBufferRange(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(args), GL_MAP_READ_BIT);
                  check(commands && !memcmp(commands, args, sizeof(args)), "indirect command and guards unchanged");
                  check(glUnmapBuffer(GL_DRAW_INDIRECT_BUFFER), "command unmap");
                  if (width) {
                     const void *elements = glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, 10 * width, GL_MAP_READ_BIT);
                     check(elements && !memcmp(elements, indices, 10 * width), "original EBO and guards unchanged");
                     check(glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER), "EBO unmap");
                  }
                  check(glGetError() == GL_NO_ERROR, "all query/readback checks");
                  fprintf(stderr, "PASS: joint=%u instances=%u width=%u bias=%d indirect=%u written=%u\n",
                     joint, instances, width, bias, indirect, written);
                  ++scenarios;
               }
            }
         }
      }
   }
   check(scenarios == 112, "complete scenario inventory");
   fprintf(stderr, "PASS: scenarios=%u checked_words=%u checks=%u\n", scenarios, checked_words, checks);
   glDeleteProgram(programs[0]);
   glDeleteProgram(programs[1]);
   glDeleteProgram(programs[2]);
   glDeleteProgram(programs[3]);
   glDeleteVertexArrays(2, vao);
   glDeleteQueries(1, &query);
   glDeleteBuffers(5, buffers);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
