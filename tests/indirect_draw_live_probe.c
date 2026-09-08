/* SPDX-License-Identifier: MIT */
/* Real native VS/TF probe. CPU, prior TF and prior CS produce argument bytes;
 * none may substitute shader execution or TF queries on the host. Compile:
 * cc -I$prefix/include this.c -L$prefix/lib -lEGL -lGLESv2 -o probe
 * Run with the project's ordinary surfaceless pvrgpu or llvmpipe environment.
 * GLES 3.1's reserved baseInstance word remains zero; the decoder unit covers
 * nonzero/extreme baseInstance and signed negative baseVertex separately.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int condition, const char *message)
{
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

static GLuint program(const char *vertex, const char *compute,
                      const char *const *varyings, unsigned count)
{
   GLuint object = glCreateProgram();
   GLuint first = shader(compute ? GL_COMPUTE_SHADER : GL_VERTEX_SHADER,
                         compute ? compute : vertex);
   glAttachShader(object, first);
   glDeleteShader(first);
   if (!compute) {
      GLuint fragment = shader(GL_FRAGMENT_SHADER,
         "#version 310 es\nprecision highp float; out vec4 color;\n"
         "void main(){color=vec4(1.0);}\n");
      glAttachShader(object, fragment);
      glDeleteShader(fragment);
      glTransformFeedbackVaryings(object, count, varyings, GL_INTERLEAVED_ATTRIBS);
   }
   glLinkProgram(object);
   GLint ok = 0;
   glGetProgramiv(object, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(object, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
   }
   check(ok, "native program link");
   return object;
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
   EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) &&
      eglChooseConfig(display, attributes, &config, 1, &count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display, surface, surface, context), "EGL 3.1 context");
   fprintf(stderr, "RENDERER: %s\n", glGetString(GL_RENDERER));
   const char *capture[] = {"ids", "gl_Position"};
   GLuint consumer = program(
      "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "layout(location=1) in highp float tag; flat out highp uvec2 ids;\n"
      "void main(){gl_Position=position; ids=uvec2(uint(tag),gl_InstanceID);}\n",
      NULL, capture, 2);
   const char *commands[] = {"command", "base_instance"};
   GLuint tf_producer = program(
      "#version 310 es\nlayout(location=0) in highp uvec4 source;\n"
      "flat out highp uvec4 command; flat out highp uint base_instance;\n"
      "void main(){gl_Position=vec4(0,0,0,1);command=source;base_instance=0u;}\n",
      NULL, commands, 2);
   GLuint cs_producer[2];
   for (unsigned indexed = 0; indexed < 2; ++indexed) {
      char source[1024];
      snprintf(source, sizeof(source),
         "#version 310 es\nlayout(local_size_x=1) in;\n"
         "layout(std430,binding=0) buffer Args {uint arg[];};\n"
         "void main(){arg[1]=4u;arg[2]=2u;arg[3]=1u;arg[4]=%uu;arg[5]=0u;}\n",
         indexed);
      cs_producer[indexed] = program(NULL, source, NULL, 0);
   }

   const float positions[6][4] = {
      {-0.75f,-0.75f,0,1}, {-0.25f,-0.75f,0,1}, {0.25f,-0.75f,0,1},
      {0.75f,-0.75f,0,1}, {-0.75f,0.75f,0,1}, {0.75f,0.75f,0,1}
   };
   const float tags[] = {0,1,2,3,4,5};
   const unsigned source_indices[] = {99,2,0,3,1};
   GLuint vaos[2], buffers[6], query, framebuffer, color;
   glGenVertexArrays(2, vaos);
   glGenBuffers(6, buffers);
   glBindVertexArray(vaos[0]);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(tags), tags, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(1);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
   glBindVertexArray(vaos[1]);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[5]);
   glVertexAttribIPointer(0, 4, GL_UNSIGNED_INT, 0, NULL);
   glEnableVertexAttribArray(0);
   glGenQueries(1, &query);
   glGenRenderbuffers(1, &color);
   glBindRenderbuffer(GL_RENDERBUFFER, color);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 4, 4);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "framebuffer");
   glViewport(0, 0, 4, 4);
   glDisable(GL_DITHER);
   unsigned scenarios = 0, checked_words = 0;
   const unsigned widths[] = {0,1,2,4};
   for (unsigned producer = 0; producer < 3; ++producer) {
      for (unsigned wi = 0; wi < 4; ++wi) {
         const unsigned width = widths[wi];
         // Nonzero offset catches treating firstIndex as a byte address and
         // incorrectly decoding the command from resource offset zero.
         for (unsigned empty = 0; empty < (producer ? 1U : 3U); ++empty) {
            uint32_t args[6] = {0xdeadbeef, empty == 1 ? 0U : 4U,
               empty == 2 ? 0U : 2U, 1, width ? 1U : 0U, 0};
            uint32_t initial[128];
            for (unsigned i = 0; i < 128; ++i) initial[i] = 0xdeadbeef;
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, buffers[3]);
            glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(args), producer ? NULL : args, GL_DYNAMIC_COPY);
            glClear(GL_COLOR_BUFFER_BIT);
            if (producer == 1) {
               glUseProgram(tf_producer);
               glBindVertexArray(vaos[1]);
               glBindBuffer(GL_ARRAY_BUFFER, buffers[5]);
               glBufferData(GL_ARRAY_BUFFER, 16, &args[1], GL_STATIC_DRAW);
               glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[3], 4, 20);
               glBeginTransformFeedback(GL_POINTS);
               glDrawArrays(GL_POINTS, 0, 1);
               glEndTransformFeedback();
               check(glGetError() == GL_NO_ERROR, "native TF argument producer");
               glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
            } else if (producer == 2) {
               glUseProgram(cs_producer[width != 0]);
               glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[3]);
               glDispatchCompute(1,1,1);
               glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
               check(glGetError() == GL_NO_ERROR, "native CS argument producer");
               glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
            }
            glUseProgram(consumer);
            glBindVertexArray(vaos[0]);
            if (width) {
               uint8_t indices[20] = {0};
               for (unsigned i = 0; i < 5; ++i) {
                  if (width == 1) indices[i] = source_indices[i];
                  else if (width == 2) {
                     const uint16_t narrow = source_indices[i];
                     memcpy(indices + i * 2, &narrow, 2);
                  } else memcpy(indices + i * 4, &source_indices[i], 4);
               }
               glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
               glBufferData(GL_ELEMENT_ARRAY_BUFFER, width * 5, indices, GL_STATIC_DRAW);
            }
            glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffers[4]);
            glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(initial), initial, GL_DYNAMIC_READ);
            glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[4]);
            glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
            glBeginTransformFeedback(GL_POINTS);
            if (width) glDrawElementsIndirect(GL_POINTS, width == 1 ? GL_UNSIGNED_BYTE :
                                              width == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                                              (void *)(uintptr_t)4);
            else glDrawArraysIndirect(GL_POINTS, (void *)(uintptr_t)4);
            glEndTransformFeedback();
            glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
            check(glGetError() == GL_NO_ERROR, "native indirect consumer");
            GLuint written = UINT32_MAX;
            glGetQueryObjectuiv(query, GL_QUERY_RESULT, &written);
            const unsigned vertices = empty ? 0 : 8;
            check(written == vertices, "indirect TF primitive query");
            printf("QUERY %u %u %u %u\n", producer, width, empty, written);
            const uint32_t *actual = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
               sizeof(initial), GL_MAP_READ_BIT);
            check(actual != NULL, "TF map");
            uint32_t expected[128];
            memcpy(expected, initial, sizeof(expected));
            for (unsigned v = 0; v < vertices; ++v) {
               const unsigned vertex = width ? source_indices[1 + v % 4] + 1 : 1 + v % 4;
               expected[v * 6] = vertex;
               expected[v * 6 + 1] = v / 4;
               memcpy(&expected[v * 6 + 2], positions[vertex], 16);
            }
            for (unsigned i = 0; i < 128; ++i) {
               if (actual[i] != expected[i])
                  fprintf(stderr, "producer=%u width=%u empty=%u word=%u actual=%08x expected=%08x\n",
                          producer, width, empty, i, actual[i], expected[i]);
               check(actual[i] == expected[i], "exact native TF bytes and untouched tail");
               printf("WORD %u %u %u %u %08x\n", producer, width, empty, i, actual[i]);
               ++checked_words;
            }
            check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER), "TF unmap");
            check(glGetError() == GL_NO_ERROR, "query and TF readback");
            fprintf(stderr, "PASS: producer=%u width=%u empty=%u written=%u\n", producer, width, empty, written);
            ++scenarios;
         }
      }
   }
   fprintf(stderr, "PASS: scenarios=%u checked_words=%u\n", scenarios, checked_words);
   glDeleteProgram(consumer);
   glDeleteProgram(tf_producer);
   glDeleteProgram(cs_producer[0]);
   glDeleteProgram(cs_producer[1]);
   glDeleteQueries(1, &query);
   glDeleteBuffers(6, buffers);
   glDeleteFramebuffers(1, &framebuffer);
   glDeleteRenderbuffers(1, &color);
   glDeleteVertexArrays(2, vaos);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
