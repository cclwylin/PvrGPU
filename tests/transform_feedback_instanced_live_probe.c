/* SPDX-License-Identifier: MIT */
/* Real EGL/GLES 3.0 TF byte/query regression for independent instance
 * boundaries. The shader executes natively; host data are test oracles only.
 * Run with the ordinary surfaceless Mesa/pvrgpu environment, or llvmpipe.
 * Build: cc -I$prefix/include this.c -L$prefix/lib -lEGL -lGLESv2 -o probe */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(int condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "FAIL: %s (GL error 0x%x)\n", message, glGetError());
      exit(1);
   }
}

static GLuint
program(void)
{
   const char *sources[] = {
      "#version 300 es\nlayout(location=0) in highp vec4 position;\n"
      "layout(location=1) in highp float tag;\n"
      "flat out highp uvec2 ids;\n"
      "void main(){gl_Position=position; ids=uvec2(uint(tag),gl_InstanceID);}\n",
      "#version 300 es\nprecision highp float; out vec4 color;\n"
      "void main(){color=vec4(1.0);}\n"
   };
   const GLenum stages[] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   GLuint result = glCreateProgram();
   for (unsigned s = 0; s < 2; ++s) {
      GLuint shader = glCreateShader(stages[s]);
      glShaderSource(shader, 1, &sources[s], NULL);
      glCompileShader(shader);
      GLint ok = 0;
      glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
      if (!ok) {
         char log[4096];
         glGetShaderInfoLog(shader, sizeof(log), NULL, log);
         fprintf(stderr, "%s\n", log);
      }
      check(ok, "shader compile");
      glAttachShader(result, shader);
      glDeleteShader(shader);
   }
   const char *varyings[] = {"ids", "gl_Position"};
   glTransformFeedbackVaryings(result, 2, varyings, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(result);
   GLint ok = 0;
   glGetProgramiv(result, GL_LINK_STATUS, &ok);
   check(ok, "program link");
   return result;
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
   const EGLint surface_attributes[] = {EGL_WIDTH, 4, EGL_HEIGHT, 4, EGL_NONE};
   const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
   EGLConfig config;
   EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) &&
      eglChooseConfig(display, attributes, &config, 1, &count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display, surface, surface, context), "EGL context");
   fprintf(stderr, "RENDERER: %s\n", glGetString(GL_RENDERER));
   GLuint p = program();
   glUseProgram(p);
   const float positions[4][4] = {
      {-0.75f, -0.75f, 0.0f, 1.0f}, {0.75f, -0.75f, 0.0f, 1.0f},
      {-0.75f, 0.75f, 0.0f, 1.0f}, {0.75f, 0.75f, 0.0f, 1.0f},
   };
   const uint8_t source_indices[] = {3, 1, 0, 2};
   static const struct {
      GLenum mode, tf_mode;
      unsigned count, width;
      uint8_t elements[8];
   } cases[] = {
      {GL_LINE_STRIP, GL_LINES, 6, 2, {0, 1, 1, 2, 2, 3}},
      {GL_LINE_LOOP, GL_LINES, 8, 2, {0, 1, 1, 2, 2, 3, 3, 0}},
      {GL_TRIANGLE_STRIP, GL_TRIANGLES, 6, 3, {0, 1, 2, 2, 1, 3}},
      {GL_TRIANGLE_FAN, GL_TRIANGLES, 6, 3, {0, 1, 2, 0, 2, 3}},
   };
   GLuint vao, buffers[4], query, framebuffer, color;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(4, buffers);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   const float tags[] = {0, 1, 2, 3};
   glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(tags), tags, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(1);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1]);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(source_indices), source_indices, GL_STATIC_DRAW);
   glGenQueries(1, &query);
   glGenRenderbuffers(1, &color);
   glBindRenderbuffer(GL_RENDERBUFFER, color);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 4, 4);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "output framebuffer");
   glViewport(0, 0, 4, 4);
   glDisable(GL_DITHER);
   unsigned scenarios = 0, checked_words = 0;
   for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
      for (unsigned indexed = 0; indexed < 2; ++indexed) {
         glClear(GL_COLOR_BUFFER_BIT);
         uint32_t initial[128];
         for (unsigned i = 0; i < 128; ++i) initial[i] = 0xdeadbeef;
         glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffers[2]);
         glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(initial), initial, GL_DYNAMIC_READ);
         glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[2]);
         glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
         glBeginTransformFeedback(cases[c].tf_mode);
         if (indexed) glDrawElementsInstanced(cases[c].mode, 4, GL_UNSIGNED_BYTE, NULL, 2);
         else glDrawArraysInstanced(cases[c].mode, 0, 4, 2);
         glEndTransformFeedback();
         glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
         check(glGetError() == GL_NO_ERROR, "TF draw");
         GLuint written = 0;
         glGetQueryObjectuiv(query, GL_QUERY_RESULT, &written);
         check(written == cases[c].count * 2 / cases[c].width, "primitive query instance boundary");
         printf("QUERY %u %u %u\n", c, indexed, written);
         const uint32_t *actual = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
            sizeof(initial), GL_MAP_READ_BIT);
         check(actual != NULL, "map TF output");
         for (unsigned v = 0; v < cases[c].count * 2; ++v) {
            const unsigned instance = v / cases[c].count;
            unsigned vertex = cases[c].elements[v % cases[c].count];
            if (indexed) vertex = source_indices[vertex];
            uint32_t expected[6] = {vertex, instance};
            memcpy(&expected[2], positions[vertex], sizeof(positions[vertex]));
            for (unsigned word = 0; word < 6; ++word) {
               if (actual[v * 6 + word] != expected[word])
                  fprintf(stderr, "mismatch topology=%u indexed=%u vertex=%u word=%u actual=%08x expected=%08x\n",
                     c, indexed, v, word, actual[v * 6 + word], expected[word]);
               check(actual[v * 6 + word] == expected[word], "TF bytes/IDs instance boundary");
               ++checked_words;
            }
         }
         for (unsigned i = cases[c].count * 12; i < 128; ++i) {
            check(actual[i] == initial[i], "TF overwrote untouched tail");
            ++checked_words;
         }
         for (unsigned i = 0; i < 128; ++i)
            printf("WORD %u %u %u %08x\n", c, indexed, i, actual[i]);
         check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER), "unmap TF output");
         check(glGetError() == GL_NO_ERROR, "TF readback");
         fprintf(stderr, "PASS: topology=%u indexed=%u written=%u\n", c, indexed, written);
         ++scenarios;
      }
   }
   fprintf(stderr, "PASS: scenarios=%u checked_words=%u\n", scenarios, checked_words);
   glDeleteQueries(1, &query);
   glDeleteBuffers(4, buffers);
   glDeleteFramebuffers(1, &framebuffer);
   glDeleteRenderbuffers(1, &color);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(p);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
