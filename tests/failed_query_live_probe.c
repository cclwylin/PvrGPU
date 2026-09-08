/* SPDX-License-Identifier: MIT */
/* Real EGL/GLES negative-path test. A deliberately nonexistent SystemC API
 * library makes the native submit fail. Query reads must terminate with the
 * retained GL error and leave result memory untouched, not return a fake zero.
 * Then the same object is reused for a genuinely empty query and deleted.
 * Use a task-specific driver output directory and a finite process timeout. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(int ok, const char *message)
{
   if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}

static GLuint
make_program(void)
{
   const char *source[] = {
      "#version 300 es\nlayout(location=0) in highp vec4 position;\n"
      "void main(){gl_Position=position;}\n",
      "#version 300 es\nprecision highp float; out vec4 color;\n"
      "void main(){color=vec4(1.0);}\n",
   };
   const GLenum stage[] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   GLuint program = glCreateProgram();
   for (unsigned i = 0; i < 2; ++i) {
      GLuint shader = glCreateShader(stage[i]);
      glShaderSource(shader, 1, &source[i], NULL);
      glCompileShader(shader);
      GLint compiled = 0;
      glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
      check(compiled, "compile shader");
      glAttachShader(program, shader);
      glDeleteShader(shader);
   }
   const char *varying = "gl_Position";
   glTransformFeedbackVaryings(program, 1, &varying, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(program);
   GLint linked = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   check(linked, "link program");
   return program;
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint config_attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
   const EGLint surface_attributes[] = {EGL_WIDTH, 4, EGL_HEIGHT, 4, EGL_NONE};
   const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
   EGLConfig config;
   EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display, config_attributes,
      &config, 1, &count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display, surface, surface, context), "EGL context");
   fprintf(stderr, "RENDERER: %s\n", glGetString(GL_RENDERER));
   check(strstr((const char *)glGetString(GL_RENDERER), "PvrGPU") != NULL, "requires real PvrGPU backend");
   GLuint program = make_program(), vao, buffers[2];
   glUseProgram(program);
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(2, buffers);
   glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
   const float position[] = {0, 0, 0, 1};
   glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffers[1]);
   const uint32_t initial[] = {0x12345678, 0x87654321, 0xdeadbeef, 0xcafebabe};
   glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(initial), initial, GL_DYNAMIC_READ);
   glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[1]);
   check(glGetError() == GL_NO_ERROR, "setup");
   /* No mock shader output: failure is the ordinary dlopen/submit path. */
   check(setenv("PVRGPU_SYSTEMC_API_LIB", "/nonexistent/pvrgpu-fault-injection.dylib", 1) == 0,
      "configure submission failure");

   for (unsigned repetition = 0; repetition < 2; ++repetition) {
      GLuint query;
      glGenQueries(1, &query);
      glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
      glBeginTransformFeedback(GL_POINTS);
      glDrawArrays(GL_POINTS, 0, 1);
      glEndTransformFeedback();
      glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
      check(glGetError() == GL_OUT_OF_MEMORY, "failed EndQuery retains GL_OUT_OF_MEMORY");
      GLint current = -1;
      glGetQueryiv(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, GL_CURRENT_QUERY, &current);
      check(glGetError() == GL_NO_ERROR && current == 0, "failed query is no longer bound");
      const GLenum pnames[] = {GL_QUERY_RESULT, GL_QUERY_RESULT_AVAILABLE, GL_QUERY_RESULT};
      for (unsigned i = 0; i < 3; ++i) {
         GLuint result = 0xfeed1234;
         glGetQueryObjectuiv(query, pnames[i], &result);
         check(glGetError() == GL_OUT_OF_MEMORY, "failed result read reports error without polling");
         check(result == 0xfeed1234, "failed result read overwrote caller memory");
      }
      const void *bytes = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
         sizeof(initial), GL_MAP_READ_BIT);
      check(bytes && memcmp(bytes, initial, sizeof(initial)) == 0, "rejected shader fabricated TF bytes");
      check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER), "unmap target");
      check(glGetError() == GL_NO_ERROR, "buffer readback");
      if (!repetition) {
         glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
         glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
         GLuint result = 0xfeed1234;
         glGetQueryObjectuiv(query, GL_QUERY_RESULT, &result);
         check(glGetError() == GL_NO_ERROR && result == 0, "reuse for a genuinely empty interval");
      }
      glDeleteQueries(1, &query);
      check(glGetError() == GL_NO_ERROR && !glIsQuery(query), "delete used or failed query");
   }
   glDeleteBuffers(2, buffers);
   glDeleteVertexArrays(1, &vao);
   glDeleteProgram(program);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   puts("PASS: failed query terminates, preserves errors/params/TF bytes, reuses and deletes safely");
   return 0;
}
