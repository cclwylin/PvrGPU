/* SPDX-License-Identifier: MIT */
/* Reuse only the existing context-independent check/compile helpers. The old
 * 12-pass regression stays unchanged and its renamed main is never called. */
#define main original_fragment_output_pruning_main
#include "fragment_output_pruning_live_probe.c"
#undef main

int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint attrs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
   EGLConfig config = NULL;
   EGLint count = 0;
   check(eglChooseConfig(display, attrs, &config, 1, &count) && count == 1, "EGL config");
   check(eglBindAPI(EGL_OPENGL_ES_API), "EGL bind GLES");
   const EGLint surface_attrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   const EGLint context_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display, surface, surface, context), "EGL context");
   fprintf(stderr, "GL_VERSION=%s GL_RENDERER=%s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));
   const char *vs = "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "void main(){gl_Position=position;}\n";
   const char *fs = "#version 310 es\nprecision highp float;\n"
      "uniform highp float depth_value; void main(){gl_FragDepth=depth_value;}\n";
   GLuint vertex = compile(GL_VERTEX_SHADER, vs), fragment = compile(GL_FRAGMENT_SHADER, fs);
   GLuint program = glCreateProgram();
   glAttachShader(program, vertex); glAttachShader(program, fragment); glLinkProgram(program);
   GLint linked = 0; glGetProgramiv(program, GL_LINK_STATUS, &linked);
   check(linked, "link actual depth-only FS");
   glDeleteShader(vertex); glDeleteShader(fragment);
   glUseProgram(program);
   GLint depth_uniform = glGetUniformLocation(program, "depth_value");
   check(depth_uniform >= 0, "active shader depth input");

   GLuint texture[2], fbo[2], vao, vbo;
   glGenTextures(2, texture);
   for (unsigned t = 0; t < 2; ++t) {
      glBindTexture(GL_TEXTURE_2D, texture[t]);
      glTexStorage2D(GL_TEXTURE_2D, 1, t ? GL_DEPTH_COMPONENT32F : GL_RGBA8, SIDE, SIDE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   }
   glGenFramebuffers(2, fbo);
   for (unsigned target = 0; target < 2; ++target) {
      glBindFramebuffer(GL_FRAMEBUFFER, fbo[target]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture[1], 0);
      const GLenum draw = target ? GL_NONE : GL_COLOR_ATTACHMENT0;
      if (!target)
         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[0], 0);
      glDrawBuffers(1, &draw); glReadBuffer(draw);
      check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "complete depth FBO");
   }
   const float positions[] = {-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
   glGenVertexArrays(1, &vao); glBindVertexArray(vao);
   glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL); glEnableVertexAttribArray(0);
   glViewport(0, 0, SIDE, SIDE);
   glDisable(GL_BLEND); glDisable(GL_CULL_FACE); glDisable(GL_DITHER); glDisable(GL_SCISSOR_TEST);
   glEnable(GL_DEPTH_TEST); glDepthFunc(GL_ALWAYS); glDepthMask(GL_TRUE);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   check_gl("setup");
   const float values[] = {0.125f, 0.25f, 0.5f, 0.75f};
   const GLfloat initial_color[] = {1,0,1,1}, initial_depth = 1;
   const uint8_t expected_color[] = {255,0,255,255};
   for (unsigned target = 0; target < 2; ++target) {
      for (unsigned value = 0; value < 4; ++value) {
         glBindFramebuffer(GL_FRAMEBUFFER, fbo[0]);
         glClearBufferfv(GL_COLOR, 0, initial_color);
         glClearBufferfv(GL_DEPTH, 0, &initial_depth);
         glBindFramebuffer(GL_FRAMEBUFFER, fbo[target]);
         glUniform1f(depth_uniform, values[value]);
         check_gl("before actual depth draw");
         fprintf(stderr, "DRAW: color_attachment=%u shader_depth=%08x\n", !target, float_bits(values[value]));
         glDrawArrays(GL_TRIANGLES, 0, 3); glFinish(); check_gl("actual depth draw");
         uint8_t color_bytes[GUARD + PIXELS * 4 + GUARD];
         memset(color_bytes, 0xa5, sizeof(color_bytes));
         glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[0]); glReadBuffer(GL_COLOR_ATTACHMENT0);
         glReadPixels(0, 0, SIDE, SIDE, GL_RGBA, GL_UNSIGNED_BYTE, color_bytes + GUARD);
         check_gl("actual unchanged color readback");
         for (unsigned g = 0; g < GUARD; ++g)
            check(color_bytes[g] == 0xa5 && color_bytes[GUARD + PIXELS * 4 + g] == 0xa5, "color bounds");
         for (unsigned p = 0; p < PIXELS; ++p)
            check(!memcmp(color_bytes + GUARD + p * 4, expected_color, 4), "output-less color preserved");
         uint32_t depth_bits[PIXELS + 8];
         for (unsigned p = 0; p < PIXELS + 8; ++p) depth_bits[p] = 0xa5a5a5a5;
         glReadPixels(0, 0, SIDE, SIDE, GL_DEPTH_COMPONENT, GL_FLOAT, depth_bits + 4);
         check_gl("actual shader depth readback");
         for (unsigned p = 0; p < 4; ++p)
            check(depth_bits[p] == 0xa5a5a5a5 && depth_bits[4 + PIXELS + p] == 0xa5a5a5a5, "depth bounds");
         for (unsigned p = 0; p < PIXELS; ++p)
            check(depth_bits[4+p] == float_bits(values[value]), "native depth feedback value");
         printf("pass=%u color_attachment=%u color=ff00ffff depth=%08x pixels=%u\n",
            passes++, !target, depth_bits[4], PIXELS);
      }
   }
   glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
   glDeleteFramebuffers(2, fbo); glDeleteTextures(2, texture); glDeleteProgram(program);
   check_gl("cleanup");
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context); eglDestroySurface(display, surface); eglTerminate(display);
   fprintf(stderr, "Depth-only live probe PASS: passes=%u checks=%u\n", passes, checks);
   return 0;
}
