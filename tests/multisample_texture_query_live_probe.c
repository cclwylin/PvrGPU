/* SPDX-License-Identifier: MIT */
/* Standalone real EGL/GLES 3.1 descriptor-query probe. No texture fetch,
 * texture initialization, or host-generated shader output is performed.
 * The host checks textureSize against actual allocated GL storage. Requested
 * sample counts may round upward; they are recorded, not shader queried.
 * textureSamples is not a GLES GLSL builtin in pinned Mesa 26.2.1 (desktop
 * GL450/ARB_shader_texture_image_samples only), so it is deliberately absent.
 * Build like multisample_texture_live_probe.c; no command-line arguments. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { OUTPUT_WIDTH = 2, OUTPUT_HEIGHT = 2 };
static unsigned failures, draws, reads, checks;

static int
check_gl(const char *where)
{
   int ok = 1;
   for (GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError()) {
      fprintf(stderr, "GL_ERROR\t%s\t0x%x\n", where, error);
      ++failures;
      ok = 0;
   }
   return ok;
}

static GLuint
query_program(int array)
{
   static const char *vertex =
      "#version 310 es\nlayout(location=0) in highp vec2 position;\n"
      "void main(){gl_Position=vec4(position,0.0,1.0);}\n";
   char fragment[1024];
   snprintf(fragment, sizeof(fragment),
      "#version 310 es\n%s"
      "precision highp float; precision highp int;\n"
      "uniform highp sampler2DMS%s source_texture;\n"
      "layout(location=0) out highp uvec4 color;\n"
      "void main(){color=uvec4(textureSize(source_texture),%s0x51u);}\n",
      array ? "#extension GL_OES_texture_storage_multisample_2d_array : require\n" : "",
      array ? "Array" : "", array ? "" : "1,");
   const char *sources[] = {vertex, fragment};
   const GLenum stages[] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   GLuint program = glCreateProgram();
   for (unsigned stage = 0; stage < 2; ++stage) {
      GLuint shader = glCreateShader(stages[stage]);
      GLint ok = GL_FALSE;
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
   GLint ok = GL_FALSE;
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

static void
run_queries(PFNGLTEXSTORAGE3DMULTISAMPLEOESPROC storage3d,
            GLuint program, unsigned requested, int array, unsigned layers)
{
   GLuint textures[2] = {0};
   glGenTextures(2, textures);
   const GLenum target = array ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY_OES :
                                 GL_TEXTURE_2D_MULTISAMPLE;
   /* Rebind the same program to different non-square dimensions. Its native
    * descriptor query must read this draw's snapshot, never a stale one. */
   for (unsigned binding = 0; binding < 2; ++binding) {
      const GLint width = binding ? 7 : 13, height = binding ? 11 : 19;
      glBindTexture(target, textures[binding]);
      if (array)
         storage3d(target, requested, GL_R8, width, height, layers, GL_TRUE);
      else
         glTexStorage2DMultisample(target, requested, GL_R8, width, height, GL_TRUE);
      if (!check_gl("allocate_query_source"))
         break;
      GLint actual_samples = 0, actual_width = 0, actual_height = 0, actual_layers = 1;
      glGetTexLevelParameteriv(target, 0, GL_TEXTURE_SAMPLES, &actual_samples);
      glGetTexLevelParameteriv(target, 0, GL_TEXTURE_WIDTH, &actual_width);
      glGetTexLevelParameteriv(target, 0, GL_TEXTURE_HEIGHT, &actual_height);
      if (array)
         glGetTexLevelParameteriv(target, 0, GL_TEXTURE_DEPTH, &actual_layers);
      if (!check_gl("query_allocated_storage"))
         break;
      if (actual_width != width || actual_height != height ||
          actual_layers != (GLint)layers || actual_samples < (GLint)requested) {
         fprintf(stderr, "STORAGE_ERROR\trequested=%u\tactual=%d\n",
                 requested, actual_samples);
         ++failures;
         break;
      }
      const unsigned before = failures;
      const GLuint sentinel[] = {0xdeadbeefu, 0xabcdef01u, 0x76543210u, 0xbaadf00du};
      glClearBufferuiv(GL_COLOR, 0, sentinel);
      glUseProgram(program);
      glUniform1i(glGetUniformLocation(program, "source_texture"), 0);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      ++draws;
      uint32_t observed[OUTPUT_WIDTH * OUTPUT_HEIGHT * 4];
      memset(observed, 0xa5, sizeof(observed));
      glReadPixels(0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT, GL_RGBA_INTEGER,
                   GL_UNSIGNED_INT, observed);
      ++reads;
      if (!check_gl("query_draw_readback"))
         break;
      const uint32_t expected[] = {(uint32_t)actual_width, (uint32_t)actual_height,
                                  (uint32_t)actual_layers, UINT32_C(0x51)};
      for (unsigned pixel = 0; pixel < OUTPUT_WIDTH * OUTPUT_HEIGHT; ++pixel) {
         for (unsigned channel = 0; channel < 4; ++channel) {
            const uint32_t actual = observed[pixel * 4 + channel];
            printf("%d\t%u\t%u\t%u\t%d\t%u\t%u\t%08" PRIx32 "\n",
                   array, layers, requested, binding, actual_samples,
                   pixel, channel, actual);
            ++checks;
            if (actual != expected[channel]) {
               fprintf(stderr, "VALUE_FAIL\tarray=%d\tlayers=%u\trequested=%u\t"
                       "binding=%u\tpixel=%u\tchannel=%u\tactual=%08" PRIx32
                       "\texpected=%08" PRIx32 "\n", array, layers, requested,
                       binding, pixel, channel, actual, expected[channel]);
               ++failures;
            }
         }
      }
      fprintf(stderr, "SCENARIO\tarray=%d\tlayers=%u\trequested=%u\tactual=%d\t"
              "binding=%u\tchecks=%s\n", array, layers, requested, actual_samples,
              binding, before == failures ? "Pass" : "Fail");
   }
   glDeleteTextures(2, textures);
   check_gl("query_source_cleanup");
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0, count = 0;
   EGLConfig config = NULL;
   EGLContext context = EGL_NO_CONTEXT;
   EGLSurface surface = EGL_NO_SURFACE;
   GLuint vao = 0, vbo = 0, output = 0, framebuffer = 0, programs[2] = {0};
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
   const EGLint surface_attributes[] = {EGL_WIDTH, 2, EGL_HEIGHT, 2, EGL_NONE};
   if (!eglBindAPI(EGL_OPENGL_ES_API) ||
       !eglChooseConfig(display, config_attributes, &config, 1, &count) || count != 1)
      goto cleanup;
   surface = eglCreatePbufferSurface(display, config, surface_attributes);
   context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      goto cleanup;
   current = 1;
   fprintf(stderr, "GL\tvendor=%s\trenderer=%s\tversion=%s\n",
           glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION));
   PFNGLTEXSTORAGE3DMULTISAMPLEOESPROC storage3d =
      (PFNGLTEXSTORAGE3DMULTISAMPLEOESPROC)eglGetProcAddress("glTexStorage3DMultisampleOES");
   if (!storage3d) {
      fprintf(stderr, "UNSUPPORTED\tglTexStorage3DMultisampleOES\n");
      goto cleanup;
   }
   programs[0] = query_program(0);
   programs[1] = query_program(1);
   if (!programs[0] || !programs[1]) {
      status = 1;
      goto cleanup;
   }
   const GLfloat triangle[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(triangle), triangle, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glGenTextures(1, &output);
   glBindTexture(GL_TEXTURE_2D, output);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, OUTPUT_WIDTH, OUTPUT_HEIGHT);
   glGenFramebuffers(1, &framebuffer);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
       !check_gl("query_output_setup")) {
      fprintf(stderr, "FBO_ERROR\tquery output\n");
      status = 1;
      goto cleanup;
   }
   glViewport(0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_BLEND);
   glDisable(GL_DITHER);
   glActiveTexture(GL_TEXTURE0);
   puts("array\tlayers\trequested\tbinding\tactual_samples\tpixel\tchannel\tu32_hex");
   for (unsigned samples = 1; samples <= 8; samples *= 2) {
      run_queries(storage3d, programs[0], samples, 0, 1);
      run_queries(storage3d, programs[1], samples, 1, 1);
      run_queries(storage3d, programs[1], samples, 1, 3);
   }
   if (draws != 24 || reads != 24 || checks != 384)
      ++failures;
   fprintf(stderr, "RESULT\tdraws=%u\treads=%u\tchecks=%u\tfailures=%u\n",
           draws, reads, checks, failures);
   status = failures ? 1 : 0;
cleanup:
   if (current) {
      glDeleteProgram(programs[0]);
      glDeleteProgram(programs[1]);
      glDeleteFramebuffers(1, &framebuffer);
      glDeleteTextures(1, &output);
      glDeleteBuffers(1, &vbo);
      glDeleteVertexArrays(1, &vao);
      if (!check_gl("query_cleanup"))
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
