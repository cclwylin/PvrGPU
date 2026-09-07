/* SPDX-License-Identifier: MIT */
/* 真實 EGL/GLES3 A/B 探針：所有像素均由 GL shader、MSAA 與 resolve 產生。
 * 使用單取樣 pbuffer 和自建 4/8 倍 FBO，避免預設多取樣 surface 的問題。
 * stdout 是逐像素 TSV；stderr 是實際環境、錯誤及基本不變量驗證。
 * 本程式不比較中間 alpha 的預備答案，也不辨識 renderer 以選擇答案。 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kWidth = 8, kHeight = 8 };
static unsigned failures;
static unsigned scenarios;
static GLuint program;
static GLint color_location;
static GLint depth_location;

static void
check_gl(const char *where)
{
   GLenum error;
   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "GL_ERROR\t%s\t0x%04x\n", where, error);
      ++failures;
   }
}

static GLuint
compile_shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   GLint ok = 0;
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "SHADER_ERROR\t%s\n", log);
      glDeleteShader(shader);
      return 0;
   }
   return shader;
}

static int
make_program(void)
{
   static const char *vs_source =
      "#version 300 es\n"
      "layout(location=0) in highp vec2 position;\n"
      "uniform highp float depth;\n"
      "void main(){ gl_Position=vec4(position,depth,1.0); }\n";
   static const char *fs_source =
      "#version 300 es\n"
      "precision highp float;\n"
      "uniform highp vec4 color;\n"
      "layout(location=0) out highp vec4 result;\n"
      "void main(){ result=color; }\n";
   GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
   if (!vs || !fs) {
      glDeleteShader(vs);
      glDeleteShader(fs);
      return 0;
   }
   program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);
   GLint ok = 0;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "LINK_ERROR\t%s\n", log);
      return 0;
   }
   color_location = glGetUniformLocation(program, "color");
   depth_location = glGetUniformLocation(program, "depth");
   return color_location >= 0 && depth_location >= 0;
}

static void
draw_quad(float red, float green, float blue, float alpha, float depth)
{
   glUniform4f(color_location, red, green, blue, alpha);
   glUniform1f(depth_location, depth);
   glDrawArrays(GL_TRIANGLES, 0, 6);
}

static void
reset_target(GLuint framebuffer, int dither)
{
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glViewport(0, 0, kWidth, kHeight);
   glDisable(GL_BLEND);
   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_CULL_FACE);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_STENCIL_TEST);
   glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
   glDisable(GL_SAMPLE_COVERAGE);
   if (dither)
      glEnable(GL_DITHER);
   else
      glDisable(GL_DITHER);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDepthMask(GL_TRUE);
   glStencilMask(0xff);
   glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
   glClearDepthf(1.0f);
   glClearStencil(0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
   check_gl("reset_target");
}

static void
emit_pixels(GLuint source, GLuint resolve, unsigned samples, int dither,
            const char *scene, const char *coverage, float alpha,
            const GLubyte *expected)
{
   GLubyte pixels[kWidth * kHeight * 4];
   memset(pixels, 0xa5, sizeof(pixels));
   glBindFramebuffer(GL_READ_FRAMEBUFFER, source);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolve);
   glBlitFramebuffer(0, 0, kWidth, kHeight, 0, 0, kWidth, kHeight,
                     GL_COLOR_BUFFER_BIT, GL_NEAREST);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, resolve);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, kWidth, kHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   check_gl(scene);
   ++scenarios;
   unsigned wrong = 0;
   for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
         const GLubyte *pixel = pixels + (y * kWidth + x) * 4;
         printf("%u\t%d\t%s\t%s\t%.9g\t%u\t%u\t%02x%02x%02x%02x\t%u\t%u\t%u\t%u\n",
                samples, dither, scene, coverage, (double)alpha, x, y,
                pixel[0], pixel[1], pixel[2], pixel[3],
                pixel[0], pixel[1], pixel[2], pixel[3]);
         if (expected && memcmp(pixel, expected, 4))
            ++wrong;
      }
   }
   if (wrong) {
      fprintf(stderr, "INVARIANT_FAIL\tsamples=%u\tcolor_dither=%d\tscene=%s\tcoverage=%s\talpha=%.9g\tpixels=%u\n",
              samples, dither, scene, coverage, (double)alpha, wrong);
      ++failures;
   }
}

static int
run_samples(unsigned samples)
{
   GLuint framebuffers[2] = {0, 0};
   GLuint renderbuffers[3] = {0, 0, 0};
   glGenFramebuffers(2, framebuffers);
   glGenRenderbuffers(3, renderbuffers);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[0]);
   glBindRenderbuffer(GL_RENDERBUFFER, renderbuffers[0]);
   glRenderbufferStorageMultisample(GL_RENDERBUFFER, (GLsizei)samples,
                                    GL_RGBA8, kWidth, kHeight);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_RENDERBUFFER, renderbuffers[0]);
   glBindRenderbuffer(GL_RENDERBUFFER, renderbuffers[1]);
   glRenderbufferStorageMultisample(GL_RENDERBUFFER, (GLsizei)samples,
                                    GL_DEPTH24_STENCIL8, kWidth, kHeight);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                             GL_RENDERBUFFER, renderbuffers[1]);
   GLint actual_samples = 0;
   glGetIntegerv(GL_SAMPLES, &actual_samples);
   GLenum source_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[1]);
   glBindRenderbuffer(GL_RENDERBUFFER, renderbuffers[2]);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, kWidth, kHeight);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_RENDERBUFFER, renderbuffers[2]);
   GLenum resolve_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   const unsigned before = failures;
   check_gl("create_framebuffers");
   fprintf(stderr, "FBO\trequested_samples=%u\tactual_samples=%d\tsource_status=0x%x\tresolve_status=0x%x\twidth=%d\theight=%d\n",
           samples, actual_samples, source_status, resolve_status, kWidth, kHeight);
   int ok = before == failures && actual_samples == (GLint)samples &&
            source_status == GL_FRAMEBUFFER_COMPLETE &&
            resolve_status == GL_FRAMEBUFFER_COMPLETE;
   if (!ok) {
      fprintf(stderr, "FBO_UNAVAILABLE\tsamples=%u\n", samples);
      ++failures;
      goto cleanup;
   }

   static const float alphas[] = {0.0f, 0.0001f, 0.125f, 0.25f,
                                  0.375f, 0.5f, 0.75f, 1.0f};
   static const float coverages[] = {1.0f, 0.5f, 0.5f};
   static const char *coverage_names[] = {"full", "half", "half_inverted"};
   static const GLubyte blue[] = {0, 0, 255, 255};
   static const GLubyte red[] = {255, 0, 0, 255};
   static const GLubyte green[] = {0, 255, 0, 255};
   for (int dither = 0; dither <= 1; ++dither) {
      for (unsigned coverage = 0; coverage < 3; ++coverage) {
         for (unsigned a = 0; a < sizeof(alphas) / sizeof(alphas[0]); ++a) {
            reset_target(framebuffers[0], dither);
            glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            glEnable(GL_SAMPLE_COVERAGE);
            glSampleCoverage(coverages[coverage], coverage == 2 ? GL_TRUE : GL_FALSE);
            draw_quad(1.0f, 0.0f, 0.0f, alphas[a], 0.0f);
            /* 只有端點是跨實作的基本不變量；中間門檻保留真實像素供 A/B。 */
            const GLubyte *expected = alphas[a] == 0.0f ? blue :
               (alphas[a] == 1.0f && coverage == 0 ? red : NULL);
            emit_pixels(framebuffers[0], framebuffers[1], samples, dither,
                        "alpha_coverage", coverage_names[coverage], alphas[a], expected);
         }
      }
   }

   static const char *preservation_names[] = {
      "depth_preservation", "stencil_preservation", "depth_stencil_preservation"
   };
   for (unsigned mode = 0; mode < 3; ++mode) {
      for (unsigned foreground = 0; foreground < 2; ++foreground) {
         reset_target(framebuffers[0], 1);
         if (mode != 1) {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
         }
         if (mode != 0) {
            glEnable(GL_STENCIL_TEST);
            glStencilFunc(GL_ALWAYS, 1, 0xff);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
         }
         /* 先畫近處前景，再畫遠處不透明綠色；alpha=0 不得寫入 Z/S。
          * alpha=1 是正向對照，必須阻擋後面的綠色。 */
         glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
         draw_quad(1.0f, 0.0f, 0.0f, (float)foreground, -0.5f);
         glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
         if (mode != 0) {
            glStencilFunc(GL_EQUAL, 0, 0xff);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
         }
         draw_quad(0.0f, 1.0f, 0.0f, 1.0f, 0.5f);
         emit_pixels(framebuffers[0], framebuffers[1], samples, 1,
                     preservation_names[mode], "disabled", (float)foreground,
                     foreground ? red : green);
      }
   }

cleanup:
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glBindRenderbuffer(GL_RENDERBUFFER, 0);
   glDeleteFramebuffers(2, framebuffers);
   glDeleteRenderbuffers(3, renderbuffers);
   check_gl("delete_framebuffers");
   return ok;
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
      fprintf(stderr, "EGL_INITIALIZE_ERROR\t0x%x\n", eglGetError());
      return 2;
   }
   fprintf(stderr, "EGL\tmajor=%d\tminor=%d\tvendor=%s\tversion=%s\n",
           major, minor, eglQueryString(display, EGL_VENDOR), eglQueryString(display, EGL_VERSION));
   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_SAMPLE_BUFFERS, 0, EGL_SAMPLES, 0, EGL_NONE
   };
   EGLint count = 0;
   EGLConfig config = NULL;
   EGLConfig *configs = NULL;
   EGLSurface surface = EGL_NO_SURFACE;
   EGLContext context = EGL_NO_CONTEXT;
   GLuint vao = 0, vbo = 0;
   int current = 0;
   int status = 2;
   if (!eglBindAPI(EGL_OPENGL_ES_API) ||
       !eglChooseConfig(display, config_attributes, NULL, 0, &count) || count <= 0)
      goto cleanup;
   configs = calloc((size_t)count, sizeof(*configs));
   if (!configs || !eglChooseConfig(display, config_attributes, configs, count, &count))
      goto cleanup;
   for (EGLint i = 0; i < count; ++i) {
      EGLint sample_buffers = -1, samples = -1;
      if (eglGetConfigAttrib(display, configs[i], EGL_SAMPLE_BUFFERS, &sample_buffers) &&
          eglGetConfigAttrib(display, configs[i], EGL_SAMPLES, &samples) &&
          sample_buffers == 0 && samples == 0) {
         config = configs[i];
         break;
      }
   }
   if (!config)
      goto cleanup;
   const EGLint pbuffer_attributes[] = {EGL_WIDTH, kWidth, EGL_HEIGHT, kHeight, EGL_NONE};
   const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
   surface = eglCreatePbufferSurface(display, config, pbuffer_attributes);
   context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      goto cleanup;
   current = 1;
   fprintf(stderr, "GL\tvendor=%s\trenderer=%s\tversion=%s\tglsl=%s\n",
           glGetString(GL_VENDOR), glGetString(GL_RENDERER),
           glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));
   GLint samples = -1, sample_buffers = -1;
   glGetIntegerv(GL_SAMPLES, &samples);
   glGetIntegerv(GL_SAMPLE_BUFFERS, &sample_buffers);
   fprintf(stderr, "PBUFFER\tsamples=%d\tsample_buffers=%d\n", samples, sample_buffers);
   /* GL_DITHER 只控制顏色量化；A2C dithering 的 NV 擴充狀態保持預設。
    * 不把兩個不同控制誤稱為同一個開關。 */
   fprintf(stderr, "DITHER\tcolor_dither=GL_DITHER\talpha_to_coverage_dither=GL_default_unchanged\n");
   check_gl("context");
   if (samples || sample_buffers || failures || !make_program())
      goto cleanup;
   static const GLfloat positions[] = {
      -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
      -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
   };
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), NULL);
   glEnableVertexAttribArray(0);
   glUseProgram(program);
   check_gl("geometry");
   puts("samples\tcolor_dither\tscene\tcoverage\talpha\tx\ty\trgba_hex\tr\tg\tb\ta");
   run_samples(4);
   run_samples(8);
   fprintf(stderr, "RESULT\tscenarios=%u\texpected_scenarios=108\tpixels_per_scenario=64\tfailures=%u\n", scenarios, failures);
   status = failures ? 1 : 0;

cleanup:
   if (current) {
      glDeleteBuffers(1, &vbo);
      glDeleteVertexArrays(1, &vao);
      glDeleteProgram(program);
      check_gl("delete_program_and_geometry");
      if (failures && status == 0)
         status = 1;
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   }
   if (context != EGL_NO_CONTEXT)
      eglDestroyContext(display, context);
   if (surface != EGL_NO_SURFACE)
      eglDestroySurface(display, surface);
   free(configs);
   if (status == 2)
      fprintf(stderr, "SETUP_FAILED\tegl_error=0x%x\n", eglGetError());
   eglTerminate(display);
   return status;
}
