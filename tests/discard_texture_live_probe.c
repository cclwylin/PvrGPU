/* SPDX-License-Identifier: MIT */
/* Real EGL/GLES render, never a sampler-response or framebuffer substitute.
 * mode 0: discard -> implicit LOD texture -> derivative.
 * mode 1: texture/derivative + image atomic, geometric helpers must not store.
 * mode 2: both; an unsupported native contract must remain an explicit failure.
 * Arguments: output-directory width(2|8) mask(0=none,1=all,2=checker) triangle mode. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned failures;
static void check(int good, const char *where) {
   if (!good) { fprintf(stderr, "CHECK_FAILED %s\n", where); ++failures; }
}
static void check_gl(const char *where) {
   GLenum error;
   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "GL_ERROR %s 0x%x\n", where, error); ++failures;
   }
}
static GLuint compile(GLenum stage, const char *source) {
   GLuint shader = glCreateShader(stage);
   GLint ok = 0;
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[8192]; glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "SHADER_ERROR %s\n%s\n", log, source); exit(2);
   }
   return shader;
}
static int killed(unsigned pattern, unsigned x, unsigned y) {
   return pattern == 1 || (pattern == 2 && ((x + y) & 1) == 0);
}
static void save(const char *directory, const char *name, const void *bytes, size_t size) {
   char path[4096];
   check(snprintf(path, sizeof(path), "%s/%s", directory, name) < (int)sizeof(path), "output path");
   FILE *file = fopen(path, "wb");
   if (!file) { perror(path); exit(2); }
   check(fwrite(bytes, 1, size, file) == size, "save observed bytes");
   check(fclose(file) == 0, "close observed bytes");
}

int main(int argc, char **argv) {
   if (argc != 6) return 2;
   const unsigned size = (unsigned)atoi(argv[2]), pattern = (unsigned)atoi(argv[3]);
   const int triangle = atoi(argv[4]), mode = atoi(argv[5]);
   if ((size != 2 && size != 8) || pattern > 2 || triangle < 0 || triangle > 1 || mode < 0 || mode > 2) return 2;
   const int do_discard = mode != 1, do_image = mode != 0;
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor, count;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
      fprintf(stderr, "EGL_ERROR initialize 0x%x\n", eglGetError()); return 2;
   }
   const EGLint attrs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
   EGLConfig config;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display, attrs, &config, 1, &count) && count == 1,
         "EGL config");
   const EGLint context_attrs[] = {EGL_CONTEXT_MAJOR_VERSION_KHR, 3,
      EGL_CONTEXT_MINOR_VERSION_KHR, 1, EGL_NONE};
   const EGLint surface_attrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   check(context != EGL_NO_CONTEXT && surface != EGL_NO_SURFACE &&
      eglMakeCurrent(display, surface, surface, context), "EGL current");
   if (failures) return 2;
   fprintf(stderr, "GL renderer=%s version=%s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   const char *vs_source = "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "void main(){gl_Position=position;}\n";
   char fs_source[4096];
   snprintf(fs_source, sizeof(fs_source),
      "#version 310 es\n%s\nprecision highp float; precision highp int;\n"
      "layout(binding=0) uniform highp sampler2D maskTex;\n"
      "layout(binding=1) uniform highp sampler2D colorTex;\n"
      "%s\nlayout(location=0) out vec4 color;\n"
      "void main(){vec2 uv=gl_FragCoord.xy*(1.0/%u.0);\n"
      "float mask=texture(maskTex,uv).r;\n%s\n"
      "vec4 value=texture(colorTex,uv);\n"
      /* Plain GLSL derivatives may legally use coarse or fine differences.
       * Their magnitudes agree for this checker, while signs need not. */
      "float dx=abs(dFdx(mask)),dy=abs(dFdy(mask));\n%s\n"
      "color=vec4(value.r,0.5+0.25*dx,0.5+0.25*dy,1.0);}\n",
      do_image ? "#extension GL_OES_shader_image_atomic : require" : "",
      do_image ? "layout(r32ui,binding=0) uniform highp uimage2D writes;" : "",
      size, do_discard ? "if(mask<0.5)discard;" : "",
      do_image ? "imageAtomicAdd(writes,ivec2(gl_FragCoord.xy),1u);" : "");
   save(argv[1], "fragment.glsl", fs_source, strlen(fs_source));
   GLuint vs = compile(GL_VERTEX_SHADER, vs_source), fs = compile(GL_FRAGMENT_SHADER, fs_source);
   GLuint program = glCreateProgram(); glAttachShader(program, vs); glAttachShader(program, fs); glLinkProgram(program);
   GLint linked; glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) { char log[8192]; glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "LINK_ERROR %s\n", log); return 2; }
   glUseProgram(program);
   GLuint textures[4]; glGenTextures(4, textures);
   uint8_t rgba[16 * 16 * 4];
   glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textures[0]);
   for (unsigned y = 0; y < size; ++y) for (unsigned x = 0; x < size; ++x) {
      uint8_t *p = rgba + 4 * (y * size + x);
      p[0] = killed(pattern, x, y) ? 0 : 255; p[1] = p[2] = 0; p[3] = 255;
   }
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, textures[1]);
   for (unsigned level = 0, width = 2 * size; width; ++level, width >>= 1) {
      for (unsigned i = 0; i < width * width; ++i) {
         rgba[4*i] = (uint8_t)(32 + 40 * level); rgba[4*i+1] = 0;
         rgba[4*i+2] = 0; rgba[4*i+3] = 255;
      }
      glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA8, width, width, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
   }
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   // Separate texture units keep the original sampler bindings intact.
   glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, textures[2]);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, size, size);
   GLuint fb[2]; glGenFramebuffers(2, fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb[0]);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[2], 0);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "RGBA FBO complete");
   uint32_t atomics[64]; for (unsigned i = 0; i < size * size; ++i) atomics[i] = 100;
   if (do_image) {
      glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, textures[3]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, size, size);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size, size, GL_RED_INTEGER, GL_UNSIGNED_INT, atomics);
      glBindImageTexture(0, textures[3], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
   }
   const float end = triangle ? 1.125f : 3.0f;
   const float vertices[] = {-1,-1,0,1, end,-1,0,1, -1,end,0,1};
   GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
   glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL); glEnableVertexAttribArray(0);
   glViewport(0, 0, size, size); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE); glDisable(GL_SCISSOR_TEST); glDisable(GL_DITHER);
   glClearColor(17.f/255.f,33.f/255.f,49.f/255.f,1); glClear(GL_COLOR_BUFFER_BIT);
   check_gl("before draw");
   glDrawArrays(GL_TRIANGLES, 0, 3); glFinish();
   if (do_image) glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);
   check_gl("draw finish");
   memset(rgba, 0xa5, size * size * 4);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
   check_gl("read color"); save(argv[1], "color.rgba8", rgba, size * size * 4);
   if (do_image) {
      glBindFramebuffer(GL_READ_FRAMEBUFFER, fb[1]);
      glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[3], 0);
      check(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "image read FBO complete");
      glReadPixels(0, 0, size, size, GL_RED_INTEGER, GL_UNSIGNED_INT, atomics);
      check_gl("read image"); save(argv[1], "image.r32ui", atomics, size * size * 4);
   }
   unsigned visible = 0, rejected = 0, image_bad = 0;
   for (unsigned y = 0; y < size; ++y) for (unsigned x = 0; x < size; ++x) {
      const unsigned i = y * size + x;
      const int covered = !triangle || (float)(x+y+1) < 1.0625f * size;
      const int kept = covered && (!do_discard || !killed(pattern,x,y));
      const uint8_t clear[] = {17,33,49,255};
      if (kept) { ++visible; check(memcmp(rgba+4*i, clear, 4) != 0, "surviving fragment drawn"); }
      else { ++rejected; check(memcmp(rgba+4*i, clear, 4) == 0, "discard/helper color not committed"); }
      if (do_image && atomics[i] != (uint32_t)(100 + kept)) ++image_bad;
   }
   check(image_bad == 0, "discard/helper image atomic not committed");
   fprintf(stderr, "RESULT width=%u pattern=%u triangle=%d mode=%d visible=%u rejected=%u image_bad=%u failures=%u\n",
      size, pattern, triangle, mode, visible, rejected, image_bad, failures);
   glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao); glDeleteFramebuffers(2,fb);
   glDeleteTextures(4,textures); glDeleteProgram(program); glDeleteShader(vs); glDeleteShader(fs);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
   eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
   return failures ? 1 : 0;
}
