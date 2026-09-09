/* SPDX-License-Identifier: MIT */
/* Genuine GL findMSB and dependent texture-LOD tests with a CPU bit oracle.
 * No capture, reference image, or simulator-specific result is supplied. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIDE = 4 };
static unsigned checks, draws;
static void check(int good, const char *what)
{
   ++checks;
   if (!good) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
}
static void clean(const char *what)
{
   GLenum e = glGetError();
   if (e) fprintf(stderr, "GL_ERROR %s 0x%x\n", what, e);
   check(e == GL_NO_ERROR, what);
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint s = glCreateShader(stage);
   glShaderSource(s, 1, &source, NULL); glCompileShader(s);
   GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) { char log[8192]; glGetShaderInfoLog(s, sizeof(log), NULL, log); fprintf(stderr, "%s\n%s\n", source, log); }
   check(ok, "compile"); return s;
}
static GLuint program(unsigned mode)
{
   const char *vs = "#version 310 es\nprecision highp float;\n"
      "layout(location=0) in vec2 position;void main(){gl_Position=vec4(position,0,1);}\n";
   const char *bodies[] = {
      "uniform highp uvec4 words;layout(location=0) out highp uvec4 color;"
      "void main(){color=uvec4(findMSB(words));}",
      "uniform highp uvec4 words;layout(location=0) out highp uvec4 color;"
      "void main(){color=uvec4(findMSB(ivec4(words)));}",
      "uniform highp float radius;uniform highp sampler2D image;"
      "layout(location=0) out highp uvec4 color;"
      "void main(){int bit=findMSB(uint(abs(radius)));int level=clamp(bit-2,0,3);"
      "vec4 texel=textureLod(image,vec2(.5),float(level));"
      "color=uvec4(uint(level),uint(texel.r),uint(bit),uint(texel.a));}",
      "uniform highp float radius;uniform highp sampler2D image;"
      "layout(location=0) out highp float color;"
      "void main(){int bit=findMSB(uint(abs(radius)));int level=clamp(bit-2,0,3);"
      "color=textureLod(image,vec2(.5),float(level)).r/3.0;}"
   };
   char fs[2048];
   int n = snprintf(fs, sizeof(fs), "#version 310 es\nprecision highp float;precision highp int;\n%s\n", bodies[mode]);
   check(n > 0 && (size_t)n < sizeof(fs), "source bound");
   GLuint p = glCreateProgram(), v = shader(GL_VERTEX_SHADER, vs), f = shader(GL_FRAGMENT_SHADER, fs);
   glAttachShader(p, v); glAttachShader(p, f); glDeleteShader(v); glDeleteShader(f); glLinkProgram(p);
   GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) { char log[8192]; glGetProgramInfoLog(p, sizeof(log), NULL, log); fprintf(stderr, "%s\n", log); }
   check(ok, "link"); return p;
}
static int msb(uint32_t x)
{
   int result = -1;
   while (x) { ++result; x >>= 1; }
   return result;
}
int main(void)
{
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(d != EGL_NO_DISPLAY && eglInitialize(d, NULL, NULL), "EGL initialize");
   const EGLint config_attrs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
   const EGLint surface_attrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   const EGLint context_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(d, config_attrs, &config, 1, &count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(d, config, surface_attrs);
   EGLContext context = eglCreateContext(d, config, EGL_NO_CONTEXT, context_attrs);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT && eglMakeCurrent(d, surface, surface, context), "EGL current");
   fprintf(stderr, "RENDERER: %s\nVERSION: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   GLuint programs[4]; for (unsigned i = 0; i < 4; ++i) programs[i] = program(i);
   GLuint textures[3], fbo, vao, buffer;
   glGenTextures(3, textures);
   glBindTexture(GL_TEXTURE_2D, textures[0]); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, SIDE, SIDE);
   glBindTexture(GL_TEXTURE_2D, textures[1]); glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, SIDE, SIDE);
   glBindTexture(GL_TEXTURE_2D, textures[2]); glTexStorage2D(GL_TEXTURE_2D, 4, GL_RGBA32F, 8, 8);
   for (unsigned level = 0; level < 4; ++level) {
      unsigned size = 8u >> level; float pixels[8 * 8 * 4];
      for (unsigned i = 0; i < size * size; ++i) {
         pixels[i * 4] = (float)level; pixels[i * 4 + 1] = 11;
         pixels[i * 4 + 2] = 22; pixels[i * 4 + 3] = (float)(77 + level);
      }
      glTexSubImage2D(GL_TEXTURE_2D, level, 0, 0, size, size, GL_RGBA, GL_FLOAT, pixels);
   }
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   GLenum attachment = GL_COLOR_ATTACHMENT0; glDrawBuffers(1, &attachment); glReadBuffer(attachment);
   const float positions[] = {-1, -1, 1, -1, -1, 1, 1, 1};
   glGenVertexArrays(1, &vao); glBindVertexArray(vao); glGenBuffers(1, &buffer);
   glBindBuffer(GL_ARRAY_BUFFER, buffer); glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL); glEnableVertexAttribArray(0);
   glViewport(0, 0, SIDE, SIDE); glDisable(GL_DITHER); glDisable(GL_BLEND);
   glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE); glDisable(GL_SCISSOR_TEST);
   clean("setup");
   const float radii[] = {0, .25f, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256,
      1024, 1048576, -1, -4, -8, -16, -32, -128};
   for (unsigned mode = 0; mode < 4; ++mode) {
      glUseProgram(programs[mode]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, textures[mode == 3], 0);
      check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "FBO");
      GLint input = glGetUniformLocation(programs[mode], mode < 2 ? "words" : "radius"); check(input >= 0, "input uniform");
      if (mode >= 2) { GLint image = glGetUniformLocation(programs[mode], "image"); check(image >= 0, "sampler uniform"); glUniform1i(image, 0); }
      unsigned cases = mode < 2 ? 32 : sizeof(radii) / sizeof(radii[0]);
      for (unsigned c = 0; c < cases; ++c) {
         uint32_t expected[4];
         if (mode < 2) {
            uint32_t bit = UINT32_C(1) << c;
            uint32_t words[] = {bit, bit - 1u, bit | UINT32_C(0x55555555), ~bit};
            glUniform4uiv(input, 1, words);
            for (unsigned k = 0; k < 4; ++k) {
               uint32_t magnitude = mode == 1 && (words[k] & UINT32_C(0x80000000)) ? ~words[k] : words[k];
               expected[k] = (uint32_t)msb(magnitude);
            }
         } else {
            float radius = radii[c]; glUniform1f(input, radius);
            int bit = msb((uint32_t)(radius < 0 ? -radius : radius));
            int level = bit - 2; if (level < 0) level = 0; if (level > 3) level = 3;
            expected[0] = level; expected[1] = level; expected[2] = (uint32_t)bit; expected[3] = 77 + level;
         }
         glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); glFinish(); clean("findMSB draw");
         if (mode != 3) {
            uint32_t pixels[SIDE * SIDE * 4];
            glReadPixels(0, 0, SIDE, SIDE, GL_RGBA_INTEGER, GL_UNSIGNED_INT, pixels); clean("integer readback");
            for (unsigned i = 0; i < SIDE * SIDE * 4; ++i) {
               if (pixels[i] != expected[i % 4]) fprintf(stderr, "DRAW%u word%u actual%08x expected%08x\n", draws, i, pixels[i], expected[i % 4]);
               check(pixels[i] == expected[i % 4], "MSB and dependent texture level oracle");
            }
         } else {
            uint8_t pixels[SIDE * SIDE * 4];
            glReadPixels(0, 0, SIDE, SIDE, GL_RGBA, GL_UNSIGNED_BYTE, pixels); clean("R8 readback");
            for (unsigned i = 0; i < SIDE * SIDE * 4; ++i) {
               unsigned value = i % 4 == 0 ? expected[0] * 85 : i % 4 == 3 ? 255 : 0;
               if (pixels[i] != value) fprintf(stderr, "DRAW%u byte%u actual%u expected%u\n", draws, i, pixels[i], value);
               check(pixels[i] == value, "dependent scalar R8 output and absent channel defaults");
            }
         }
         printf("DRAW %u mode=%u case=%u expected=%08x,%08x,%08x,%08x PASS\n", draws, mode, c, expected[0], expected[1], expected[2], expected[3]);
         fflush(stdout); ++draws;
      }
   }
   for (unsigned i = 0; i < 4; ++i) glDeleteProgram(programs[i]);
   glDeleteBuffers(1, &buffer); glDeleteVertexArrays(1, &vao); glDeleteFramebuffers(1, &fbo); glDeleteTextures(3, textures); clean("cleanup");
   printf("PASS draws=%u checks=%u\n", draws, checks);
   eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); eglDestroyContext(d, context); eglDestroySurface(d, surface); eglTerminate(d);
   return 0;
}
