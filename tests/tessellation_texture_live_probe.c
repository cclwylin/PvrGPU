/* SPDX-License-Identifier: MIT */
/* Manual live GLES 3.1 + EXT_tessellation_shader integration regression.
 * Build with the selected Mesa EGL/GLES headers and libraries, then execute
 * unchanged on llvmpipe and pvrgpu with an explicit native model bridge.
 * No version/capability override, recorded shaders, or reference pixels.
 *
 * Eight draws: dynamic per-invocation TCS LOD, ordinary/explicit TES LOD,
 * independent stage samplers and resource rebinding. Guarded TES transform
 * feedback, one-primitive query, and all RGBA32UI pixels have a CPU oracle
 * derived solely from exact uploaded float values. Both stages must sample
 * real textures: no shader replacement or injected sampled results.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 4, GUARD = 8, TF_WORDS = GUARD + 12 + GUARD };
static unsigned checks, draws;
static void check(int good, const char *message)
{
   ++checks;
   if (!good) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static void check_gl(const char *message)
{
   GLenum error = glGetError();
   if (error) fprintf(stderr, "GL_ERROR %s: 0x%x\n", message, error);
   check(error == GL_NO_ERROR, message);
}
static uint32_t bits(float value)
{
   uint32_t result; memcpy(&result, &value, sizeof(result)); return result;
}
static int extension(const char *name)
{
   GLint count = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &count);
   for (GLint i = 0; i < count; ++i)
      if (!strcmp((const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i), name)) return 1;
   return 0;
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint object = glCreateShader(stage);
   glShaderSource(object, 1, &source, NULL); glCompileShader(object);
   GLint ok = 0; glGetShaderiv(object, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[8192]; glGetShaderInfoLog(object, sizeof(log), NULL, log);
      fprintf(stderr, "SHADER 0x%x:\n%s\n%s\n", stage, log, source);
   }
   check(ok, "compile real tessellation texture shader");
   return object;
}
static GLuint program(unsigned explicit_lod)
{
   const char *vs = "#version 310 es\n"
      "layout(location=0) in highp vec4 position;\n"
      "void main(){gl_Position=position;}\n";
   const char *tc = "#version 310 es\n"
      "#extension GL_EXT_tessellation_shader : require\n"
      "precision highp float; precision highp int;\n"
      "layout(vertices=3) out;\n"
      "uniform highp sampler2D control_texture;\n"
      "out highp vec4 control_sample[];\n"
      "void main(){\n"
      "gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;\n"
      "control_sample[gl_InvocationID]=textureLod(control_texture,vec2(.375,.625),float(gl_InvocationID));\n"
      "if(gl_InvocationID==0){gl_TessLevelInner[0]=1.0;"
      "gl_TessLevelOuter[0]=1.0;gl_TessLevelOuter[1]=1.0;gl_TessLevelOuter[2]=1.0;}\n"
      "}\n";
   const char *te_prefix = "#version 310 es\n"
      "#extension GL_EXT_tessellation_shader : require\n"
      "precision highp float; precision highp int;\n"
      "layout(triangles,equal_spacing,ccw) in;\n"
      "uniform highp sampler2D evaluation_texture; uniform highp float evaluation_lod;\n"
      "in highp vec4 control_sample[]; flat out highp vec4 observed;\n"
      "void main(){\n"
      "gl_Position=gl_TessCoord.x*gl_in[0].gl_Position+gl_TessCoord.y*gl_in[1].gl_Position+gl_TessCoord.z*gl_in[2].gl_Position;\n"
      "observed=control_sample[0]+control_sample[1]+control_sample[2]+";
   const char *fs = "#version 310 es\nprecision highp float; precision highp int;\n"
      "flat in highp vec4 observed; layout(location=0) out highp uvec4 color;\n"
      "void main(){color=floatBitsToUint(observed);}\n";
   char te[2048];
   int length = snprintf(te, sizeof(te), "%s%s", te_prefix, explicit_lod ?
      "textureLod(evaluation_texture,vec2(.375,.625),evaluation_lod);}\n" :
      "texture(evaluation_texture,vec2(.375,.625));}\n");
   check(length > 0 && (size_t)length < sizeof(te), "TES source bound");
   const GLenum stages[] = {GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER_EXT,
      GL_TESS_EVALUATION_SHADER_EXT, GL_FRAGMENT_SHADER};
   const char *sources[] = {vs, tc, te, fs};
   GLuint object = glCreateProgram();
   for (unsigned i = 0; i < 4; ++i) {
      GLuint s = shader(stages[i], sources[i]);
      glAttachShader(object, s); glDeleteShader(s);
   }
   const char *varying = "observed";
   glTransformFeedbackVaryings(object, 1, &varying, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(object);
   GLint ok = 0; glGetProgramiv(object, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[8192]; glGetProgramInfoLog(object, sizeof(log), NULL, log);
      fprintf(stderr, "LINK: %s\n", log);
   }
   check(ok, "link native tessellation texture program");
   return object;
}
int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint config_attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
   const EGLint surface_attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   const EGLint context_attributes[] = {EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) &&
      eglChooseConfig(display, config_attributes, &config, 1, &count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display, surface, surface, context), "EGL current");
   fprintf(stderr, "RENDERER: %s\nVERSION: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   if (!extension("GL_EXT_tessellation_shader")) {
      fprintf(stderr, "UNSUPPORTED: public GL_EXT_tessellation_shader\n"); return 2;
   }
   PFNGLPATCHPARAMETERIEXTPROC patch_parameter =
      (PFNGLPATCHPARAMETERIEXTPROC)eglGetProcAddress("glPatchParameteriEXT");
   check(patch_parameter != NULL, "public EXT patch entrypoint");
   GLuint programs[] = {program(0), program(1)};
   GLuint textures[2], output, framebuffer, vao, buffers[2], query;
   glGenTextures(2, textures);
   for (unsigned texture = 0; texture < 2; ++texture) {
      glBindTexture(GL_TEXTURE_2D, textures[texture]);
      glTexStorage2D(GL_TEXTURE_2D, 3, GL_RGBA32F, 4, 4);
      for (unsigned level = 0; level < 3; ++level) {
         float values[16 * 4];
         unsigned side = 4U >> level;
         for (unsigned pixel = 0; pixel < side * side; ++pixel)
            for (unsigned channel = 0; channel < 4; ++channel)
               values[pixel * 4 + channel] = (float)((texture ? 16U : 1U) * (level + 1U) * (1U << channel));
         glTexSubImage2D(GL_TEXTURE_2D, (GLint)level, 0, 0, (GLsizei)side, (GLsizei)side, GL_RGBA, GL_FLOAT, values);
      }
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 2);
   }
   glGenTextures(1, &output); glBindTexture(GL_TEXTURE_2D, output);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, SIDE, SIDE);
   glGenFramebuffers(1, &framebuffer); glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output, 0);
   const GLenum color = GL_COLOR_ATTACHMENT0; glDrawBuffers(1, &color); glReadBuffer(color);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "RGBA32UI FBO complete");
   const float positions[] = {-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
   glGenVertexArrays(1, &vao); glBindVertexArray(vao);
   glGenBuffers(2, buffers); glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL); glEnableVertexAttribArray(0);
   glGenQueries(1, &query); patch_parameter(GL_PATCH_VERTICES_EXT, 3);
   glViewport(0, 0, SIDE, SIDE); glDisable(GL_BLEND); glDisable(GL_DITHER);
   glDisable(GL_CULL_FACE); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST);
   check_gl("texture and draw setup");
   for (unsigned explicit_lod = 0; explicit_lod < 2; ++explicit_lod)
      for (unsigned swap = 0; swap < 2; ++swap)
         for (unsigned lod = 0; lod < 3; lod += 2) {
            glUseProgram(programs[explicit_lod]);
            GLint tc = glGetUniformLocation(programs[explicit_lod], "control_texture");
            GLint te = glGetUniformLocation(programs[explicit_lod], "evaluation_texture");
            GLint te_lod = glGetUniformLocation(programs[explicit_lod], "evaluation_lod");
            check(tc >= 0 && te >= 0 && (!explicit_lod || te_lod >= 0), "active independent stage samplers");
            glUniform1i(tc, 0); glUniform1i(te, 1);
            if (explicit_lod) glUniform1f(te_lod, (float)lod);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textures[swap]);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, textures[!swap]);
            uint32_t expected[4], initial[TF_WORDS];
            for (unsigned channel = 0; channel < 4; ++channel)
               expected[channel] = bits((float)((6U * (swap ? 16U : 1U) +
                  ((explicit_lod ? lod : 0U) + 1U) * (swap ? 1U : 16U)) * (1U << channel)));
            for (unsigned word = 0; word < TF_WORDS; ++word) initial[word] = 0xdead0000U ^ (word * 0x1021U);
            glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffers[1]);
            glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(initial), initial, GL_DYNAMIC_READ);
            glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffers[1], GUARD * 4, 12 * 4);
            const GLuint clear[] = {0,0,0,0}; glClearBufferuiv(GL_COLOR, 0, clear);
            glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query);
            glBeginTransformFeedback(GL_TRIANGLES); glDrawArrays(GL_PATCHES_EXT, 0, 3);
            glEndTransformFeedback(); glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
            glFinish(); check_gl("live native TCS/TES texture draw");
            GLuint written = UINT32_MAX; glGetQueryObjectuiv(query, GL_QUERY_RESULT, &written);
            check(written == 1, "one complete tessellation triangle");
            const uint32_t *actual = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(initial), GL_MAP_READ_BIT);
            check(actual != NULL, "TES transform-feedback map");
            for (unsigned word = 0; word < TF_WORDS; ++word) {
               const uint32_t wanted = word >= GUARD && word < GUARD + 12 ? expected[(word - GUARD) % 4] : initial[word];
               if (actual[word] != wanted) fprintf(stderr, "DRAW %u TF %u actual=%08x expected=%08x\n", draws, word, actual[word], wanted);
               check(actual[word] == wanted, "raw TES values and TF prefix/tail guards");
               printf("TF %u %u %08x\n", draws, word, actual[word]);
            }
            check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER), "TES transform-feedback unmap");
            uint32_t pixels[SIDE * SIDE * 4];
            glReadPixels(0, 0, SIDE, SIDE, GL_RGBA_INTEGER, GL_UNSIGNED_INT, pixels);
            check_gl("RGBA32UI readback");
            for (unsigned word = 0; word < SIDE * SIDE * 4; ++word) {
               if (pixels[word] != expected[word % 4]) fprintf(stderr, "DRAW %u PIXELWORD %u actual=%08x expected=%08x\n", draws, word, pixels[word], expected[word % 4]);
               check(pixels[word] == expected[word % 4], "exact raw sampled color over complete output");
               printf("PIXEL %u %u %08x\n", draws, word, pixels[word]);
            }
            printf("DRAW %u explicit=%u swap=%u lod=%u primitives=%u PASS\n", draws, explicit_lod, swap, lod, written);
            ++draws;
         }
   glDeleteQueries(1, &query); glDeleteBuffers(2, buffers); glDeleteVertexArrays(1, &vao);
   glDeleteFramebuffers(1, &framebuffer); glDeleteTextures(1, &output); glDeleteTextures(2, textures);
   glDeleteProgram(programs[0]); glDeleteProgram(programs[1]); check_gl("cleanup");
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context); eglDestroySurface(display, surface); eglTerminate(display);
   printf("PASS draws=%u checks=%u\n", draws, checks);
   return 0;
}
