/* SPDX-License-Identifier: MIT */
/* Real GLES EXT tessellation integration test for coalesced dynamic UBO
 * reads. All 16 words of each matrix influence guarded TES feedback and
 * every integer color pixel. Same executable on llvmpipe and native model;
 * no reference shader or sampled result is supplied to native execution. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 4, GUARD = 8, TF_WORDS = GUARD + 12 + GUARD, BLOCK_BYTES = 192 };
static unsigned checks, draws;
static void check(int good, const char *message)
{
   ++checks;
   if (!good) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static void clean(const char *message)
{
   GLenum error = glGetError();
   if (error) fprintf(stderr, "GL_ERROR %s 0x%x\n", message, error);
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
   GLuint s = glCreateShader(stage); glShaderSource(s, 1, &source, NULL); glCompileShader(s);
   GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) { char log[8192]; glGetShaderInfoLog(s, sizeof(log), NULL, log); fprintf(stderr, "%s\n%s\n", source, log); }
   check(ok, "compile real UBO shader"); return s;
}
static GLuint program(unsigned mode)
{
   const char *vs = "#version 310 es\nlayout(location=0) in highp vec4 position;"
      "void main(){gl_Position=position;}\n";
   char tc[2048], te[2048];
   int n = snprintf(tc, sizeof(tc),
      "#version 310 es\n#extension GL_EXT_tessellation_shader : require\n"
      "precision highp float; precision highp int; layout(vertices=3) out;\n"
      "layout(std140,binding=0) uniform Control {mat4 control[3];};\n"
      "out highp vec4 values[]; void main(){\n"
      "gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;\n"
      "int i=gl_InvocationID; values[gl_InvocationID]=%s;\n"
      "if(gl_InvocationID==0){gl_TessLevelInner[0]=1.0;"
      "gl_TessLevelOuter[0]=1.0;gl_TessLevelOuter[1]=1.0;gl_TessLevelOuter[2]=1.0;} }\n",
      mode != 1 ? "control[i][0]+control[i][1]+control[i][2]+control[i][3]" : "vec4(0)");
   check(n > 0 && (size_t)n < sizeof(tc), "TCS source bound");
   n = snprintf(te, sizeof(te),
      "#version 310 es\n#extension GL_EXT_tessellation_shader : require\n"
      "precision highp float; precision highp int; layout(triangles,equal_spacing,ccw) in;\n"
      "layout(std140,binding=1) uniform Evaluation {mat4 evaluation[3];};\n"
      "uniform highp int matrix_index; in highp vec4 values[]; flat out highp vec4 observed;\n"
      "void main(){gl_Position=gl_TessCoord.x*gl_in[0].gl_Position+"
      "gl_TessCoord.y*gl_in[1].gl_Position+gl_TessCoord.z*gl_in[2].gl_Position;\n"
      "int i=matrix_index; observed=values[0]+values[1]+values[2]+%s;}\n",
      mode != 0 ? "evaluation[i][0]+evaluation[i][1]+evaluation[i][2]+evaluation[i][3]" : "vec4(0)");
   check(n > 0 && (size_t)n < sizeof(te), "TES source bound");
   const char *fs = "#version 310 es\nprecision highp float; precision highp int;\n"
      "flat in highp vec4 observed; layout(location=0) out highp uvec4 color;"
      "void main(){color=floatBitsToUint(observed);}\n";
   const GLenum stages[] = {GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER_EXT, GL_TESS_EVALUATION_SHADER_EXT, GL_FRAGMENT_SHADER};
   const char *sources[] = {vs, tc, te, fs};
   GLuint p = glCreateProgram();
   for (unsigned i = 0; i < 4; ++i) { GLuint s = shader(stages[i], sources[i]); glAttachShader(p, s); glDeleteShader(s); }
   const char *varying = "observed"; glTransformFeedbackVaryings(p, 1, &varying, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(p); GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) { char log[8192]; glGetProgramInfoLog(p, sizeof(log), NULL, log); fprintf(stderr, "%s\n", log); }
   check(ok, "link real tessellation UBO program"); return p;
}
static unsigned word_value(unsigned buffer, unsigned page, unsigned matrix, unsigned column, unsigned channel)
{
   return (1 + buffer * 32 + page * 8 + matrix * 4 + column) * (1U << channel);
}
int main(void)
{
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(d != EGL_NO_DISPLAY && eglInitialize(d, NULL, NULL), "EGL initialize");
   const EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint sa[] = {EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(d,attrs,&config,1,&count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(d,config,sa);
   EGLContext context = eglCreateContext(d,config,EGL_NO_CONTEXT,ca);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT && eglMakeCurrent(d,surface,surface,context), "EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   check(extension("GL_EXT_tessellation_shader"), "public EXT tessellation support");
   PFNGLPATCHPARAMETERIEXTPROC patch = (PFNGLPATCHPARAMETERIEXTPROC)eglGetProcAddress("glPatchParameteriEXT");
   check(patch != NULL, "public EXT patch entrypoint");
   GLuint programs[] = {program(0), program(1), program(2)};
   GLuint ubo[2], tex, fbo, vao, buffers[2], query;
   GLint alignment = 0; glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &alignment);
   check(alignment > 0 && alignment <= 65536, "bounded UBO alignment");
   size_t stride = ((BLOCK_BYTES + (size_t)alignment - 1) / (size_t)alignment) * (size_t)alignment;
   glGenBuffers(2, ubo);
   for (unsigned b = 0; b < 2; ++b) {
      unsigned char *storage = malloc(stride * 2); check(storage != NULL, "UBO allocation");
      memset(storage, 0xa7, stride * 2);
      for (unsigned page = 0; page < 2; ++page) {
         float words[48];
         for (unsigned m = 0; m < 3; ++m) for (unsigned c = 0; c < 4; ++c) for (unsigned k = 0; k < 4; ++k)
            words[m * 16 + c * 4 + k] = (float)word_value(b,page,m,c,k);
         memcpy(storage + page * stride, words, sizeof(words));
      }
      glBindBuffer(GL_UNIFORM_BUFFER, ubo[b]); glBufferData(GL_UNIFORM_BUFFER,(GLsizeiptr)(stride * 2),storage,GL_STATIC_DRAW); free(storage);
   }
   glGenTextures(1,&tex); glBindTexture(GL_TEXTURE_2D,tex); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,SIDE,SIDE);
   glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0);
   GLenum color = GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&color); glReadBuffer(color);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "integer output FBO");
   const float positions[] = {-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
   glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(2,buffers);
   glBindBuffer(GL_ARRAY_BUFFER,buffers[0]); glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);
   glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL); glEnableVertexAttribArray(0);
   glGenQueries(1,&query); patch(GL_PATCH_VERTICES_EXT,3);
   glViewport(0,0,SIDE,SIDE); glDisable(GL_BLEND); glDisable(GL_DITHER); glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE);
   clean("setup");
   for (unsigned mode = 0; mode < 3; ++mode) for (unsigned swap = 0; swap < 2; ++swap)
      for (unsigned page = 0; page < 2; ++page) for (unsigned matrix = 0; matrix < 3; matrix += 2) {
         glUseProgram(programs[mode]);
         GLint index = glGetUniformLocation(programs[mode],"matrix_index");
         check(mode == 0 || index >= 0, "active TES runtime index");
         if (index >= 0) glUniform1i(index,(GLint)matrix);
         glBindBufferRange(GL_UNIFORM_BUFFER,0,ubo[swap],(GLintptr)(page * stride),BLOCK_BYTES);
         glBindBufferRange(GL_UNIFORM_BUFFER,1,ubo[!swap],(GLintptr)((1-page) * stride),BLOCK_BYTES);
         uint32_t expected[4], initial[TF_WORDS];
         for (unsigned k = 0; k < 4; ++k) {
            unsigned sum = 0;
            if (mode != 1) for (unsigned m = 0; m < 3; ++m) for (unsigned c = 0; c < 4; ++c) sum += word_value(swap,page,m,c,k);
            if (mode != 0) for (unsigned c = 0; c < 4; ++c) sum += word_value(!swap,1-page,matrix,c,k);
            expected[k] = bits((float)sum);
         }
         for (unsigned i = 0; i < TF_WORDS; ++i) initial[i] = 0xdead0000U ^ (i * 0x1021U);
         glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER,buffers[1]); glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,sizeof(initial),initial,GL_DYNAMIC_READ);
         glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,0,buffers[1],GUARD * 4,12 * 4);
         const GLuint zero[] = {0,0,0,0}; glClearBufferuiv(GL_COLOR,0,zero);
         glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN,query); glBeginTransformFeedback(GL_TRIANGLES);
         glDrawArrays(GL_PATCHES_EXT,0,3); glEndTransformFeedback(); glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
         glFinish(); clean("real wide UBO draw");
         GLuint written = UINT32_MAX; glGetQueryObjectuiv(query,GL_QUERY_RESULT,&written); check(written == 1, "one TES triangle");
         const uint32_t *actual = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,0,sizeof(initial),GL_MAP_READ_BIT);
         check(actual != NULL, "feedback map");
         for (unsigned i = 0; i < TF_WORDS; ++i) {
            uint32_t wanted = i >= GUARD && i < GUARD + 12 ? expected[(i-GUARD)%4] : initial[i];
            if (actual[i] != wanted) fprintf(stderr,"TF draw=%u word=%u actual=%08x expected=%08x\n",draws,i,actual[i],wanted);
            check(actual[i] == wanted,"exact feedback and guards");
         }
         check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER),"feedback unmap");
         uint32_t pixels[SIDE * SIDE * 4]; glReadPixels(0,0,SIDE,SIDE,GL_RGBA_INTEGER,GL_UNSIGNED_INT,pixels); clean("integer readback");
         for (unsigned i = 0; i < SIDE * SIDE * 4; ++i) check(pixels[i] == expected[i%4],"all color pixels match independent integer oracle");
         printf("DRAW %u mode=%u swap=%u page=%u matrix=%u rgba=%08x,%08x,%08x,%08x PASS\n",draws,mode,swap,page,matrix,expected[0],expected[1],expected[2],expected[3]); fflush(stdout); ++draws;
      }
   glDeleteQueries(1,&query); glDeleteBuffers(2,ubo); glDeleteBuffers(2,buffers); glDeleteVertexArrays(1,&vao);
   glDeleteFramebuffers(1,&fbo); glDeleteTextures(1,&tex); for (unsigned i = 0; i < 3; ++i) glDeleteProgram(programs[i]); clean("cleanup");
   printf("PASS draws=%u checks=%u\n",draws,checks);
   eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT); eglDestroyContext(d,context); eglDestroySurface(d,surface); eglTerminate(d);
   return 0;
}
