/* SPDX-License-Identifier: MIT */
/* Every static binding affects an exact integer output through a real float
 * sample. Mutate each binding separately; retain high CB0 and a real UBO.
 * The native driver must not drop bindings or enlarge the FS256 budget. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
static void check(int ok, const char *message)
{
   ++checks;
   if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static void clean(const char *message)
{
   GLenum e = glGetError();
   if (e) fprintf(stderr, "GL_ERROR %s 0x%x\n", message, e);
   check(e == GL_NO_ERROR, message);
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint s = glCreateShader(stage);
   glShaderSource(s, 1, &source, NULL); glCompileShader(s);
   GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n%s\n", source, log); }
   check(ok, "compile"); return s;
}
int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint surf_attrs[] = {EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint ctx_attrs[] = {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attrs,&config,1,&count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display,config,surf_attrs);
   EGLContext context = eglCreateContext(display,config,EGL_NO_CONTEXT,ctx_attrs);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
      eglMakeCurrent(display,surface,surface,context), "EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLuint output, fbo, vao, vbo, ubo, textures[12];
   glGenTextures(1,&output); glBindTexture(GL_TEXTURE_2D,output);
   glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,4,4);
   glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,output,0);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"FBO complete");
   GLenum attachment = GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&attachment); glReadBuffer(attachment);
   const GLfloat positions[] = {-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
   glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER,vbo);
   glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);
   glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL); glEnableVertexAttribArray(0);
   glGenBuffers(1,&ubo); glBindBuffer(GL_UNIFORM_BUFFER,ubo);
   glBufferData(GL_UNIFORM_BUFFER,16,NULL,GL_DYNAMIC_DRAW); glBindBufferBase(GL_UNIFORM_BUFFER,0,ubo);
   glGenTextures(12,textures);
   for (unsigned i = 0; i < 12; ++i) {
      glActiveTexture(GL_TEXTURE0+i); glBindTexture(GL_TEXTURE_2D,textures[i]);
      glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32F,1,1);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
   }
   glViewport(0,0,4,4); glDisable(GL_DITHER); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST);
   glPixelStorei(GL_PACK_ALIGNMENT,1);
   const unsigned counts[] = {8,9,12}; unsigned draw = 0;
   for (unsigned variant = 0; variant < 3; ++variant) {
      const unsigned n = counts[variant], cb_vectors = n == 12 ? 3 : 14;
      char fs[8192]; size_t used = (size_t)snprintf(fs,sizeof(fs),
         "#version 310 es\nprecision highp float;precision highp int;\n"
         "uniform vec4 values[%u];layout(std140,binding=0) uniform Shared {vec4 bias;};\n", cb_vectors);
      for (unsigned i = 0; i < n; ++i)
         used += (size_t)snprintf(fs+used,sizeof(fs)-used,"uniform highp sampler2D t%u;\n",i);
      used += (size_t)snprintf(fs+used,sizeof(fs)-used,
         "layout(location=0) out uvec4 color;void main(){vec4 s=values[0]+values[%u]+bias;\n",cb_vectors-1);
      for (unsigned i = 0; i < n; ++i)
         used += (size_t)snprintf(fs+used,sizeof(fs)-used,"s+=texture(t%u,vec2(0.5))*%u.0;\n",i,i+1);
      used += (size_t)snprintf(fs+used,sizeof(fs)-used,"color=uvec4(s);}\n");
      check(used < sizeof(fs),"source bounded");
      const char *vs = "#version 310 es\nlayout(location=0) in vec4 position;void main(){gl_Position=position;}\n";
      GLuint p = glCreateProgram(), v = shader(GL_VERTEX_SHADER,vs), f = shader(GL_FRAGMENT_SHADER,fs);
      glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
      GLint ok = 0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
      if (!ok) { char log[4096]; glGetProgramInfoLog(p,sizeof(log),NULL,log); fprintf(stderr,"%s\n",log); }
      check(ok,"link"); glDeleteShader(v); glDeleteShader(f); glUseProgram(p);
      for (unsigned i = 0; i < n; ++i) {
         char name[16]; snprintf(name,sizeof(name),"t%u",i);
         GLint location = glGetUniformLocation(p,name); check(location >= 0,"every sampler active");
         glUniform1i(location,(GLint)i);
      }
      GLint location = glGetUniformLocation(p,"values[0]"); check(location >= 0,"CB0 active");
      GLuint block = glGetUniformBlockIndex(p,"Shared"); check(block != GL_INVALID_INDEX,"UBO active");
      glUniformBlockBinding(p,block,0);
      for (unsigned scenario = 0; scenario <= n; ++scenario) {
         GLfloat values[14][4] = {{0}}, bias[4]; GLuint expected[4] = {0};
         for (unsigned c = 0; c < 4; ++c) {
            values[0][c] = 2+c; values[cb_vectors-1][c] = 7+2*c+scenario;
            bias[c] = 11+3*c+2*scenario;
            expected[c] = (GLuint)(values[0][c]+values[cb_vectors-1][c]+bias[c]);
         }
         glUniform4fv(location,(GLsizei)cb_vectors,&values[0][0]);
         glBindBuffer(GL_UNIFORM_BUFFER,ubo); glBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(bias),bias);
         for (unsigned i = 0; i < n; ++i) {
            GLfloat rgba[4];
            for (unsigned c = 0; c < 4; ++c) {
               unsigned code = 3+5*i+2*c+(scenario == i+1 ? 19+3*c : 0);
               rgba[c] = (GLfloat)code; expected[c] += code*(i+1);
            }
            glActiveTexture(GL_TEXTURE0+i); glBindTexture(GL_TEXTURE_2D,textures[i]);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,1,1,GL_RGBA,GL_FLOAT,rgba);
         }
         clean("uploaded input"); glDrawArrays(GL_TRIANGLES,0,3); glFinish(); clean("native draw");
         GLuint pixels[64]; glReadPixels(0,0,4,4,GL_RGBA_INTEGER,GL_UNSIGNED_INT,pixels); clean("raw output");
         for (unsigned i = 0; i < 64; ++i) check(pixels[i] == expected[i%4],"all bindings and constants exact oracle");
         printf("DRAW %u textures=%u cb0_vectors=%u changed=%u rgba=%u,%u,%u,%u PASS\n",
            draw,n,cb_vectors,scenario,expected[0],expected[1],expected[2],expected[3]);
         printf("WORDS %u",draw); for (unsigned i = 0; i < 64; ++i) printf(" %u",pixels[i]);
         printf("\n"); fflush(stdout); ++draw;
      }
      glDeleteProgram(p);
   }
   glDeleteTextures(12,textures); glDeleteTextures(1,&output); glDeleteFramebuffers(1,&fbo);
   glDeleteBuffers(1,&ubo); glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao); clean("cleanup");
   printf("PASS draws=%u checks=%u\n",draw,checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
   eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display); return 0;
}
