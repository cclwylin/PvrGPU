/* SPDX-License-Identifier: MIT */
/* Actual GL proof of descriptor/CB0 pressure. The first program retains a
 * 96-word VS CB0 and shares a program-level UBO used only by FS. The second
 * reads that UBO in VS too, with a 92-word CB0 fitting the unchanged96 bank.
 * No reference shader substitution or driver-cap override. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
static void check(int ok, const char *message)
{
   ++checks;
   if (!ok) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
static void clean(const char *message)
{
   GLenum e=glGetError();
   if (e) fprintf(stderr,"GL_ERROR %s 0x%x\n",message,e);
   check(e==GL_NO_ERROR,message);
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint s=glCreateShader(stage); glShaderSource(s,1,&source,NULL); glCompileShader(s);
   GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
   if (!ok) { char log[4096]; glGetShaderInfoLog(s,sizeof(log),NULL,log); fprintf(stderr,"%s\n%s\n",source,log); }
   check(ok,"compile"); return s;
}
int main(void)
{
   EGLDisplay display=eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display!=EGL_NO_DISPLAY && eglInitialize(display,NULL,NULL),"EGL initialize");
   const EGLint attrs[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint surf_attrs[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint ctx_attrs[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config=NULL; EGLint count=0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attrs,&config,1,&count) && count==1,"EGL config");
   EGLSurface surface=eglCreatePbufferSurface(display,config,surf_attrs);
   EGLContext context=eglCreateContext(display,config,EGL_NO_CONTEXT,ctx_attrs);
   check(surface!=EGL_NO_SURFACE && context!=EGL_NO_CONTEXT && eglMakeCurrent(display,surface,surface,context),"EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLuint tex,fbo,vao,vbo,ubo;
   glGenTextures(1,&tex); glBindTexture(GL_TEXTURE_2D,tex); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,4,4);
   glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"FBO complete");
   GLenum attachment=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&attachment); glReadBuffer(attachment);
   const GLfloat positions[]={-1,-1,0,1,3,-1,0,1,-1,3,0,1};
   glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER,vbo);
   glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);
   glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL); glEnableVertexAttribArray(0);
   glGenBuffers(1,&ubo); glBindBuffer(GL_UNIFORM_BUFFER,ubo); glBufferData(GL_UNIFORM_BUFFER,16,NULL,GL_DYNAMIC_DRAW);
   glBindBufferBase(GL_UNIFORM_BUFFER,0,ubo);
   glViewport(0,0,4,4); glDisable(GL_DITHER); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST);
   glPixelStorei(GL_PACK_ALIGNMENT,1);
   for(unsigned live=0;live<2;++live) {
      char vs[1024];
      int n=snprintf(vs,sizeof(vs),
         "#version 310 es\nprecision highp float;\n"
         "layout(location=0) in vec4 position;uniform vec4 values[%u];\n"
         "layout(std140,binding=0) uniform Shared {vec4 bias;};\n"
         "out vec4 v;void main(){gl_Position=position+values[0];v=values[%u]%s;}\n",
         live?23:24,live?22:23,live?"+bias":"");
      check(n>0 && (size_t)n<sizeof(vs),"VS source size");
      const char *fs="#version 310 es\nprecision highp float;\n"
         "layout(std140,binding=0) uniform Shared {vec4 bias;};\n"
         "in vec4 v;layout(location=0) out vec4 color;void main(){color=v+bias;}\n";
      GLuint p=glCreateProgram(),v=shader(GL_VERTEX_SHADER,vs),f=shader(GL_FRAGMENT_SHADER,fs);
      glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
      GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
      if(!ok){char log[4096];glGetProgramInfoLog(p,sizeof(log),NULL,log);fprintf(stderr,"%s\n",log);}
      check(ok,"link");glDeleteShader(v);glDeleteShader(f);glUseProgram(p);
      GLint location=glGetUniformLocation(p,"values[0]");check(location>=0,"active CB0 array");
      GLuint block=glGetUniformBlockIndex(p,"Shared");check(block!=GL_INVALID_INDEX,"active shared UBO");
      glUniformBlockBinding(p,block,0);
      for(unsigned scenario=0;scenario<4;++scenario) {
         GLfloat values[24][4]={{0}},bias[4];
         unsigned expected[4];
         for(unsigned c=0;c<4;++c){
            unsigned value=16+scenario*12+c*8, delta=4+scenario+c;
            /* Multiples of1/256 are exactly representable; finalUNORM8 uses
             * nearest(value*255/256), calculated independently in integers. */
            values[live?22:23][c]=(GLfloat)value/256.0f;bias[c]=(GLfloat)delta/256.0f;
            unsigned sum=value+(live?2:1)*delta;expected[c]=(sum*255U+128U)/256U;
         }
         glUniform4fv(location,live?23:24,&values[0][0]);
         glBindBuffer(GL_UNIFORM_BUFFER,ubo);glBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(bias),bias);
         glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);
         glDrawArrays(GL_TRIANGLES,0,3);glFinish();clean("native pressure draw");
         unsigned char pixels[4*4*4];glReadPixels(0,0,4,4,GL_RGBA,GL_UNSIGNED_BYTE,pixels);clean("exact raw output");
         for(unsigned pixel=0;pixel<16;++pixel)for(unsigned c=0;c<4;++c)
            check(pixels[4*pixel+c]==expected[c],"CB0 and UBO output matches integer oracle");
         printf("DRAW %u live_vs_ubo=%u cb0_words=%u rgba=%u,%u,%u,%u PASS\n",
            live*4+scenario,live,live?92:96,expected[0],expected[1],expected[2],expected[3]);fflush(stdout);
      }
      glDeleteProgram(p);
   }
   glDeleteFramebuffers(1,&fbo);glDeleteTextures(1,&tex);glDeleteBuffers(1,&ubo);glDeleteBuffers(1,&vbo);glDeleteVertexArrays(1,&vao);
   clean("cleanup");printf("PASS draws=8 checks=%u\n",checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
   eglDestroyContext(display,context);eglDestroySurface(display,surface);eglTerminate(display);return 0;
}
