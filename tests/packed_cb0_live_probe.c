/* SPDX-License-Identifier: MIT */
/* Eight genuine samplers + UBO + sparse default uniforms, including a
 * four-element runtime-indexed scalar array. A fitting control precedes the
 * otherwise-over-budget program. No reference pixels or shader replacement. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
static void check(int ok,const char *why) { ++checks;if(!ok){fprintf(stderr,"FAIL: %s\n",why);exit(1);} }
static void clean(const char *why) {GLenum e=glGetError();if(e)fprintf(stderr,"GL_ERROR %s %x\n",why,e);check(e==GL_NO_ERROR,why);}
static GLuint shader(GLenum stage,const char *source) {
   GLuint s=glCreateShader(stage);glShaderSource(s,1,&source,NULL);glCompileShader(s);
   GLint ok=0;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
   if(!ok){char log[4096];glGetShaderInfoLog(s,sizeof(log),NULL,log);fprintf(stderr,"%s\n%s\n",source,log);}
   check(ok,"compile shader");return s;
}
static GLint location(GLuint p,const char *name) {GLint i=glGetUniformLocation(p,name);check(i>=0,name);return i;}
int main(void) {
   EGLDisplay display=eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display!=EGL_NO_DISPLAY&&eglInitialize(display,NULL,NULL),"EGL initialize");
   const EGLint attrs[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint sa[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint ca[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config=NULL;EGLint n=0;
   check(eglBindAPI(EGL_OPENGL_ES_API)&&eglChooseConfig(display,attrs,&config,1,&n)&&n==1,"EGL config");
   EGLSurface surface=eglCreatePbufferSurface(display,config,sa);EGLContext context=eglCreateContext(display,config,EGL_NO_CONTEXT,ca);
   check(surface!=EGL_NO_SURFACE&&context!=EGL_NO_CONTEXT&&eglMakeCurrent(display,surface,surface,context),"EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLuint output,fbo,vao,vbo,ubo,textures[8];glGenTextures(1,&output);glBindTexture(GL_TEXTURE_2D,output);
   glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,4,4);glGenFramebuffers(1,&fbo);glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,output,0);
   GLenum attachment=GL_COLOR_ATTACHMENT0;glDrawBuffers(1,&attachment);glReadBuffer(attachment);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"output FBO");
   const GLfloat positions[]={-1,-1,0,1,3,-1,0,1,-1,3,0,1};
   glGenVertexArrays(1,&vao);glBindVertexArray(vao);glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);
   glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);glEnableVertexAttribArray(0);
   glGenBuffers(1,&ubo);glBindBuffer(GL_UNIFORM_BUFFER,ubo);glBufferData(GL_UNIFORM_BUFFER,16,NULL,GL_DYNAMIC_DRAW);glBindBufferBase(GL_UNIFORM_BUFFER,0,ubo);
   glGenTextures(8,textures);
   for(unsigned i=0;i<8;++i){glActiveTexture(GL_TEXTURE0+i);glBindTexture(GL_TEXTURE_2D,textures[i]);
      glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32F,1,1);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);}
   glViewport(0,0,4,4);glDisable(GL_DITHER);glDisable(GL_BLEND);glDisable(GL_DEPTH_TEST);glPixelStorei(GL_PACK_ALIGNMENT,1);
   unsigned draw=0;
   for(unsigned variant=0;variant<2;++variant) {
      const unsigned vectors=variant?40:2;char fs[8192];
      size_t used=(size_t)snprintf(fs,sizeof(fs),"#version 310 es\nprecision highp float;precision highp int;\n"
         "uniform float lead;uniform vec4 values[%u];uniform float choices[4];uniform int selector;\n"
         "layout(std140,binding=0) uniform Shared{vec4 bias;};layout(location=0) out uvec4 color;\n",vectors);
      for(unsigned i=0;i<8;++i)used+=(size_t)snprintf(fs+used,sizeof(fs)-used,"uniform highp sampler2D t%u;\n",i);
      used+=(size_t)snprintf(fs+used,sizeof(fs)-used,"void main(){vec4 s=values[0]+values[%u]+vec4(lead)+vec4(choices[selector])+bias;\n",vectors-1);
      for(unsigned i=0;i<8;++i)used+=(size_t)snprintf(fs+used,sizeof(fs)-used,"s+=texture(t%u,vec2(0.5))*%u.0;\n",i,i+1);
      used+=(size_t)snprintf(fs+used,sizeof(fs)-used,"color=uvec4(s);}\n");check(used<sizeof(fs),"bounded GLSL");
      const char *vs="#version 310 es\nlayout(location=0) in vec4 position;void main(){gl_Position=position;}\n";
      GLuint p=glCreateProgram(),v=shader(GL_VERTEX_SHADER,vs),f=shader(GL_FRAGMENT_SHADER,fs);
      glAttachShader(p,v);glAttachShader(p,f);glLinkProgram(p);GLint ok=0;glGetProgramiv(p,GL_LINK_STATUS,&ok);
      if(!ok){char log[4096];glGetProgramInfoLog(p,sizeof(log),NULL,log);fprintf(stderr,"%s\n",log);}check(ok,"link");glDeleteShader(v);glDeleteShader(f);glUseProgram(p);
      GLint lead=location(p,"lead"),values=location(p,"values[0]"),choices=location(p,"choices[0]"),selector=location(p,"selector");
      GLuint block=glGetUniformBlockIndex(p,"Shared");check(block!=GL_INVALID_INDEX,"active UBO");glUniformBlockBinding(p,block,0);
      for(unsigned i=0;i<8;++i){char name[8];snprintf(name,sizeof(name),"t%u",i);glUniform1i(location(p,name),(GLint)i);}
      for(unsigned selected=0;selected<4;++selected) for(unsigned change=0;change<14;++change) {
         GLfloat vectors_data[40][4],options[4],bias[4];GLuint expected[4]={0};
         const GLfloat leading=3+(change==1?17:0);
         for(unsigned i=0;i<vectors;++i)for(unsigned c=0;c<4;++c)vectors_data[i][c]=(GLfloat)(701+i*3+c);
         for(unsigned i=0;i<4;++i)options[i]=(GLfloat)(11+5*i+(change==4&&i==selected?29:0));
         for(unsigned c=0;c<4;++c){vectors_data[0][c]=(GLfloat)(2+c+(change==2?21+c:0));
            vectors_data[vectors-1][c]=(GLfloat)(7+2*c+(change==3?23+2*c:0));bias[c]=(GLfloat)(13+3*c+(change==5?31:0));
            expected[c]=(GLuint)(vectors_data[0][c]+vectors_data[vectors-1][c]+leading+options[selected]+bias[c]);}
         glUniform1f(lead,leading);glUniform4fv(values,(GLsizei)vectors,&vectors_data[0][0]);glUniform1fv(choices,4,options);glUniform1i(selector,(GLint)selected);
         glBindBuffer(GL_UNIFORM_BUFFER,ubo);glBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(bias),bias);
         for(unsigned i=0;i<8;++i){GLfloat rgba[4];for(unsigned c=0;c<4;++c){unsigned code=3+5*i+2*c+(change==6+i?37+c:0);rgba[c]=(GLfloat)code;expected[c]+=code*(i+1);}
            glActiveTexture(GL_TEXTURE0+i);glBindTexture(GL_TEXTURE_2D,textures[i]);glTexSubImage2D(GL_TEXTURE_2D,0,0,0,1,1,GL_RGBA,GL_FLOAT,rgba);}
         clean("input uploads");fprintf(stderr,"DRAW_BEGIN %u variant=%u selector=%u change=%u\n",draw,variant,selected,change);
         glDrawArrays(GL_TRIANGLES,0,3);glFinish();clean("draw finish");
         GLuint pixels[96];for(unsigned i=0;i<96;++i)pixels[i]=0xdeadbeef;
         glReadPixels(0,0,4,4,GL_RGBA_INTEGER,GL_UNSIGNED_INT,pixels+16);clean("raw output");
         for(unsigned i=0;i<16;++i)check(pixels[i]==0xdeadbeef&&pixels[80+i]==0xdeadbeef,"output guards");
         for(unsigned i=0;i<64;++i){if(pixels[16+i]!=expected[i%4])fprintf(stderr,"MISMATCH draw=%u word=%u actual=%u expected=%u\n",draw,i,pixels[16+i],expected[i%4]);check(pixels[16+i]==expected[i%4],"all samplers/CB0/dynamic array/UBO independent oracle");}
         printf("DRAW %u variant=%u vectors=%u selector=%u change=%u PASS\n",draw,variant,vectors,selected,change);
         printf("WORDS %u",draw);for(unsigned i=0;i<64;++i)printf(" %u",pixels[16+i]);printf("\n");fflush(stdout);++draw;
      }
      glDeleteProgram(p);
   }
   glDeleteTextures(8,textures);glDeleteTextures(1,&output);glDeleteFramebuffers(1,&fbo);glDeleteBuffers(1,&ubo);glDeleteBuffers(1,&vbo);glDeleteVertexArrays(1,&vao);clean("cleanup");
   printf("PASS draws=%u checks=%u\n",draw,checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);eglDestroyContext(display,context);eglDestroySurface(display,surface);eglTerminate(display);return 0;
}
