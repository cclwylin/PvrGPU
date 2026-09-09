/* Independent capture-shader-semantics probe. No capture/reference pixels.
 * GLSL is reconstructed from GLSL249 NIR: gather-max then optional odd-size
 * textureOffset samples. This variant explicitly preserves the extra+0.5
 * visible in captured NIR; the native driver already uses half-pixel coords.
 * Eight/nine-sized D32F inputs reduce to4x4, keeping interpolated UV dyadic.
 * Build: clang -std=c11 -O2 -Wall -Wextra -Werror plus Mesa EGL/GLES paths.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { SIDE=4, PIXELS=16, GUARD=16, MAX_TEXELS=81 };
static unsigned checks, passes;
static void check(int good,const char *why) { ++checks; if(!good){fprintf(stderr,"FAIL: %s\n",why);exit(1);} }
static void check_gl(const char *why) { GLenum e=glGetError(); if(e)fprintf(stderr,"GL_ERROR: %s %x\n",why,e);check(!e,why); }
static uint32_t bits(float x) {uint32_t u;memcpy(&u,&x,4);return u;}
static GLuint shader(GLenum type,const char *text) {
   GLuint s=glCreateShader(type);glShaderSource(s,1,&text,NULL);glCompileShader(s);
   GLint good=0;glGetShaderiv(s,GL_COMPILE_STATUS,&good);
   if(!good){char log[8192];glGetShaderInfoLog(s,sizeof(log),NULL,log);fprintf(stderr,"SHADER: %s\n",log);}
   check(good,"shader compile");return s;
}
static GLuint program(void) {
   const char *vs="#version 310 es\nlayout(location=0) in highp vec4 in_position;\n"
      "out mediump vec2 out_texcoord0;\n"
      "void main(){out_texcoord0=in_position.xy*0.5+0.5;gl_Position=vec4(in_position.xyz,1.0);}\n";
   const char *fs="#version 310 es\nprecision highp float;precision highp int;\n"
      "uniform highp sampler2D texture_unit0;uniform mediump ivec2 texture_size;\n"
      "in mediump vec2 out_texcoord0;\n"
      "void main(){vec4 d=textureGather(texture_unit0,out_texcoord0,0);\n"
      "float z=max(max(d.x,d.y),max(d.z,d.w));ivec2 q=ivec2(gl_FragCoord.xy+vec2(0.5))*2;\n"
      "bool oddx=(texture_size.x&1)!=0;bool oddy=(texture_size.y&1)!=0;\n"
      "if(oddx&&q.x==texture_size.x-3){\n"
      "if(oddy&&q.y==texture_size.y-3)z=max(z,textureOffset(texture_unit0,out_texcoord0,ivec2(1,1)).x);\n"
      "float a=textureOffset(texture_unit0,out_texcoord0,ivec2(1,0)).x;\n"
      "float b=textureOffset(texture_unit0,out_texcoord0,ivec2(1,-1)).x;z=max(z,max(a,b));\n"
      "}else if(oddy&&q.y==texture_size.y-3){\n"
      "float a=textureOffset(texture_unit0,out_texcoord0,ivec2(-1,1)).x;\n"
      "float b=textureOffset(texture_unit0,out_texcoord0,ivec2(0,1)).x;z=max(z,max(a,b));}\n"
      "gl_FragDepth=z;}\n";
   GLuint p=glCreateProgram(),v=shader(GL_VERTEX_SHADER,vs),f=shader(GL_FRAGMENT_SHADER,fs);
   glAttachShader(p,v);glAttachShader(p,f);glDeleteShader(v);glDeleteShader(f);glLinkProgram(p);
   GLint good=0;glGetProgramiv(p,GL_LINK_STATUS,&good);
   if(!good){char log[8192];glGetProgramInfoLog(p,sizeof(log),NULL,log);fprintf(stderr,"LINK: %s\n",log);}
   check(good,"program link");return p;
}
static unsigned clamp(int value,unsigned extent) {return value<0?0:value>=(int)extent?extent-1:(unsigned)value;}
static void gather_axis(float uv,unsigned extent,unsigned *a,unsigned *b) {
   volatile float s=uv*(float)extent;
   volatile float low=s-.5f,high=s+.5f;
   *a=clamp((int)low,extent);*b=clamp((int)high,extent);
}
static float sample(const float *input,unsigned w,unsigned h,float u,float v,int dx,int dy) {
   volatile float x=u*(float)w,y=v*(float)h;
   return input[clamp((int)floorf(y)+dy,h)*w+clamp((int)floorf(x)+dx,w)];
}
static float expected(const float *input,unsigned w,unsigned h,unsigned x,unsigned y,unsigned *branch) {
   float u=((float)x+.5f)/SIDE,v=((float)y+.5f)/SIDE;
   unsigned x0,x1,y0,y1;gather_axis(u,w,&x0,&x1);gather_axis(v,h,&y0,&y1);
   float z=fmaxf(fmaxf(input[y1*w+x0],input[y1*w+x1]),fmaxf(input[y0*w+x1],input[y0*w+x0]));
   const int take_x=(w&1U)&&2*(x+1)==w-3, take_y=(h&1U)&&2*(y+1)==h-3;
   *branch=take_x?(take_y?3:1):(take_y?2:0);
   if(take_x){
      if(take_y)z=fmaxf(z,sample(input,w,h,u,v,1,1));
      z=fmaxf(z,fmaxf(sample(input,w,h,u,v,1,0),sample(input,w,h,u,v,1,-1)));
   }else if(take_y)z=fmaxf(z,fmaxf(sample(input,w,h,u,v,-1,1),sample(input,w,h,u,v,0,1)));
   return z;
}
int main(void) {
   EGLDisplay display=eglGetDisplay(EGL_DEFAULT_DISPLAY);check(display!=EGL_NO_DISPLAY&&eglInitialize(display,NULL,NULL),"EGL initialize");
   const EGLint ca[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint sa[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE},ctxa[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config=NULL;EGLint count=0;
   check(eglBindAPI(EGL_OPENGL_ES_API)&&eglChooseConfig(display,ca,&config,1,&count)&&count==1,"EGL config");
   EGLSurface surface=eglCreatePbufferSurface(display,config,sa);EGLContext ctx=eglCreateContext(display,config,EGL_NO_CONTEXT,ctxa);
   check(surface!=EGL_NO_SURFACE&&ctx!=EGL_NO_CONTEXT&&eglMakeCurrent(display,surface,surface,ctx),"EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLuint p=program();glUseProgram(p);
   GLint texture=glGetUniformLocation(p,"texture_unit0"),size=glGetUniformLocation(p,"texture_size");
   check(texture>=0&&size>=0,"active texture/size uniforms");glUniform1i(texture,0);
   GLuint fbo[2],output,vao,vbo;glGenFramebuffers(2,fbo);glGenTextures(1,&output);glBindTexture(GL_TEXTURE_2D,output);
   glTexStorage2D(GL_TEXTURE_2D,1,GL_DEPTH_COMPONENT32F,SIDE,SIDE);glBindFramebuffer(GL_FRAMEBUFFER,fbo[0]);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,output,0);
   const GLenum none=GL_NONE;glDrawBuffers(1,&none);glReadBuffer(GL_NONE);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"D32F output FBO");
   const float vertices[]={-1,-1,1,-1,-1,1,1,1};glGenVertexArrays(1,&vao);glBindVertexArray(vao);
   glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);
   glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,NULL);glEnableVertexAttribArray(0);
   glViewport(0,0,SIDE,SIDE);glDisable(GL_BLEND);glDisable(GL_CULL_FACE);glDisable(GL_DITHER);glDisable(GL_SCISSOR_TEST);
   glEnable(GL_DEPTH_TEST);glDepthFunc(GL_ALWAYS);glDepthMask(GL_TRUE);glPixelStorei(GL_PACK_ALIGNMENT,1);glPixelStorei(GL_UNPACK_ALIGNMENT,1);
   const unsigned dimensions[4][2]={{8,8},{8,9},{9,8},{9,9}};
   for(unsigned test=0;test<4;++test){
      unsigned w=dimensions[test][0],h=dimensions[test][1],texels=w*h;
      float input[MAX_TEXELS];for(unsigned i=0;i<texels;++i)input[i]=(float)(i+1)/128.f;
      GLuint source;glGenTextures(1,&source);glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,source);
      glTexStorage2D(GL_TEXTURE_2D,1,GL_DEPTH_COMPONENT32F,(GLsizei)w,(GLsizei)h);
      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,(GLsizei)w,(GLsizei)h,GL_DEPTH_COMPONENT,GL_FLOAT,input);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_COMPARE_MODE,GL_NONE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_BASE_LEVEL,0);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
      glBindFramebuffer(GL_FRAMEBUFFER,fbo[1]);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,source,0);glDrawBuffers(1,&none);glReadBuffer(GL_NONE);
      check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"input D32F FBO");
      uint32_t upload[GUARD+MAX_TEXELS+GUARD];for(unsigned i=0;i<GUARD+MAX_TEXELS+GUARD;++i)upload[i]=0xdeadbeef;
      glReadPixels(0,0,(GLsizei)w,(GLsizei)h,GL_DEPTH_COMPONENT,GL_FLOAT,upload+GUARD);check_gl("input readback");
      for(unsigned i=0;i<GUARD;++i)check(upload[i]==0xdeadbeef&&upload[GUARD+texels+i]==0xdeadbeef,"input guards");
      for(unsigned i=0;i<texels;++i){check(upload[GUARD+i]==bits(input[i]),"exact input oracle");printf("INPUT %u %u %08x\n",test,i,upload[GUARD+i]);}
      glBindFramebuffer(GL_FRAMEBUFFER,fbo[0]);const float clear=1.f;glClearBufferfv(GL_DEPTH,0,&clear);glUniform2i(size,(GLint)w,(GLint)h);
      check_gl("before reduction draw");fprintf(stderr,"DRAW: test=%u source=%ux%u output=4x4\n",test,w,h);
      glDrawArrays(GL_TRIANGLE_STRIP,0,4);glFinish();check_gl("reduction draw");
      uint32_t actual[GUARD+PIXELS+GUARD];for(unsigned i=0;i<GUARD+PIXELS+GUARD;++i)actual[i]=0xa5a5a5a5;
      glReadPixels(0,0,SIDE,SIDE,GL_DEPTH_COMPONENT,GL_FLOAT,actual+GUARD);check_gl("depth readback");
      for(unsigned i=0;i<GUARD;++i)check(actual[i]==0xa5a5a5a5&&actual[GUARD+PIXELS+i]==0xa5a5a5a5,"output guards");
      for(unsigned i=0;i<PIXELS;++i){unsigned branch=0;uint32_t value=bits(expected(input,w,h,i%SIDE,i/SIDE,&branch));
         if(actual[GUARD+i]!=value)fprintf(stderr,"MISMATCH: test=%u pixel=%u branch=%u actual=%08x expected=%08x\n",test,i,branch,actual[GUARD+i],value);
         check(actual[GUARD+i]==value,"depth reduction independent oracle");printf("DEPTH %u %u %u %08x\n",test,i,branch,actual[GUARD+i]);}
      ++passes;fprintf(stderr,"PASS: test=%u source=%ux%u\n",test,w,h);
      glBindFramebuffer(GL_FRAMEBUFFER,fbo[1]);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,0,0);glDeleteTextures(1,&source);
   }
   glDeleteProgram(p);glDeleteVertexArrays(1,&vao);glDeleteBuffers(1,&vbo);glDeleteFramebuffers(2,fbo);glDeleteTextures(1,&output);check_gl("cleanup");
   fprintf(stderr,"PASS: cases=%u checks=%u\n",passes,checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);eglDestroyContext(display,ctx);eglDestroySurface(display,surface);eglTerminate(display);return 0;
}
