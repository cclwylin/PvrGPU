/* SPDX-License-Identifier: MIT */
/* Manual genuine GLES CubeArray non-shadow depth probe. Synthetic inputs only.
 * Exact uploaded depth codes are checked before sampling. Each face/mip is
 * spatially constant but distinct, so interior N/L samples have an independent
 * normalized-depth oracle without assuming a seamless cube filter algorithm.
 * LP words are reported separately: oracle tolerance is not bit-exact parity.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CUBES=2,FACES=6,LEVELS=3,SIDE=2,PIXELS=4,GUARD=16 };
static unsigned checks,draws,uploads;
static void check(int ok,const char *s){++checks;if(!ok){fprintf(stderr,"FAIL: %s\n",s);exit(1);}}
static void glcheck(const char *s){GLenum e=glGetError();if(e)fprintf(stderr,"GL_ERROR %s 0x%x\n",s,e);check(e==GL_NO_ERROR,s);}
static uint32_t bits(float f){uint32_t u;memcpy(&u,&f,4);return u;}
static float value(uint32_t u){float f;memcpy(&f,&u,4);return f;}
static GLuint shader(GLenum type,const char *s){
   GLuint n=glCreateShader(type);glShaderSource(n,1,&s,NULL);glCompileShader(n);GLint ok=0;glGetShaderiv(n,GL_COMPILE_STATUS,&ok);
   if(!ok){char log[8192];glGetShaderInfoLog(n,sizeof(log),NULL,log);fprintf(stderr,"SHADER %s\n%s\n",log,s);}check(ok,"shader compile");return n;
}
typedef struct {GLuint id;GLint coordinate,lod;} Program;
static Program program(unsigned explicit_lod){
   const char *vs="#version 310 es\nlayout(location=0) in highp vec4 position;void main(){gl_Position=position;}\n";
   char fs[1024];snprintf(fs,sizeof(fs),"#version 310 es\n#extension GL_EXT_texture_cube_map_array : require\n"
      "precision highp float;precision highp int;uniform highp samplerCubeArray source_texture;"
      "uniform highp vec4 coordinate;uniform highp float selected_lod;layout(location=0) out highp uvec4 color;"
      "void main(){color=floatBitsToUint(%s);}\n",explicit_lod?"textureLod(source_texture,coordinate,selected_lod)":"texture(source_texture,coordinate)");
   GLuint v=shader(GL_VERTEX_SHADER,vs),f=shader(GL_FRAGMENT_SHADER,fs);Program p={0};p.id=glCreateProgram();glAttachShader(p.id,v);glAttachShader(p.id,f);
   glLinkProgram(p.id);GLint ok=0;glGetProgramiv(p.id,GL_LINK_STATUS,&ok);if(!ok){char log[8192];glGetProgramInfoLog(p.id,sizeof(log),NULL,log);fprintf(stderr,"LINK %s\n",log);}check(ok,"program link");
   glDeleteShader(v);glDeleteShader(f);glUseProgram(p.id);GLint t=glGetUniformLocation(p.id,"source_texture");
   p.coordinate=glGetUniformLocation(p.id,"coordinate");p.lod=glGetUniformLocation(p.id,"selected_lod");
   check(t>=0&&p.coordinate>=0&&(explicit_lod?p.lod>=0:p.lod==-1),"active texture/coordinate/LOD uniforms");glUniform1i(t,0);glcheck("program state");return p;
}
static unsigned identity(unsigned layer,unsigned mip){return (layer/6)*18+(layer%6)*3+mip+1;}
static uint32_t code(unsigned format,unsigned layer,unsigned mip){
   unsigned id=identity(layer,mip);return format==0?id*1536u:format==1?id*393216u:bits((float)id/64.f);
}
static double normalized(unsigned format,unsigned layer,unsigned mip){uint32_t c=code(format,layer,mip);return format==0?(double)c/65535.0:format==1?(double)c/16777215.0:(double)value(c);}
static void direction(unsigned face,float d[3]){
   /* Face center also avoids cross-face weights in the final 1x1 mip. */
   const float sc=0.f,tc=0.f;
   switch(face){case 0:d[0]=1;d[1]=-tc;d[2]=-sc;break;case 1:d[0]=-1;d[1]=-tc;d[2]=sc;break;
   case 2:d[0]=sc;d[1]=1;d[2]=tc;break;case 3:d[0]=sc;d[1]=-1;d[2]=-tc;break;
   case 4:d[0]=sc;d[1]=-tc;d[2]=1;break;default:d[0]=-sc;d[1]=-tc;d[2]=-1;break;}
}
static void source_upload(unsigned format,GLuint input,GLuint fbo){
   const GLenum internal[3]={GL_DEPTH_COMPONENT16,GL_DEPTH_COMPONENT24,GL_DEPTH_COMPONENT32F};
   const GLenum types[3]={GL_UNSIGNED_SHORT,GL_UNSIGNED_INT,GL_FLOAT};
   glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,input);glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,LEVELS,internal[format],4,4,CUBES*FACES);glcheck("depth CubeArray storage");
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_BASE_LEVEL,0);glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MAX_LEVEL,LEVELS-1);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_COMPARE_MODE,GL_NONE);
   glTexParameterf(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MIN_LOD,0);glTexParameterf(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MAX_LOD,1000);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_SWIZZLE_R,GL_RED);glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_SWIZZLE_G,GL_RED);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_SWIZZLE_B,GL_RED);glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_SWIZZLE_A,GL_ONE);glcheck("non-shadow depth sampler state");
   glBindFramebuffer(GL_FRAMEBUFFER,fbo);const GLenum none=GL_NONE;glDrawBuffers(1,&none);glReadBuffer(GL_NONE);
   for(unsigned mip=0;mip<LEVELS;mip++){
      unsigned n=4u>>mip;uint16_t u16[192];uint32_t u32[192];float f32[192];
      for(unsigned l=0;l<12;l++)for(unsigned i=0;i<n*n;i++){uint32_t c=code(format,l,mip);unsigned index=l*n*n+i;u16[index]=(uint16_t)c;u32[index]=(c<<8)|0x80u;f32[index]=value(c);}
      glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,(GLint)mip,0,0,0,n,n,12,GL_DEPTH_COMPONENT,types[format],format==0?(const void*)u16:format==1?(const void*)u32:(const void*)f32);glcheck("depth CubeArray upload");
      for(unsigned l=0;l<12;l++){
         glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,input,(GLint)mip,(GLint)l);glcheck("attach source depth face");check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"depth source FBO complete");
         uint32_t raw[GUARD+16+GUARD];for(unsigned i=0;i<GUARD+16+GUARD;i++)raw[i]=0xa5a5a5a5;
         glReadPixels(0,0,n,n,GL_DEPTH_COMPONENT,format==2?GL_FLOAT:GL_UNSIGNED_INT,raw+GUARD);glcheck("direct source depth readback");
         for(unsigned i=0;i<GUARD;i++)check(raw[i]==0xa5a5a5a5&&raw[GUARD+n*n+i]==0xa5a5a5a5,"depth upload readback guards");
         uint32_t c=code(format,l,mip),expected=format==0?(c<<16)|c:format==1?(c<<8)|(c>>16):c;
         for(unsigned i=0;i<n*n;i++){if(raw[GUARD+i]!=expected)fprintf(stderr,"UPLOAD_MISMATCH fmt%u mip%u layer%u actual%08x expected%08x\n",format,mip,l,raw[GUARD+i],expected);check(raw[GUARD+i]==expected,"exact source depth code");printf("UPLOAD %u %u %u %u %08x\n",format,mip,l,i,raw[GUARD+i]);++uploads;}
      }
   }
   glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,0,0,0);glcheck("detach source depth");
}
int main(void){
   EGLDisplay d=eglGetDisplay(EGL_DEFAULT_DISPLAY);check(d!=EGL_NO_DISPLAY&&eglInitialize(d,NULL,NULL),"EGL init");
   const EGLint ca[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint sa[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE},xa[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};EGLConfig cfg=NULL;EGLint count=0;
   check(eglBindAPI(EGL_OPENGL_ES_API)&&eglChooseConfig(d,ca,&cfg,1,&count)&&count==1,"EGL config");EGLSurface surface=eglCreatePbufferSurface(d,cfg,sa);EGLContext context=eglCreateContext(d,cfg,EGL_NO_CONTEXT,xa);
   check(surface!=EGL_NO_SURFACE&&context!=EGL_NO_CONTEXT&&eglMakeCurrent(d,surface,surface,context),"EGL current");fprintf(stderr,"RENDERER %s\nVERSION %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLint extn;int cube=0;glGetIntegerv(GL_NUM_EXTENSIONS,&extn);for(GLint i=0;i<extn;i++)if(!strcmp((const char*)glGetStringi(GL_EXTENSIONS,i),"GL_EXT_texture_cube_map_array"))cube=1;
   if(!cube){fprintf(stderr,"UNSUPPORTED GL_EXT_texture_cube_map_array\n");return 2;}glcheck("extension query");
   Program p[2]={program(0),program(1)};GLuint out,fbo[2],vao,vbo;glGenFramebuffers(2,fbo);glGenTextures(1,&out);glBindTexture(GL_TEXTURE_2D,out);glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,SIDE,SIDE);
   glBindFramebuffer(GL_FRAMEBUFFER,fbo[0]);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,out,0);const GLenum color=GL_COLOR_ATTACHMENT0;glDrawBuffers(1,&color);glReadBuffer(color);check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"output framebuffer");
   const float verts[]={-1,-1,0,1,3,-1,0,1,-1,3,0,1};glGenVertexArrays(1,&vao);glBindVertexArray(vao);glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);glEnableVertexAttribArray(0);
   glViewport(0,0,SIDE,SIDE);glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);glDisable(GL_CULL_FACE);glDisable(GL_DITHER);glDisable(GL_SCISSOR_TEST);glPixelStorei(GL_PACK_ALIGNMENT,1);glPixelStorei(GL_UNPACK_ALIGNMENT,1);glcheck("draw setup");
   for(unsigned format=0;format<3;format++){
      GLuint input;glGenTextures(1,&input);glActiveTexture(GL_TEXTURE0);source_upload(format,input,fbo[1]);glBindFramebuffer(GL_FRAMEBUFFER,fbo[0]);
      for(unsigned filter=0;filter<2;filter++){
         glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MIN_FILTER,filter?GL_LINEAR_MIPMAP_NEAREST:GL_NEAREST_MIPMAP_NEAREST);glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MAG_FILTER,filter?GL_LINEAR:GL_NEAREST);glcheck("N/N or L/L mip sampler");
         for(unsigned mode=0;mode<2;mode++)for(unsigned mip=0;mip<(mode?3u:1u);mip++)for(unsigned layer=0;layer<12;layer++){
            float c[4];direction(layer%6,c);c[3]=(float)(layer/6);glUseProgram(p[mode].id);glUniform4fv(p[mode].coordinate,1,c);if(mode)glUniform1f(p[mode].lod,(float)mip);
            const GLuint clear[4]={0xdeadbeef,0xdeadbeef,0xdeadbeef,0xdeadbeef};glClearBufferuiv(GL_COLOR,0,clear);glcheck("pre draw");
            double expected=normalized(format,layer,mip);printf("CASE %u format=%u filter=%u mode=%u mip=%u layer=%u expected=%.17g\n",draws,format,filter,mode,mip,layer,expected);fflush(stdout);
            glDrawArrays(GL_TRIANGLES,0,3);glFinish();glcheck("depth CubeArray draw");uint32_t raw[GUARD+PIXELS*4+GUARD];for(unsigned i=0;i<GUARD+PIXELS*4+GUARD;i++)raw[i]=0xa5a5a5a5;
            glReadPixels(0,0,SIDE,SIDE,GL_RGBA_INTEGER,GL_UNSIGNED_INT,raw+GUARD);glcheck("output bits readback");for(unsigned i=0;i<GUARD;i++)check(raw[i]==0xa5a5a5a5&&raw[GUARD+PIXELS*4+i]==0xa5a5a5a5,"output guard words");
            for(unsigned pixel=0;pixel<PIXELS;pixel++)for(unsigned k=0;k<4;k++){uint32_t u=raw[GUARD+pixel*4+k];float f=value(u);printf("WORD %u %u %u %08x\n",draws,pixel,k,u);check(isfinite(f),"finite depth sample");if(k==3)check(u==0x3f800000,"ONE alpha swizzle");else{if(fabs((double)f-expected)>2e-7)fprintf(stderr,"VALUE draw%u actual%.17g expected%.17g\n",draws,(double)f,expected);check(fabs((double)f-expected)<=2e-7,"independent depth oracle abs <= 2e-7");}}
            printf("DONE %u\n",draws++);fflush(stdout);
         }
      }
      glDeleteTextures(1,&input);glcheck("delete input");
   }
   check(draws==288&&uploads==756,"complete format/face/cube/mip inventory");glDeleteProgram(p[0].id);glDeleteProgram(p[1].id);glDeleteBuffers(1,&vbo);glDeleteVertexArrays(1,&vao);glDeleteFramebuffers(2,fbo);glDeleteTextures(1,&out);glcheck("cleanup");
   printf("PASS draws=%u uploads=%u checks=%u\n",draws,uploads,checks);fflush(stdout);check(eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT),"EGL release");check(eglDestroyContext(d,context)&&eglDestroySurface(d,surface)&&eglTerminate(d),"EGL cleanup");return 0;
}
