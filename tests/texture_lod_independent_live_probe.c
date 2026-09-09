/* SPDX-License-Identifier: MIT */
/* Real divergent implicit samples. Equal min/mag + no mip filtering makes
 * the result independent of derivatives, even with nonconstant UV, texels,
 * and four distinct stored mips. No shader rewrite or reference pixels. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIDE = 4, TEX = 8, LEVELS = 4, GUARD = 16 };
static unsigned checks;
static void check(int ok, const char *why) {
   ++checks; if (!ok) { fprintf(stderr,"FAIL: %s\n",why); exit(1); }
}
static void clean(const char *why) {
   GLenum e = glGetError();
   if (e) fprintf(stderr,"GL_ERROR: %s %x\n",why,e);
   check(e == GL_NO_ERROR,why);
}
static GLuint shader(GLenum stage, const char *source) {
   GLuint s = glCreateShader(stage); glShaderSource(s,1,&source,NULL); glCompileShader(s);
   GLint ok = 0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
   if (!ok) { char log[4096]; glGetShaderInfoLog(s,sizeof(log),NULL,log); fprintf(stderr,"%s\n",log); }
   check(ok,"compile shader"); return s;
}
static int repeat(int x) { return (x % TEX + TEX) % TEX; }
static float texel(unsigned scenario, unsigned level, unsigned x, unsigned y, unsigned c) {
   return (float)(13 + scenario * 17 + level * 193 + x * 7 + y * 11 + c * 3);
}
static GLuint expected(unsigned scenario, unsigned linear, unsigned pixel, unsigned c, float cutoff) {
   float u = ((float)(pixel % SIDE) + .5f) / SIDE;
   float v = ((float)(pixel / SIDE) + .5f) / SIDE;
   if (!(u > cutoff)) return 0;
   float x = (u * 1.5f + .0625f) * TEX;
   float y = (v * .5f - .125f) * TEX;
   if (!linear) return (GLuint)(16 * texel(scenario,0,(unsigned)repeat((int)floorf(x)),(unsigned)repeat((int)floorf(y)),c));
   x -= .5f; y -= .5f;
   int x0 = (int)floorf(x), y0 = (int)floorf(y);
   float fx = x - x0, fy = y - y0;
   float a = texel(scenario,0,(unsigned)repeat(x0),(unsigned)repeat(y0),c);
   float b = texel(scenario,0,(unsigned)repeat(x0+1),(unsigned)repeat(y0),c);
   float d = texel(scenario,0,(unsigned)repeat(x0),(unsigned)repeat(y0+1),c);
   float e = texel(scenario,0,(unsigned)repeat(x0+1),(unsigned)repeat(y0+1),c);
   return (GLuint)(16 * ((a*(1-fx)+b*fx)*(1-fy)+(d*(1-fx)+e*fx)*fy));
}
int main(void) {
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display,NULL,NULL),"EGL initialize");
   const EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint sa[] = {EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attrs,&config,1,&count) && count == 1,"EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display,config,sa);
   EGLContext context = eglCreateContext(display,config,EGL_NO_CONTEXT,ca);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT && eglMakeCurrent(display,surface,surface,context),"EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   const char *vs = "#version 310 es\nlayout(location=0) in highp vec4 position;out highp vec2 uv;"
      "void main(){gl_Position=position;uv=position.xy*0.5+0.5;}\n";
   const char *fs = "#version 310 es\nprecision highp float;precision highp int;"
      "uniform highp sampler2D image;uniform highp float cutoff;in highp vec2 uv;"
      "layout(location=0) out highp uvec4 color;void main(){vec4 value=vec4(0);"
      "if(uv.x>cutoff)value=texture(image,uv*vec2(1.5,0.5)+vec2(0.0625,-0.125));"
      "color=uvec4(value*16.0);}\n";
   GLuint p = glCreateProgram(), v = shader(GL_VERTEX_SHADER,vs), f = shader(GL_FRAGMENT_SHADER,fs);
   glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
   GLint ok = 0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
   if (!ok) { char log[4096]; glGetProgramInfoLog(p,sizeof(log),NULL,log); fprintf(stderr,"%s\n",log); }
   check(ok,"link program"); glDeleteShader(v); glDeleteShader(f); glUseProgram(p);
   GLint image = glGetUniformLocation(p,"image"), cutoff = glGetUniformLocation(p,"cutoff");
   check(image >= 0 && cutoff >= 0,"active uniforms"); glUniform1i(image,0);
   GLuint source, output, sampler, fbo, vao, vbo;
   glGenTextures(1,&output); glBindTexture(GL_TEXTURE_2D,output);
   glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,SIDE,SIDE);
   glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,output,0);
   GLenum attachment = GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&attachment); glReadBuffer(attachment);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"output FBO");
   glGenTextures(1,&source); glBindTexture(GL_TEXTURE_2D,source);
   glTexStorage2D(GL_TEXTURE_2D,LEVELS,GL_RGBA32F,TEX,TEX);
   /* Nonzero sampler object overrides deliberately different texture state. */
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
   glGenSamplers(1,&sampler); glBindSampler(0,sampler);
   glSamplerParameteri(sampler,GL_TEXTURE_WRAP_S,GL_REPEAT);
   glSamplerParameteri(sampler,GL_TEXTURE_WRAP_T,GL_REPEAT);
   glSamplerParameterf(sampler,GL_TEXTURE_MIN_LOD,0);
   glSamplerParameterf(sampler,GL_TEXTURE_MAX_LOD,0);
   const float positions[] = {-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
   glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER,vbo);
   glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);
   glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL); glEnableVertexAttribArray(0);
   glViewport(0,0,SIDE,SIDE); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_DITHER);
   glPixelStorei(GL_PACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ALIGNMENT,1);
   const float cutoffs[] = {0,.25f,.75f,1}; unsigned draw = 0;
   for (unsigned linear = 0; linear < 2; ++linear) {
      GLenum filter = linear ? GL_LINEAR : GL_NEAREST;
      glSamplerParameteri(sampler,GL_TEXTURE_MIN_FILTER,(GLint)filter);
      glSamplerParameteri(sampler,GL_TEXTURE_MAG_FILTER,(GLint)filter);
      GLint queried = 0; glGetSamplerParameteriv(sampler,GL_TEXTURE_MIN_FILTER,&queried); check(queried == (GLint)filter,"sampler min filter");
      glGetSamplerParameteriv(sampler,GL_TEXTURE_MAG_FILTER,&queried); check(queried == (GLint)filter,"sampler mag filter");
      GLfloat lod = -1; glGetSamplerParameterfv(sampler,GL_TEXTURE_MAX_LOD,&lod); check(lod == 0,"captured-style max LOD");
      for (unsigned scenario = 0; scenario < 2; ++scenario) {
         for (unsigned level = 0; level < LEVELS; ++level) {
            unsigned extent = TEX >> level; GLfloat pixels[TEX*TEX*4];
            for (unsigned y = 0; y < extent; ++y) for (unsigned x = 0; x < extent; ++x)
               for (unsigned c = 0; c < 4; ++c) pixels[(y*extent+x)*4+c] = texel(scenario,level,x,y,c);
            glTexSubImage2D(GL_TEXTURE_2D,(GLint)level,0,0,(GLsizei)extent,(GLsizei)extent,GL_RGBA,GL_FLOAT,pixels);
         }
         for (unsigned mode = 0; mode < 4; ++mode) {
            glUniform1f(cutoff,cutoffs[mode]); clean("before draw");
            fprintf(stderr,"DRAW_BEGIN %u linear=%u scenario=%u cutoff=%.2f\n",draw,linear,scenario,cutoffs[mode]);
            glDrawArrays(GL_TRIANGLES,0,3); glFinish(); clean("draw finish");
            GLuint pixels[GUARD+SIDE*SIDE*4+GUARD];
            for (unsigned i = 0; i < sizeof(pixels)/sizeof(pixels[0]); ++i) pixels[i] = UINT32_C(0xdeadbeef);
            glReadPixels(0,0,SIDE,SIDE,GL_RGBA_INTEGER,GL_UNSIGNED_INT,pixels+GUARD); clean("read output");
            for (unsigned i = 0; i < GUARD; ++i) check(pixels[i] == UINT32_C(0xdeadbeef) && pixels[GUARD+SIDE*SIDE*4+i] == UINT32_C(0xdeadbeef),"output guards");
            for (unsigned i = 0; i < SIDE*SIDE*4; ++i) {
               GLuint want = expected(scenario,linear,i/4,i%4,cutoffs[mode]);
               if (pixels[GUARD+i] != want) fprintf(stderr,"MISMATCH draw=%u word=%u actual=%u expected=%u\n",draw,i,pixels[GUARD+i],want);
               check(pixels[GUARD+i] == want,"independent bilinear/nearest base-mip oracle");
            }
            printf("DRAW %u linear=%u scenario=%u cutoff=%.2f PASS\n",draw,linear,scenario,cutoffs[mode]);
            printf("WORDS %u",draw); for (unsigned i = 0; i < SIDE*SIDE*4; ++i) printf(" %u",pixels[GUARD+i]);
            printf("\n"); fflush(stdout); ++draw;
         }
      }
   }
   glBindSampler(0,0); glDeleteSamplers(1,&sampler); glDeleteTextures(1,&source); glDeleteTextures(1,&output);
   glDeleteFramebuffers(1,&fbo); glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao); glDeleteProgram(p); clean("cleanup");
   printf("PASS draws=%u checks=%u\n",draw,checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
   eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display); return 0;
}
