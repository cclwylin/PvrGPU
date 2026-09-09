/* SPDX-License-Identifier: MIT */
/* Independent integer oracle for raster-facing transport through derivative
 * quads, uncovered helper lanes and a conditional native texture instruction. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIDE = 8 };
static unsigned checks, draws;
static void check(int good, const char *message)
{
   ++checks;
   if (!good) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
static void clean(const char *message)
{
   GLenum error = glGetError();
   if (error) fprintf(stderr,"GL_ERROR %s 0x%x\n",message,error);
   check(error == GL_NO_ERROR,message);
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint s = glCreateShader(stage); glShaderSource(s,1,&source,NULL); glCompileShader(s);
   GLint ok = 0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
   if (!ok) { char log[8192]; glGetShaderInfoLog(s,sizeof(log),NULL,log); fprintf(stderr,"%s\n%s\n",source,log); }
   check(ok,"compile helper-facing shader"); return s;
}
static GLuint program(unsigned textured)
{
   const char *vs = "#version 310 es\nprecision highp float;\n"
      "layout(location=0) in vec2 position;uniform float reflect_x;"
      "void main(){gl_Position=vec4(position.x*reflect_x,position.y,0,1);}\n";
   char fs[1536];
   int n = snprintf(fs,sizeof(fs),
      "#version 310 es\nprecision highp float; precision highp int;\n"
      "uniform highp sampler2D source_texture;layout(location=0) out highp uvec4 color;\n"
      "void main(){vec2 delta=vec2(dFdx(gl_FragCoord.x),dFdy(gl_FragCoord.y));"
      "if(gl_FrontFacing){color=uvec4(11,22,33,44);}else{color=%s;}"
      "color.yz=floatBitsToUint(delta);}\n",
      textured ? "uvec4(textureLod(source_texture,vec2(.5),0.0))" : "uvec4(55,66,77,88)");
   check(n > 0 && (size_t)n < sizeof(fs),"source bound");
   GLuint p = glCreateProgram(), v = shader(GL_VERTEX_SHADER,vs), f = shader(GL_FRAGMENT_SHADER,fs);
   glAttachShader(p,v); glAttachShader(p,f); glDeleteShader(v); glDeleteShader(f); glLinkProgram(p);
   GLint ok = 0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
   if (!ok) { char log[8192]; glGetProgramInfoLog(p,sizeof(log),NULL,log); fprintf(stderr,"%s\n",log); }
   check(ok,"link helper-facing shader"); return p;
}
int main(void)
{
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
   GLuint programs[] = {program(0),program(1)};
   GLuint textures[2], fbo, vao, buffers[2];
   glGenTextures(2,textures); glBindTexture(GL_TEXTURE_2D,textures[0]); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,SIDE,SIDE);
   glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,textures[0],0);
   GLenum attachment = GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&attachment); glReadBuffer(attachment);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"integer output FBO");
   glBindTexture(GL_TEXTURE_2D,textures[1]); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32F,1,1);
   const float texel[] = {55,66,77,88}; glTexSubImage2D(GL_TEXTURE_2D,0,0,0,1,1,GL_RGBA,GL_FLOAT,texel);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
   /* Window-space vertices (1.25,1.25), (6.5,1.25), (1.25,6.5).
    * No pixel center lies on an edge: coverage does not depend on tie rules. */
   const float positions[] = {-.6875f,-.6875f,.625f,-.6875f,-.6875f,.625f};
   const uint16_t indices[] = {0,1,2,0,2,1};
   glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(2,buffers);
   glBindBuffer(GL_ARRAY_BUFFER,buffers[0]); glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);
   glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,NULL); glEnableVertexAttribArray(0);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,buffers[1]); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(indices),indices,GL_STATIC_DRAW);
   glViewport(0,0,SIDE,SIDE); glDisable(GL_DITHER); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST);
   clean("setup");
   for (unsigned textured = 0; textured < 2; ++textured) {
      glUseProgram(programs[textured]);
      GLint reflect = glGetUniformLocation(programs[textured],"reflect_x"); check(reflect >= 0,"reflection uniform");
      if (textured) { GLint location = glGetUniformLocation(programs[textured],"source_texture"); check(location >= 0,"active texture sampler"); glUniform1i(location,0); }
      for (unsigned reflection = 0; reflection < 2; ++reflection) for (unsigned winding = 0; winding < 2; ++winding)
         for (unsigned front_cw = 0; front_cw < 2; ++front_cw) for (unsigned cull = 0; cull < 3; ++cull) {
            glUniform1f(reflect,reflection ? -1.f : 1.f); glFrontFace(front_cw ? GL_CW : GL_CCW);
            if (cull) { glEnable(GL_CULL_FACE); glCullFace(cull == 1 ? GL_FRONT : GL_BACK); } else glDisable(GL_CULL_FACE);
            unsigned front = (winding ^ reflection) == front_cw;
            unsigned culled = cull && (cull == 1 ? front : !front);
            const GLuint zero[] = {0,0,0,0}; glClearBufferuiv(GL_COLOR,0,zero);
            glDrawElements(GL_TRIANGLES,3,GL_UNSIGNED_SHORT,(void *)(uintptr_t)(winding * 3 * sizeof(uint16_t)));
            glFinish(); clean("helper-facing draw");
            uint32_t pixels[SIDE * SIDE * 4]; glReadPixels(0,0,SIDE,SIDE,GL_RGBA_INTEGER,GL_UNSIGNED_INT,pixels); clean("integer readback");
            for (unsigned y = 0; y < SIDE; ++y) for (unsigned x = 0; x < SIDE; ++x) {
               float u = reflection ? SIDE - (x + .5f) : x + .5f, v = y + .5f;
               unsigned covered = !culled && u > 1.25f && v > 1.25f && u + v < 7.75f;
               for (unsigned c = 0; c < 4; ++c) {
                  unsigned expected = covered ? ((c == 1 || c == 2) ? UINT32_C(0x3f800000) : (front ? 11 : 55) + 11 * c) : 0;
                  unsigned i = (y * SIDE + x) * 4 + c;
                  if (pixels[i] != expected) fprintf(stderr,"DRAW%u xy(%u,%u) component%u actual%u expected%u\n",draws,x,y,c,pixels[i],expected);
                  check(pixels[i] == expected,"coverage, facing and unit derivatives match independent oracle");
               }
            }
            printf("DRAW %u texture=%u reflection=%u winding_cw=%u front_cw=%u cull=%u front=%u culled=%u PASS\n",
               draws,textured,reflection,winding,front_cw,cull,front,culled); fflush(stdout); ++draws;
         }
   }
   glDeleteBuffers(2,buffers); glDeleteVertexArrays(1,&vao); glDeleteFramebuffers(1,&fbo); glDeleteTextures(2,textures);
   glDeleteProgram(programs[0]); glDeleteProgram(programs[1]); clean("cleanup");
   printf("PASS draws=%u checks=%u\n",draws,checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT); eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
   return 0;
}
