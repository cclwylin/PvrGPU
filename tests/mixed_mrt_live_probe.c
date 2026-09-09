/* SPDX-License-Identifier: MIT */
/* Tiny actual GLES mixed-MRT test: ordinary VS and real TCS/TES programs.
 * Attachments are RGBA8/RGB10_A2/RGB10_A2/RGBA8, matching portable GLES
 * internal formats. BGR10_A2's physical codec is covered in PBE unit tests;
 * this test never relabels an RGBA allocation as BGR storage.
 *
 * Uploaded raw codes, clear codes, masks, scissor and additive ONE/ONE
 * blending form independent integer-domain oracles. Each draw checks every
 * raw word of every attachment. Packed readback is used only if the queried
 * implementation read format/type explicitly advertises RGBA/2_10_10_10_REV;
 * unsupported readback is a refusal, never an RGBA8 precision fallback.
 * No extension/version override, captured reference pixels or shader swaps.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 4, TARGETS = 4, PIXELS = SIDE * SIDE };
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
static int extension(const char *name)
{
   GLint count = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &count);
   for (GLint i = 0; i < count; ++i)
      if (!strcmp((const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i), name)) return 1;
   return 0;
}
static unsigned packed(unsigned target) { return target == 1 || target == 2; }
static unsigned maximum(unsigned target, unsigned channel)
{ return packed(target) ? (channel == 3 ? 3U : 1023U) : 255U; }
static uint32_t word(unsigned target, const uint32_t codes[4])
{
   for (unsigned c = 0; c < 4; ++c) check(codes[c] <= maximum(target, c), "oracle component bound");
   return packed(target) ? codes[0] | (codes[1] << 10) | (codes[2] << 20) | (codes[3] << 30)
      : codes[0] | (codes[1] << 8) | (codes[2] << 16) | (codes[3] << 24);
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint object = glCreateShader(stage); glShaderSource(object, 1, &source, NULL); glCompileShader(object);
   GLint ok = 0; glGetShaderiv(object, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[8192]; glGetShaderInfoLog(object, sizeof(log), NULL, log);
      fprintf(stderr, "SHADER 0x%x:\n%s\n%s\n", stage, log, source);
   }
   check(ok, "mixed MRT shader compile"); return object;
}
static GLuint program(unsigned tessellation, unsigned sparse)
{
   const char *vs = "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "void main(){gl_Position=position;}\n";
   const char *tc = "#version 310 es\n#extension GL_EXT_tessellation_shader : require\n"
      "precision highp float; precision highp int; layout(vertices=3) out;\n"
      "void main(){gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;\n"
      "if(gl_InvocationID==0){gl_TessLevelInner[0]=1.0;gl_TessLevelOuter[0]=1.0;"
      "gl_TessLevelOuter[1]=1.0;gl_TessLevelOuter[2]=1.0;}}\n";
   const char *te = "#version 310 es\n#extension GL_EXT_tessellation_shader : require\n"
      "precision highp float; layout(triangles,equal_spacing,ccw) in;\n"
      "void main(){gl_Position=gl_TessCoord.x*gl_in[0].gl_Position+"
      "gl_TessCoord.y*gl_in[1].gl_Position+gl_TessCoord.z*gl_in[2].gl_Position;}\n";
   char fs[1024];
   int n = snprintf(fs, sizeof(fs),
      "#version 310 es\nprecision highp float;\n"
      "uniform vec4 source0,source1,source2%s;\n"
      "layout(location=0) out vec4 color0;layout(location=1) out vec4 color1;"
      "layout(location=2) out vec4 color2;%s\n"
      "void main(){color0=source0;color1=source1;color2=source2;%s}\n",
      sparse ? "" : ",source3", sparse ? "" : "layout(location=3) out vec4 color3;",
      sparse ? "" : "color3=source3;");
   check(n > 0 && (size_t)n < sizeof(fs), "FS source bound");
   GLuint object = glCreateProgram();
   const GLenum stages[] = {GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER_EXT, GL_TESS_EVALUATION_SHADER_EXT, GL_FRAGMENT_SHADER};
   const char *sources[] = {vs, tc, te, fs};
   for (unsigned stage = 0; stage < 4; ++stage) {
      if (!tessellation && (stage == 1 || stage == 2)) continue;
      GLuint s = shader(stages[stage], sources[stage]); glAttachShader(object, s); glDeleteShader(s);
   }
   glLinkProgram(object); GLint ok = 0; glGetProgramiv(object, GL_LINK_STATUS, &ok);
   if (!ok) { char log[8192]; glGetProgramInfoLog(object, sizeof(log), NULL, log); fprintf(stderr, "LINK: %s\n", log); }
   check(ok, "mixed MRT program link"); return object;
}
int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint surface_attributes[] = {EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint context_attributes[] = {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attributes,&config,1,&count) && count==1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display,config,surface_attributes);
   EGLContext context = eglCreateContext(display,config,EGL_NO_CONTEXT,context_attributes);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT && eglMakeCurrent(display,surface,surface,context), "EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   if (!extension("GL_EXT_tessellation_shader")) { fprintf(stderr,"UNSUPPORTED: GL_EXT_tessellation_shader\n"); return 2; }
   PFNGLPATCHPARAMETERIEXTPROC patch_parameter = (PFNGLPATCHPARAMETERIEXTPROC)eglGetProcAddress("glPatchParameteriEXT");
   check(patch_parameter != NULL, "patch entrypoint");
   GLuint programs[2][2];
   for (unsigned t=0;t<2;++t) for(unsigned s=0;s<2;++s) programs[t][s]=program(t,s);
   GLuint textures[TARGETS], framebuffer, vao, vbo;
   glGenTextures(TARGETS,textures); glGenFramebuffers(1,&framebuffer); glBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
   const GLenum attachments[] = {GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3};
   for (unsigned target=0;target<TARGETS;++target) {
      glBindTexture(GL_TEXTURE_2D,textures[target]);
      glTexStorage2D(GL_TEXTURE_2D,1,packed(target)?GL_RGB10_A2:GL_RGBA8,SIDE,SIDE);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
      glFramebufferTexture2D(GL_FRAMEBUFFER,attachments[target],GL_TEXTURE_2D,textures[target],0);
   }
   glDrawBuffers(TARGETS,attachments);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"four mixed attachments complete");
   for (unsigned target=0;target<TARGETS;++target) {
      glReadBuffer(attachments[target]); GLint format=0,type=0;
      glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT,&format); glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE,&type);
      fprintf(stderr,"READ_CONTRACT target=%u internal=0x%x format=0x%x type=0x%x\n",target,
         packed(target)?GL_RGB10_A2:GL_RGBA8,format,type);
      if (format!=GL_RGBA || type!=(GLint)(packed(target)?GL_UNSIGNED_INT_2_10_10_10_REV:GL_UNSIGNED_BYTE)) {
         fprintf(stderr,"UNSUPPORTED: exact per-target readback pair\n"); return 2;
      }
   }
   const float positions[]={-1,-1,0,1,3,-1,0,1,-1,3,0,1};
   glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER,vbo);
   glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);
   glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL); glEnableVertexAttribArray(0);
   patch_parameter(GL_PATCH_VERTICES_EXT,3); glViewport(0,0,SIDE,SIDE); glDisable(GL_DITHER); glDisable(GL_CULL_FACE); glDisable(GL_DEPTH_TEST);
   glPixelStorei(GL_PACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ALIGNMENT,1);
   uint32_t expected[TARGETS][PIXELS][4];
   for (unsigned tessellation=0;tessellation<2;++tessellation) {
      glDisable(GL_SCISSOR_TEST); glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE); glDisable(GL_BLEND);
      for(unsigned target=0;target<TARGETS;++target) {
         uint32_t upload[PIXELS];
         for(unsigned pixel=0;pixel<PIXELS;++pixel) {
            expected[target][pixel][0]=1+pixel;
            expected[target][pixel][1]=(packed(target)?257:3)+pixel*3;
            expected[target][pixel][2]=(packed(target)?769:7)+pixel;
            expected[target][pixel][3]=packed(target)?pixel%4:11+pixel;
            upload[pixel]=word(target,expected[target][pixel]);
         }
         glBindTexture(GL_TEXTURE_2D,textures[target]);
         glTexSubImage2D(GL_TEXTURE_2D,0,0,0,SIDE,SIDE,GL_RGBA,
            packed(target)?GL_UNSIGNED_INT_2_10_10_10_REV:GL_UNSIGNED_BYTE,upload);
      }
      for(unsigned scenario=0;scenario<4;++scenario) {
         const unsigned sparse=scenario==2;
         const unsigned mask=scenario==0?5:scenario==3?8:15;
         const unsigned blend=scenario==1;
         if(scenario==3) {
            glDisable(GL_SCISSOR_TEST); glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
            for(unsigned target=0;target<TARGETS;++target) {
               const uint32_t clear_codes[]={3+target,5+target,7+target,packed(target)?1:17+target};
               float clear[4]; for(unsigned c=0;c<4;++c) clear[c]=(float)clear_codes[c]/(float)maximum(target,c);
               glClearBufferfv(GL_COLOR,(GLint)target,clear);
               for(unsigned pixel=0;pixel<PIXELS;++pixel) memcpy(expected[target][pixel],clear_codes,sizeof(clear_codes));
            }
         }
         glUseProgram(programs[tessellation][sparse]);
         uint32_t source[TARGETS][4];
         for(unsigned target=0;target<TARGETS;++target) {
            source[target][0]=blend?1:17+target;
            source[target][1]=blend?2:33+target;
            source[target][2]=blend?3:65+target;
            source[target][3]=blend?0:packed(target)?2:129+target;
            char name[16]; snprintf(name,sizeof(name),"source%u",target);
            GLint location=glGetUniformLocation(programs[tessellation][sparse],name);
            check((location>=0)==!(sparse&&target==3),"exact FS target export contract");
            if(location>=0) { float color[4]; for(unsigned c=0;c<4;++c) color[c]=(float)source[target][c]/(float)maximum(target,c); glUniform4fv(location,1,color); }
         }
         if(blend) { glEnable(GL_BLEND); glBlendFunc(GL_ONE,GL_ONE); glBlendEquation(GL_FUNC_ADD); } else glDisable(GL_BLEND);
         glColorMask(mask&1,mask&2,mask&4,mask&8);
         unsigned x0=0,y0=0,x1=SIDE,y1=SIDE;
         if(scenario==0) x1=2; else if(scenario==1) y1=2; else if(scenario==3) {x0=y0=1;x1=y1=3;}
         glEnable(GL_SCISSOR_TEST); glScissor((GLint)x0,(GLint)y0,(GLsizei)(x1-x0),(GLsizei)(y1-y0));
         check_gl("MRT state before real draw");
         glDrawArrays(tessellation?GL_PATCHES_EXT:GL_TRIANGLES,0,3); glFinish(); check_gl("actual mixed MRT draw");
         for(unsigned target=0;target<TARGETS;++target) {
            if(!(sparse&&target==3)) for(unsigned y=y0;y<y1;++y) for(unsigned x=x0;x<x1;++x)
               for(unsigned c=0;c<4;++c) if(mask&(1U<<c)) {
                  unsigned value=source[target][c]+(blend?expected[target][y*SIDE+x][c]:0);
                  expected[target][y*SIDE+x][c]=value>maximum(target,c)?maximum(target,c):value;
               }
            glReadBuffer(attachments[target]); uint32_t actual[PIXELS];
            glReadPixels(0,0,SIDE,SIDE,GL_RGBA,packed(target)?GL_UNSIGNED_INT_2_10_10_10_REV:GL_UNSIGNED_BYTE,actual);
            check_gl("lossless mixed attachment readback");
            for(unsigned pixel=0;pixel<PIXELS;++pixel) {
               uint32_t wanted=word(target,expected[target][pixel]);
               if(actual[pixel]!=wanted) fprintf(stderr,"DRAW %u target=%u pixel=%u actual=%08x expected=%08x\n",draws,target,pixel,actual[pixel],wanted);
               check(actual[pixel]==wanted,"all raw attachment bits match independent code-domain oracle");
               printf("MRT %u %u %u %08x\n",draws,target,pixel,actual[pixel]);
            }
         }
         printf("DRAW %u tess=%u scenario=%u mask=%u blend=%u sparse=%u PASS\n",draws,tessellation,scenario,mask,blend,sparse); ++draws;
      }
   }
   glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao); glDeleteFramebuffers(1,&framebuffer); glDeleteTextures(TARGETS,textures);
   for(unsigned t=0;t<2;++t) for(unsigned s=0;s<2;++s) glDeleteProgram(programs[t][s]); check_gl("cleanup");
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT); eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
   printf("PASS draws=%u checks=%u\n",draws,checks); return 0;
}
