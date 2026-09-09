/* SPDX-License-Identifier: MIT */
/* Bounded experiment for a future whole-state snapshot renderbuffer codec.
 * This is NOT a RenderDoc checkpoint: prove exact D16 upload -> RB blit ->
 * readback, including every representable word and an odd-sized allocation.
 * No shaders/draws are used by the transport. A separate dependent draw
 * checks that restored storage is consumed by the actual depth test.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static unsigned checks, draws;
static int pack_invert_available, pack_reverse_available;
enum { PACK_INVERT_MESA = 0x8758, PACK_REVERSE_ROW_ORDER_ANGLE = 0x93a4 };
static uint64_t file_bytes(const char *name)
{
   const char *path = getenv(name); struct stat info;
   return path && stat(path, &info) == 0 ? (uint64_t)info.st_size : 0;
}
static void phase(const char *name, unsigned scenario)
{
   printf("PHASE %s scenario=%u driver_offset=%llu model_bytes=%llu\n", name, scenario,
          (unsigned long long)file_bytes("PVRGPU_DRIVER_COUNTER_OUT"),
          (unsigned long long)file_bytes("PVRGPU_SYSTEMC_JSONL_OUT"));
   fflush(stdout);
}
static void check(int condition, const char *message)
{
   ++checks;
   if (!condition) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static void clean(const char *where)
{
   GLenum error = glGetError();
   if (error) fprintf(stderr, "GL_ERROR %s: 0x%x\n", where, error);
   check(error == GL_NO_ERROR, where);
}
static int extension(const char *name)
{
   GLint count = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &count);
   for (GLint i = 0; i < count; ++i)
      if (!strcmp((const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i), name)) return 1;
   return 0;
}
enum { PARAMS = 8 };
static const GLenum pixel_parameters[PARAMS] = {
   GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS,
   GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS
};
struct state {
   GLint read_fbo, draw_fbo, active, texture, rb, pack, unpack, read_buffer, draw_buffer;
   GLint parameters[PARAMS];
   GLint invert, reverse, scissor_box[4];
   GLboolean scissor;
};
static struct state get_state(void)
{
   struct state s = {0};
   glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.read_fbo);
   glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s.draw_fbo);
   glGetIntegerv(GL_ACTIVE_TEXTURE, &s.active);
   glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.texture);
   glGetIntegerv(GL_RENDERBUFFER_BINDING, &s.rb);
   glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &s.pack);
   glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &s.unpack);
   glGetIntegerv(GL_READ_BUFFER, &s.read_buffer);
   glGetIntegerv(GL_DRAW_BUFFER0, &s.draw_buffer);
   glGetIntegerv(GL_SCISSOR_BOX, s.scissor_box);
   if (pack_invert_available) glGetIntegerv(PACK_INVERT_MESA, &s.invert);
   if (pack_reverse_available) glGetIntegerv(PACK_REVERSE_ROW_ORDER_ANGLE, &s.reverse);
   for (unsigned i = 0; i < PARAMS; ++i) glGetIntegerv(pixel_parameters[i], &s.parameters[i]);
   s.scissor = glIsEnabled(GL_SCISSOR_TEST);
   clean("capture touched state");
   return s;
}
static void prepare(void)
{
   glBindBuffer(GL_PIXEL_PACK_BUFFER, 0); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
   for (unsigned i = 0; i < PARAMS; ++i) glPixelStorei(pixel_parameters[i], i % 4 == 0 ? 1 : 0);
   if (pack_invert_available) glPixelStorei(PACK_INVERT_MESA, 0);
   if (pack_reverse_available) glPixelStorei(PACK_REVERSE_ROW_ORDER_ANGLE, 0);
   glDisable(GL_SCISSOR_TEST);
}
static void restore(const struct state *s)
{
   glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)s->read_fbo);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)s->draw_fbo);
   glActiveTexture((GLenum)s->active); glBindTexture(GL_TEXTURE_2D, (GLuint)s->texture);
   glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)s->rb);
   glBindBuffer(GL_PIXEL_PACK_BUFFER, (GLuint)s->pack);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, (GLuint)s->unpack);
   for (unsigned i = 0; i < PARAMS; ++i) glPixelStorei(pixel_parameters[i], s->parameters[i]);
   if (pack_invert_available) glPixelStorei(PACK_INVERT_MESA, s->invert);
   if (pack_reverse_available) glPixelStorei(PACK_REVERSE_ROW_ORDER_ANGLE, s->reverse);
   if (s->scissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
   struct state after = get_state();
   /* Compare semantic fields, never C struct padding. Original FBO read/draw
    * buffer choices and scissor box were not mutated and must remain exact. */
   check(s->read_fbo == after.read_fbo && s->draw_fbo == after.draw_fbo &&
         s->active == after.active && s->texture == after.texture && s->rb == after.rb &&
         s->pack == after.pack && s->unpack == after.unpack &&
         s->read_buffer == after.read_buffer && s->draw_buffer == after.draw_buffer &&
         s->scissor == after.scissor && s->invert == after.invert && s->reverse == after.reverse,
         "all touched state restored");
   for (unsigned i = 0; i < PARAMS; ++i) check(s->parameters[i] == after.parameters[i], "pixel parameter restored");
   for (unsigned i = 0; i < 4; ++i) check(s->scissor_box[i] == after.scissor_box[i], "scissor box unchanged");
}
static GLuint depth_fbo(GLuint rb)
{
   GLuint fbo; glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
   GLenum none = GL_NONE; glDrawBuffers(1, &none); glReadBuffer(GL_NONE);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "depth RB FBO complete");
   return fbo;
}
static void read_rb(GLuint rb, unsigned width, unsigned height, uint16_t *words)
{
   struct state before = get_state(); prepare();
   GLuint fbo = depth_fbo(rb);
   glReadPixels(0, 0, (GLsizei)width, (GLsizei)height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, words);
   clean("D16 raw capture");
   restore(&before); glDeleteFramebuffers(1, &fbo); clean("capture temporary cleanup");
}
static void read_alias(GLuint fbo, unsigned width, unsigned height, uint16_t *words)
{
   struct state before = get_state(); prepare();
   glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
   glReadPixels(0, 0, (GLsizei)width, (GLsizei)height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, words);
   clean("existing alias D16 capture"); restore(&before);
}
static void upload_rb(GLuint rb, unsigned width, unsigned height, const uint16_t *words)
{
   struct state before = get_state(); prepare();
   GLuint target = depth_fbo(rb), source, texture;
   glGenTextures(1, &texture); glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT16, (GLsizei)width, (GLsizei)height);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)width, (GLsizei)height,
                   GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, words);
   glGenFramebuffers(1, &source); glBindFramebuffer(GL_FRAMEBUFFER, source);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0);
   GLenum none = GL_NONE; glDrawBuffers(1, &none); glReadBuffer(GL_NONE);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "D16 staging FBO complete");
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target);
   glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 0, 0, (GLint)width, (GLint)height,
                       GL_DEPTH_BUFFER_BIT, GL_NEAREST);
   glFinish(); clean("exact-format depth-only staging blit");
   restore(&before);
   glDeleteFramebuffers(1, &source); glDeleteFramebuffers(1, &target); glDeleteTextures(1, &texture);
   clean("upload temporary cleanup");
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint s = glCreateShader(stage); glShaderSource(s, 1, &source, NULL); glCompileShader(s);
   GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof(log), NULL, log); fprintf(stderr, "%s\n", log); }
   check(ok, "dependent shader compiled"); return s;
}
int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display, NULL, NULL), "EGL initialize");
   const EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint surface_attrs[] = {EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint context_attrs[] = {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attrs,&config,1,&count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display,config,surface_attrs);
   EGLContext context = eglCreateContext(display,config,EGL_NO_CONTEXT,context_attrs);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT && eglMakeCurrent(display,surface,surface,context), "EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   pack_invert_available = extension("GL_MESA_pack_invert");
   pack_reverse_available = extension("GL_ANGLE_pack_reverse_row_order");
   fprintf(stderr,"OPTIONAL_PACK_STATE mesa_invert=%d angle_reverse=%d\n",pack_invert_available,pack_reverse_available);
   clean("optional pack capabilities");
   GLuint program = 0;
   const char *vs = "#version 310 es\nlayout(location=0) in highp vec4 position;void main(){gl_Position=position;}\n";
   const char *fs = "#version 310 es\nprecision highp float;layout(location=0) out vec4 color;void main(){color=vec4(1,0,0,1);}\n";
   const GLfloat positions[] = {-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
   GLuint vao,vbo; glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(1,&vbo);
   glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferData(GL_ARRAY_BUFFER,sizeof(positions),positions,GL_STATIC_DRAW);
   glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL); glEnableVertexAttribArray(0);
   glDisable(GL_DITHER); glDisable(GL_BLEND); glDisable(GL_CULL_FACE);
   for (unsigned scenario = 0; scenario < 2; ++scenario) {
      phase("TRANSPORT_BEGIN", scenario);
      const unsigned side = scenario ? 257 : 256, pixels = side * side;
      uint16_t *original = malloc(pixels * 2), *saved = malloc(pixels * 2), *readback = malloc(pixels * 2);
      uint8_t *color = malloc(pixels * 4);
      check(original && saved && readback && color, "fixture allocation");
      for (unsigned i=0;i<pixels;++i) original[i] = (uint16_t)(i * 40503U + scenario * 31U);
      GLuint rb; glGenRenderbuffers(1,&rb); glBindRenderbuffer(GL_RENDERBUFFER,rb);
      glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT16,(GLsizei)side,(GLsizei)side);
      GLuint aliases[2] = {depth_fbo(rb), depth_fbo(rb)};
      GLuint sentinel_texture, pbos[2]; glGenTextures(1,&sentinel_texture); glActiveTexture(GL_TEXTURE3);
      glBindTexture(GL_TEXTURE_2D,sentinel_texture); glGenBuffers(2,pbos);
      glBindBuffer(GL_PIXEL_PACK_BUFFER,pbos[0]); glBufferData(GL_PIXEL_PACK_BUFFER,64,NULL,GL_STREAM_READ);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbos[1]); glBufferData(GL_PIXEL_UNPACK_BUFFER,64,NULL,GL_STREAM_DRAW);
      glBindFramebuffer(GL_READ_FRAMEBUFFER,aliases[0]); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,aliases[1]);
      for (unsigned i=0;i<PARAMS;++i) glPixelStorei(pixel_parameters[i], i%4 == 0 ? 8 : (i%4 == 1 ? 19 : 2));
      if (pack_invert_available) glPixelStorei(PACK_INVERT_MESA, 1);
      if (pack_reverse_available) glPixelStorei(PACK_REVERSE_ROW_ORDER_ANGLE, 1);
      glEnable(GL_SCISSOR_TEST); glScissor(1,2,3,4); clean("dirty-state setup");
      upload_rb(rb,side,side,original); read_rb(rb,side,side,saved);
      for (unsigned i=0;i<pixels;++i) check(saved[i] == original[i], "initial D16 capture exact");
      glDisable(GL_SCISSOR_TEST); glClearDepthf(1); glDepthMask(GL_TRUE); glClear(GL_DEPTH_BUFFER_BIT);
      glEnable(GL_SCISSOR_TEST); read_rb(rb,side,side,readback);
      for (unsigned i=0;i<pixels;++i) check(readback[i] == 65535U, "storage actually overwritten");
      /* The restore source is exclusively this RB's own saved raw bytes. */
      upload_rb(rb,side,side,saved); read_rb(rb,side,side,readback);
      for (unsigned i=0;i<pixels;++i) check(readback[i] == original[i], "restored D16 exact");
      for (unsigned alias=0;alias<2;++alias) {
         glBindFramebuffer(GL_FRAMEBUFFER,aliases[alias]); GLint type=0,name=0;
         glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,&type);
         glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,&name);
         check(type == GL_RENDERBUFFER && name == (GLint)rb, "both FBO aliases preserved");
         read_alias(aliases[alias],side,side,readback);
         for (unsigned i=0;i<pixels;++i) check(readback[i] == original[i], "each original alias sees exact restored bytes");
      }
      phase("TRANSPORT_END", scenario);
      puts(scenario ? "TRANSPORT 1 width=257 height=257 PASS" : "TRANSPORT 0 width=256 height=256 PASS"); fflush(stdout);
      phase("DEPENDENT_BEGIN", scenario);
      if (!program) {
         program = glCreateProgram();
         GLuint v = shader(GL_VERTEX_SHADER, vs), f = shader(GL_FRAGMENT_SHADER, fs);
         glAttachShader(program,v); glAttachShader(program,f); glLinkProgram(program);
         GLint linked = 0; glGetProgramiv(program,GL_LINK_STATUS,&linked); check(linked,"dependent program linked");
         glDeleteShader(v); glDeleteShader(f);
      }
      glUseProgram(program);
      prepare(); glBindFramebuffer(GL_FRAMEBUFFER,aliases[0]);
      GLuint color_texture; glGenTextures(1,&color_texture); glBindTexture(GL_TEXTURE_2D,color_texture);
      glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,(GLsizei)side,(GLsizei)side);
      glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,color_texture,0);
      GLenum attachment=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&attachment); glReadBuffer(attachment);
      check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"dependent FBO complete");
      glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT); glViewport(0,0,(GLsizei)side,(GLsizei)side);
      glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glDepthMask(GL_FALSE);
      glDrawArrays(GL_TRIANGLES,0,3); ++draws; glFinish(); clean("dependent depth draw");
      glReadPixels(0,0,(GLsizei)side,(GLsizei)side,GL_RGBA,GL_UNSIGNED_BYTE,color); clean("dependent color readback");
      unsigned passed=0, ties=0;
      for (unsigned i=0;i<pixels;++i) {
         unsigned pass = original[i] > 32768U;
         passed += pass; ties += original[i] == 32768U;
         if (original[i] >= 32767U && original[i] <= 32769U)
            printf("THRESHOLD scenario=%u depth=%u red=%u alpha=%u\n",scenario,original[i],color[4*i],color[4*i+3]);
         check(color[4*i] == (pass ? 255 : 0) && color[4*i+1] == 0 && color[4*i+2] == 0 &&
               color[4*i+3] == (pass ? 255 : 0), "depth-dependent output exact");
      }
      check(ties > 0, "D16 midpoint equality is exercised and fails GL_LESS");
      printf("ORACLE scenario=%u pixels=%u passed=%u failed=%u ties=%u\n",scenario,pixels,passed,pixels-passed,ties);
      glDepthMask(GL_TRUE); glDisable(GL_DEPTH_TEST);
      glDeleteFramebuffers(2,aliases); glDeleteRenderbuffers(1,&rb); glDeleteTextures(1,&sentinel_texture);
      glDeleteTextures(1,&color_texture); glDeleteBuffers(2,pbos);
      free(original); free(saved); free(readback); free(color); clean("scenario cleanup");
      phase("DEPENDENT_END", scenario);
      printf("DEPENDENT %u PASS\n",scenario); fflush(stdout);
   }
   glDeleteProgram(program); glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao); clean("final cleanup");
   printf("PASS draws=%u checks=%u\n",draws,checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
   eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
   return 0;
}
