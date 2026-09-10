/* SPDX-License-Identifier: MIT */
/* Raw RGB9_E5 transport regression, independent CPU-supplied packed words.
 * CopyImageSubData to same-target R32UI staging is read with RED_INTEGER/UINT;
 * restoring the saved words uses RGB/UNSIGNED_INT_5_9_9_9_REV without floats.
 * Covers every exponent and independent 512-mantissa sweeps, edge combinations,
 * equal-float/different-word representations, slices and odd-size mip chains.
 * This is bounded domain sampling, not exhaustive 2^32 coverage. No shader,
 * draw, dispatch or blit API is called. Build manually with EGL/GLES libraries,
 * -std=c11 -O2 -Wall -Wextra -Werror. --cpu-only skips all GL calls.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef GL_TEXTURE_CUBE_MAP_ARRAY
#define GL_TEXTURE_CUBE_MAP_ARRAY 0x9009
#endif
#define GUARD UINT32_C(0xdeadbeef)
#define DOMAIN_WORDS 65630U
typedef void (GL_APIENTRYP copy_image_fn)(GLuint, GLenum, GLint, GLint, GLint, GLint,
   GLuint, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei);
static copy_image_fn copy_image;
static uint64_t checks, total_words, copy_calls, subresources;

static void check(int condition, const char *message)
{
   ++checks;
   if (!condition) { fprintf(stderr, "FAIL %s\n", message); exit(1); }
}
static void clean(const char *where)
{
   GLenum error = glGetError();
   if (error) fprintf(stderr, "GL_ERROR %s 0x%x\n", where, error);
   check(error == GL_NO_ERROR, where);
}
static uint64_t fnv_word(uint64_t hash, uint32_t word)
{
   for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (word >> (8 * byte)) & 255;
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}
static uint64_t hash_words(const uint32_t *words, size_t count)
{
   uint64_t result = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < count; ++i) result = fnv_word(result, words[i]);
   return result;
}
static uint64_t file_bytes(const char *name)
{
   const char *path = getenv(name); struct stat s;
   return path && stat(path, &s) == 0 ? (uint64_t)s.st_size : 0;
}
static void phase(const char *name)
{
   printf("PHASE %s driver_bytes=%" PRIu64 " model_bytes=%" PRIu64 "\n", name,
          file_bytes("PVRGPU_DRIVER_COUNTER_OUT"), file_bytes("PVRGPU_SYSTEMC_JSONL_OUT"));
   fflush(stdout);
}
static uint32_t packed(unsigned e, unsigned r, unsigned g, unsigned b)
{
   return (e << 27) | (b << 18) | (g << 9) | r;
}
static uint32_t domain_word(unsigned i)
{
   static const unsigned edge[] = {0, 1, 2, 127, 255, 256, 510, 511};
   if (i < 49152) {
      unsigned channel = i / 16384, e = (i / 512) % 32, m = i % 512;
      unsigned rgb[3] = {257, 511, 1}; rgb[channel] = m;
      return packed(e, rgb[0], rgb[1], rgb[2]);
   }
   i -= 49152;
   if (i < 16384) {
      unsigned e = i / 512, rgb = i % 512;
      return packed(e, edge[rgb % 8], edge[(rgb / 8) % 8], edge[rgb / 64]);
   }
   i -= 16384;
   if (i < 62) {
      unsigned e = i / 2 + 1;
      return i % 2 ? packed(e - 1, 2, 4, 6) : packed(e, 1, 2, 3);
   }
   return packed(i - 62, 0, 0, 0);
}
static uint32_t pattern(unsigned target, unsigned mip, unsigned layer, unsigned i)
{
   if (target == 0 && mip == 0 && layer == 0 && i < DOMAIN_WORDS) return domain_word(i);
   uint32_t v = (i + 1) ^ (target * UINT32_C(0x9e3779b9)) ^
                (mip * UINT32_C(0x85ebca6b)) ^ (layer * UINT32_C(0xc2b2ae35));
   v ^= v >> 16; v *= UINT32_C(0x7feb352d); v ^= v >> 15;
   v *= UINT32_C(0x846ca68b); v ^= v >> 16;
   return v;
}
static double component(uint32_t word, unsigned channel)
{
   return ldexp((double)((word >> (9 * channel)) & 511), (int)(word >> 27) - 24);
}
static void cpu_domain(void)
{
   unsigned seen[3][32][512] = {{{0}}};
   for (unsigned i = 0; i < 49152; ++i) {
      unsigned c = i / 16384; uint32_t word = domain_word(i);
      ++seen[c][word >> 27][(word >> (9 * c)) & 511];
   }
   for (unsigned c = 0; c < 3; ++c) for (unsigned e = 0; e < 32; ++e)
      for (unsigned m = 0; m < 512; ++m) check(seen[c][e][m] == 1, "independent complete mantissa sweep");
   for (unsigned pair = 0; pair < 31; ++pair) {
      uint32_t a = domain_word(65536 + pair * 2), b = domain_word(65537 + pair * 2);
      check(a != b, "equal-float pair has different raw words");
      for (unsigned c = 0; c < 3; ++c) check(component(a,c) == component(b,c), "equal-float pair exact");
   }
   for (unsigned e = 0; e < 32; ++e) for (unsigned c = 0; c < 3; ++c)
      check(component(domain_word(65598 + e), c) == 0.0, "all zero mantissas retain arbitrary exponent bits");
   uint64_t h = UINT64_C(14695981039346656037);
   for (unsigned i = 0; i < DOMAIN_WORDS; ++i) h = fnv_word(h, domain_word(i));
   printf("CPU words=%u exponents=32 mantissas=512 channels=3 edge_combinations=16384 "
          "equal_float_pairs=31 zero_representations=32 fnv=%016" PRIx64 "\n", DOMAIN_WORDS, h);
   fflush(stdout);
}
static int extension(const char *wanted)
{
   GLint count = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &count);
   for (GLint i = 0; i < count; ++i) {
      const char *name = (const char *)glGetStringi(GL_EXTENSIONS, (GLuint)i);
      if (name && strcmp(name, wanted) == 0) return 1;
   }
   return 0;
}
struct spec { GLenum target; unsigned width, height, depth, levels; const char *name; };
struct level { unsigned width, height, layers; uint32_t *saved; };
static unsigned reduced(unsigned n, unsigned mip) { unsigned v = n >> mip; return v ? v : 1; }
static void storage(GLenum target, unsigned levels, GLenum format, unsigned w, unsigned h, unsigned d)
{
   if (target == GL_TEXTURE_2D || target == GL_TEXTURE_CUBE_MAP)
      glTexStorage2D(target, (GLsizei)levels, format, (GLsizei)w, (GLsizei)h);
   else glTexStorage3D(target, (GLsizei)levels, format, (GLsizei)w, (GLsizei)h, (GLsizei)d);
   glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   clean("texture immutable storage and completeness");
}
static void upload_slice(GLenum target, unsigned mip, unsigned layer, struct level *level,
                         GLenum format, GLenum type, const uint32_t *words, int invert)
{
   unsigned pitch = level->width + 7;
   size_t size = (size_t)pitch * (level->height + 2);
   uint32_t *data = malloc(size * sizeof(uint32_t)); check(data != NULL, "guarded upload allocation");
   for (size_t i = 0; i < size; ++i) data[i] = GUARD;
   for (unsigned y = 0; y < level->height; ++y) for (unsigned x = 0; x < level->width; ++x) {
      uint32_t value = words ? words[y * level->width + x] : GUARD;
      data[(y + 1) * pitch + x + 3] = invert ? ~value : value;
   }
   uint64_t before = hash_words(data, size);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 4); glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)pitch);
   glPixelStorei(GL_UNPACK_SKIP_PIXELS, 3); glPixelStorei(GL_UNPACK_SKIP_ROWS, 1);
   glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0); glPixelStorei(GL_UNPACK_SKIP_IMAGES, 0);
   if (target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_2D)
      glTexSubImage2D(target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer : target,
                     (GLint)mip, 0, 0, (GLsizei)level->width, (GLsizei)level->height, format, type, data);
   else glTexSubImage3D(target, (GLint)mip, 0, 0, (GLint)layer,
                       (GLsizei)level->width, (GLsizei)level->height, 1, format, type, data);
   clean("guarded raw texture upload");
   check(hash_words(data,size) == before, "upload source and guards unchanged");
   free(data);
}
static void read_slice(const struct spec *spec, unsigned target_index, unsigned mip, unsigned layer,
                       unsigned mode, GLuint staging, struct level *level)
{
   if (spec->target == GL_TEXTURE_CUBE_MAP || spec->target == GL_TEXTURE_2D)
      glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
         spec->target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer : spec->target, staging, 0);
   else glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, staging, 0, (GLint)layer);
   glReadBuffer(GL_COLOR_ATTACHMENT0);
   check(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "R32UI read FBO complete");
   unsigned pitch = level->width + 5;
   size_t size = (size_t)pitch * (level->height + 2);
   uint32_t *data = malloc(size * sizeof(uint32_t)); check(data != NULL, "guarded read allocation");
   for (size_t i = 0; i < size; ++i) data[i] = GUARD;
   glPixelStorei(GL_PACK_ALIGNMENT, 4); glPixelStorei(GL_PACK_ROW_LENGTH, (GLint)pitch);
   glPixelStorei(GL_PACK_SKIP_PIXELS, 2); glPixelStorei(GL_PACK_SKIP_ROWS, 1);
   glReadPixels(0, 0, (GLsizei)level->width, (GLsizei)level->height, GL_RED_INTEGER, GL_UNSIGNED_INT, data);
   clean("R32UI raw readback");
   uint64_t hash = UINT64_C(14695981039346656037);
   size_t slice = (size_t)level->width * level->height;
   for (unsigned y = 0; y < level->height + 2; ++y) for (unsigned x = 0; x < pitch; ++x) {
      uint32_t value = data[y * pitch + x];
      if (y == 0 || y == level->height + 1 || x < 2 || x >= level->width + 2)
         check(value == GUARD, "readback row and tail guards intact");
      else {
         size_t i = (size_t)(y - 1) * level->width + x - 2;
         uint32_t expected = pattern(target_index, mip, layer, (unsigned)i);
         if (mode == 1) expected = ~expected;
         if (mode == 2) expected = level->saved[layer * slice + i];
         if (value != expected) fprintf(stderr,
            "MISMATCH target=%s mip=%u layer=%u mode=%u pixel=%zu expected=%08x actual=%08x\n",
            spec->name, mip, layer, mode, i, expected, value);
         check(value == expected, "packed RGB9_E5 words exact");
         if (mode == 0) level->saved[layer * slice + i] = value;
         hash = fnv_word(hash, value); ++total_words;
      }
   }
   printf("SUBRESOURCE target=%s mode=%u mip=%u layer=%u width=%u height=%u words=%zu fnv=%016" PRIx64 "\n",
          spec->name, mode, mip, layer, level->width, level->height, slice, hash);
   fflush(stdout); ++subresources; free(data);
}
static void exercise(const struct spec *spec, unsigned target_index)
{
   struct level levels[16] = {{0}};
   GLuint source = 0, fbo = 0;
   glGenTextures(1, &source); glBindTexture(spec->target, source);
   storage(spec->target, spec->levels, GL_RGB9_E5, spec->width, spec->height, spec->depth);
   for (unsigned mip = 0; mip < spec->levels; ++mip) {
      struct level *level = &levels[mip];
      level->width = reduced(spec->width,mip); level->height = reduced(spec->height,mip);
      level->layers = spec->target == GL_TEXTURE_3D ? reduced(spec->depth,mip) : spec->depth;
      size_t slice = (size_t)level->width * level->height;
      level->saved = malloc(slice * level->layers * sizeof(uint32_t));
      check(level->saved != NULL, "saved subresource storage");
      for (unsigned layer = 0; layer < level->layers; ++layer)
         for (size_t i = 0; i < slice; ++i)
            level->saved[layer * slice + i] = pattern(target_index,mip,layer,(unsigned)i);
      GLenum query = spec->target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_POSITIVE_X : spec->target;
      GLint format = 0, width = 0, height = 0, depth = 0;
      glGetTexLevelParameteriv(query, (GLint)mip, GL_TEXTURE_INTERNAL_FORMAT, &format);
      glGetTexLevelParameteriv(query, (GLint)mip, GL_TEXTURE_WIDTH, &width);
      glGetTexLevelParameteriv(query, (GLint)mip, GL_TEXTURE_HEIGHT, &height);
      glGetTexLevelParameteriv(query, (GLint)mip, GL_TEXTURE_DEPTH, &depth);
      check(format == GL_RGB9_E5 && width == (GLint)level->width && height == (GLint)level->height &&
            depth == (GLint)((spec->target == GL_TEXTURE_CUBE_MAP || spec->target == GL_TEXTURE_2D) ? 1 : level->layers),
            "actual source format and mip extent identity");
      clean("source level queries");
   }
   glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
   for (unsigned mode = 0; mode < 3; ++mode) for (unsigned mip = 0; mip < spec->levels; ++mip) {
      struct level *level = &levels[mip]; size_t slice = (size_t)level->width * level->height;
      glBindTexture(spec->target, source);
      for (unsigned layer = 0; layer < level->layers; ++layer)
         upload_slice(spec->target,mip,layer,level,GL_RGB,GL_UNSIGNED_INT_5_9_9_9_REV,
                      level->saved + layer * slice, mode == 1);
      GLuint staging = 0; glGenTextures(1, &staging); glBindTexture(spec->target,staging);
      storage(spec->target,1,GL_R32UI,level->width,level->height,level->layers);
      for (unsigned layer = 0; layer < level->layers; ++layer)
         upload_slice(spec->target,0,layer,level,GL_RED_INTEGER,GL_UNSIGNED_INT,NULL,0);
      if (spec->target == GL_TEXTURE_CUBE_MAP) {
         for (unsigned face = 0; face < 6; ++face) {
            copy_image(source,spec->target,(GLint)mip,0,0,(GLint)face,
                       staging,spec->target,0,0,0,(GLint)face,(GLsizei)level->width,(GLsizei)level->height,1);
            clean("same-target raw cube-face copy"); ++copy_calls;
         }
      } else {
         copy_image(source,spec->target,(GLint)mip,0,0,0,staging,spec->target,0,0,0,0,
                    (GLsizei)level->width,(GLsizei)level->height,(GLsizei)level->layers);
         clean("same-target raw layered copy"); ++copy_calls;
      }
      glFinish(); clean("raw copy finish");
      for (unsigned layer = 0; layer < level->layers; ++layer)
         read_slice(spec,target_index,mip,layer,mode,staging,level);
      glDeleteTextures(1,&staging); clean("staging cleanup");
   }
   glDeleteFramebuffers(1,&fbo); glDeleteTextures(1,&source);
   for (unsigned mip = 0; mip < spec->levels; ++mip) free(levels[mip].saved);
   clean("source cleanup");
}
int main(int argc, char **argv)
{
   check(argc == 1 || (argc == 2 && strcmp(argv[1],"--cpu-only") == 0), "usage: probe [--cpu-only]");
   cpu_domain();
   if (argc == 2) { printf("PASS CPU checks=%" PRIu64 "\n",checks); return 0; }
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display,NULL,NULL), "EGL initialize");
   const EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint surface_attrs[] = {EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint context_attrs[] = {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attrs,&config,1,&count) && count == 1, "EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display,config,surface_attrs);
   EGLContext context = eglCreateContext(display,config,EGL_NO_CONTEXT,context_attrs);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
         eglMakeCurrent(display,surface,surface,context), "EGL current");
   fprintf(stderr,"RENDERER %s\nVERSION %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLint major = 0, minor = 0; glGetIntegerv(GL_MAJOR_VERSION,&major); glGetIntegerv(GL_MINOR_VERSION,&minor);
   int core = major > 3 || (major == 3 && minor >= 2);
   check(core || extension("GL_EXT_texture_cube_map_array") || extension("GL_OES_texture_cube_map_array"),
         "advertised cube-array support");
   const char *copy_name = core ? "glCopyImageSubData" :
      extension("GL_EXT_copy_image") ? "glCopyImageSubDataEXT" :
      extension("GL_OES_copy_image") ? "glCopyImageSubDataOES" : NULL;
   check(copy_name != NULL, "advertised raw copy support");
   copy_image = (copy_image_fn)eglGetProcAddress(copy_name); check(copy_image != NULL, "raw copy entry point");
   printf("CAPABILITY major=%d minor=%d copy=%s cube_array=1\n",major,minor,copy_name);
   glBindBuffer(GL_PIXEL_PACK_BUFFER,0); glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
   clean("initial GL state"); phase("RAW_TRANSPORT_BEGIN");
   const struct spec specs[] = {
      {GL_TEXTURE_2D,257,257,1,9,"2d"}, {GL_TEXTURE_CUBE_MAP,17,17,6,5,"cube"},
      {GL_TEXTURE_2D_ARRAY,17,11,3,5,"array"}, {GL_TEXTURE_3D,17,11,3,5,"3d"},
      {GL_TEXTURE_CUBE_MAP_ARRAY,17,17,12,5,"cube_array"}};
   for (unsigned i = 0; i < sizeof(specs)/sizeof(specs[0]); ++i) exercise(&specs[i],i);
   glFinish(); clean("raw transport completion"); phase("RAW_TRANSPORT_END");
   check(eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT), "EGL release");
   check(eglDestroyContext(display,context) && eglDestroySurface(display,surface) && eglTerminate(display), "EGL cleanup");
   printf("PASS GL targets=5 subresources=%" PRIu64 " words=%" PRIu64 " copies=%" PRIu64
          " mismatches=0 explicit_shader_api_calls=0 draws=0 dispatches=0 explicit_blit_api_calls=0 checks=%" PRIu64 "\n",
          subresources,total_words,copy_calls,checks);
   return 0;
}
