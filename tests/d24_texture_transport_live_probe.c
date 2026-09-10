/* SPDX-License-Identifier: MIT */
/* Exhaustive standalone D24 raw transport regression. No shaders, draws, blits
 * or model work. Every 24-bit code is supplied by the CPU, not a capture/LP image.
 * GL readback must be canonical replicated U32: (z << 8) | (z >> 16).
 * Restore scratch is ceil(z * UINT32_MAX / 0xffffff), preserving archive bytes.
 * This checks the pinned Mesa UINT -> binary32 -> D24 path under FE_TONEAREST;
 * it does not assert that this implementation-specific encoding is universal GL.
 * Build manually with EGL/GLES headers/libs, -std=c11 -O2 -fno-fast-math
 * -ffp-contract=off -Wall -Wextra -Werror. --cpu-only skips all GL calls.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <fenv.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define D24_MAX UINT32_C(0xffffff)
#define DOMAIN (UINT32_C(1) << 24)
#define GUARD UINT32_C(0xdeadbeef)
static uint64_t checks;

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
static uint32_t canonical(uint32_t z) { return (z << 8) | (z >> 16); }
static uint32_t scratch(uint32_t raw)
{
   uint32_t z = raw >> 8;
   check(raw == canonical(z), "restore input is canonical D24 replicated U32");
   uint64_t n = (uint64_t)z * UINT32_MAX;
   return (uint32_t)((n + D24_MAX - 1) / D24_MAX);
}
static uint32_t mesa_uint_upload(uint32_t input)
{
   /* Pinned Mesa main/macros.h UINT_TO_FLOAT and main/pack.c depthMax24.
    * Volatile forces both binary32 stores, preventing excess precision/fusion.
    */
   volatile float normalized = (float)(input * (1.0F / 4294967295.0));
   volatile float scaled = normalized * (float)D24_MAX;
   return (uint32_t)scaled;
}
static uint64_t fnv_word(uint64_t hash, uint32_t word)
{
   for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (word >> (8 * byte)) & 255;
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}
static uint64_t bytes(const char *name)
{
   const char *path = getenv(name); struct stat s;
   return path && stat(path, &s) == 0 ? (uint64_t)s.st_size : 0;
}
static void phase(const char *name)
{
   printf("PHASE %s driver_bytes=%" PRIu64 " model_bytes=%" PRIu64 "\n", name,
          bytes("PVRGPU_DRIVER_COUNTER_OUT"), bytes("PVRGPU_SYSTEMC_JSONL_OUT"));
   fflush(stdout);
}
static void cpu_domain(void)
{
   check(fegetround() == FE_TONEAREST, "rounding mode FE_TONEAREST");
   uint32_t losses = 0, first = DOMAIN, previous = 0;
   for (uint32_t z = 0; z < DOMAIN; ++z) {
      uint32_t raw = canonical(z), upload = scratch(raw);
      uint64_t n = (uint64_t)z * UINT32_MAX;
      check((uint64_t)upload * D24_MAX >= n, "integer ceiling lower bound");
      check(upload == 0 || (uint64_t)(upload - 1) * D24_MAX < n, "integer ceiling minimality");
      check(z == 0 || upload >= previous, "scratch encoding monotonic");
      check((raw >> 8) == z && mesa_uint_upload(upload) == z, "CPU all-code candidate exact");
      previous = upload;
      if (mesa_uint_upload(raw) != z) { if (first == DOMAIN) first = z; ++losses; }
   }
   check(previous == UINT32_MAX && losses == 65535 && first == 1, "CPU old negative domain");
   check(mesa_uint_upload(canonical(159)) == 158, "actual capture code159 old negative");
   printf("CPU codes=%u old_mismatches=%u first_old=%u candidate_mismatches=0 rounding=nearest "
          "z159_raw=%08x old_z159=%u corrected159=%08x\n",
          DOMAIN, losses, first, canonical(159), mesa_uint_upload(canonical(159)), scratch(canonical(159)));
   fflush(stdout);
}
struct image {
   GLuint texture, fbo;
   unsigned width, height, pack_pitch, unpack_pitch;
   size_t pack_size, unpack_size;
   uint32_t *pack, *unpack, *saved;
};
static struct image create_image(unsigned width, unsigned height)
{
   struct image image = {0};
   image.width = width; image.height = height;
   image.pack_pitch = width + 5; image.unpack_pitch = width + 7;
   image.pack_size = (size_t)image.pack_pitch * (height + 2);
   image.unpack_size = (size_t)image.unpack_pitch * (height + 2);
   image.pack = malloc(image.pack_size * sizeof(uint32_t));
   image.unpack = malloc(image.unpack_size * sizeof(uint32_t));
   image.saved = malloc((size_t)width * height * sizeof(uint32_t));
   check(image.pack && image.unpack && image.saved, "host guarded allocations");
   glGenTextures(1, &image.texture); glBindTexture(GL_TEXTURE_2D, image.texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, (GLsizei)width, (GLsizei)height);
   GLint format = 0, actual_width = 0, actual_height = 0;
   glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
   glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &actual_width);
   glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &actual_height);
   check(format == GL_DEPTH_COMPONENT24 && actual_width == (GLint)width &&
         actual_height == (GLint)height, "actual D24 texture storage identity");
   glGenFramebuffers(1, &image.fbo); glBindFramebuffer(GL_FRAMEBUFFER, image.fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, image.texture, 0);
   GLenum none = GL_NONE; glDrawBuffers(1, &none); glReadBuffer(GL_NONE);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "D24 FBO complete");
   glBindBuffer(GL_PIXEL_PACK_BUFFER, 0); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
   glPixelStorei(GL_PACK_ALIGNMENT, 8); glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
   /* Width1024+odd pitch would be rounded for alignment8. Set alignment4 so
    * declared row lengths, including odd7-wide scenario, are exact word pitches.
    */
   glPixelStorei(GL_PACK_ALIGNMENT, 4); glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
   glPixelStorei(GL_PACK_ROW_LENGTH, (GLint)image.pack_pitch);
   glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)image.unpack_pitch);
   glPixelStorei(GL_PACK_SKIP_PIXELS, 2); glPixelStorei(GL_UNPACK_SKIP_PIXELS, 3);
   glPixelStorei(GL_PACK_SKIP_ROWS, 1); glPixelStorei(GL_UNPACK_SKIP_ROWS, 1);
   clean("D24 texture and guarded pixel storage setup");
   return image;
}
static void destroy_image(struct image *image)
{
   glDeleteFramebuffers(1, &image->fbo); glDeleteTextures(1, &image->texture);
   free(image->pack); free(image->unpack); free(image->saved);
   clean("D24 texture cleanup");
}
static uint32_t code_at(unsigned scenario, uint32_t base, unsigned i)
{
   static const uint32_t odd[] = {0, 1, 2, 158, 159, 160, 255, 256, 257, 65535, 65536,
      65537, 0x7ffffe, 0x7fffff, 0x800000, 0x800001, 0xfffffd, 0xfffffe, 0xffffff};
   return scenario ? odd[i % (sizeof(odd) / sizeof(odd[0]))] : base + i;
}
static void upload(struct image *image, unsigned mode, unsigned scenario, uint32_t base)
{
   for (size_t i = 0; i < image->unpack_size; ++i) image->unpack[i] = GUARD;
   for (unsigned y = 0; y < image->height; ++y) for (unsigned x = 0; x < image->width; ++x) {
      unsigned i = y * image->width + x;
      uint32_t raw = mode == 0 ? canonical(code_at(scenario, base, i)) : image->saved[i];
      image->unpack[(y + 1) * image->unpack_pitch + x + 3] = mode == 1 ? raw : scratch(raw);
   }
   uint64_t before = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < image->unpack_size; ++i) before = fnv_word(before, image->unpack[i]);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)image->width, (GLsizei)image->height,
                   GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, image->unpack);
   glFinish(); clean("raw D24 upload and finish");
   uint64_t after = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < image->unpack_size; ++i) after = fnv_word(after, image->unpack[i]);
   check(before == after, "upload source and row guards unchanged");
}
static void read_image(struct image *image)
{
   for (size_t i = 0; i < image->pack_size; ++i) image->pack[i] = GUARD;
   glReadPixels(0, 0, (GLsizei)image->width, (GLsizei)image->height,
                GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, image->pack);
   clean("raw D24 readback");
   for (unsigned y = 0; y < image->height + 2; ++y)
      for (unsigned x = 0; x < image->pack_pitch; ++x)
         if (y == 0 || y == image->height + 1 || x < 2 || x >= image->width + 2)
            check(image->pack[y * image->pack_pitch + x] == GUARD, "readback row/tail guards exact");
}
static uint32_t exercise(struct image *image, unsigned scenario, unsigned tile, uint32_t base)
{
   uint32_t losses = 0;
   uint64_t hashes[3];
   for (unsigned mode = 0; mode < 3; ++mode) {
      hashes[mode] = UINT64_C(14695981039346656037);
      upload(image, mode, scenario, base); read_image(image);
      for (unsigned y = 0; y < image->height; ++y) for (unsigned x = 0; x < image->width; ++x) {
         unsigned i = y * image->width + x;
         uint32_t z = code_at(scenario, base, i), value = image->pack[(y + 1) * image->pack_pitch + x + 2];
         uint32_t expected = mode == 1 ? canonical(mesa_uint_upload(image->saved[i])) : canonical(z);
         if (value != expected)
            fprintf(stderr, "MISMATCH scenario=%u tile=%u mode=%u pixel=%u z=%u expected=%08x actual=%08x\n",
                    scenario, tile, mode, i, z, expected, value);
         check(value == expected, "raw D24 actual words match independent CPU oracle");
         check(value == canonical(value >> 8), "actual readback encoding canonical");
         if (mode == 0) image->saved[i] = value;
         if (mode == 1 && value != image->saved[i]) ++losses;
         if (mode == 2) check(value == image->saved[i], "saved-own-bytes corrected restore exact");
         hashes[mode] = fnv_word(hashes[mode], value);
         if (scenario == 0 && z == 159)
            printf("WORD z=159 mode=%u saved=%08x actual=%08x\n", mode, image->saved[i], value);
      }
   }
   check(hashes[0] == hashes[2], "complete tile restore digest exact");
   printf("TILE scenario=%u tile=%u base=%u width=%u height=%u words=%u old_mismatches=%u "
          "seed_fnv=%016" PRIx64 " old_fnv=%016" PRIx64 " corrected_fnv=%016" PRIx64 "\n",
          scenario, tile, base, image->width, image->height, image->width * image->height,
          losses, hashes[0], hashes[1], hashes[2]);
   fflush(stdout);
   return losses;
}
int main(int argc, char **argv)
{
   check(argc == 1 || (argc == 2 && strcmp(argv[1], "--cpu-only") == 0), "usage: probe [--cpu-only]");
   cpu_domain();
   if (argc == 2) { printf("PASS CPU checks=%" PRIu64 "\n", checks); return 0; }
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
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT &&
         eglMakeCurrent(display,surface,surface,context), "EGL current");
   fprintf(stderr, "RENDERER %s\nVERSION %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   clean("initial GL state"); phase("RAW_TRANSPORT_BEGIN");
   struct image image = create_image(1024, 1024);
   uint32_t losses = 0;
   for (unsigned tile = 0; tile < 16; ++tile) losses += exercise(&image, 0, tile, tile * 1024U * 1024U);
   check(losses == 65535, "actual full-domain old path loses exactly65535 codes");
   destroy_image(&image);
   image = create_image(7, 5); uint32_t odd_losses = exercise(&image, 1, 0, 0); destroy_image(&image);
   glFinish(); clean("transport completion"); phase("RAW_TRANSPORT_END");
   check(eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT), "EGL release");
   check(eglDestroyContext(display,context) && eglDestroySurface(display,surface) && eglTerminate(display), "EGL cleanup");
   printf("PASS GL domain=%u supplementary_words=35 old_mismatches=%u odd_old_mismatches=%u "
          "corrected_mismatches=0 shaders=0 draws=0 blits=0 checks=%" PRIu64 "\n",
          DOMAIN, losses, odd_losses, checks);
   return 0;
}
