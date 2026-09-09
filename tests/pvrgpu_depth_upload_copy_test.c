/* SPDX-License-Identifier: MIT */
/* Isolated actual Gallium Z24X8 upload/copy + Mesa UINT depth conversion.
 * Run with script/run_mesa_resource_unit.sh depth-upload.
 * The bridge is stubbed and must never be needed for an idle upload/copy. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_resource.c"
#include "main/mtypes.h"
#include "main/pack.h"

static unsigned checks, failures, copies, unsupported;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); } } while (0)

void pvrgpu_counter_eventf(const char *event, const char *format, ...)
{
   (void)format;
   copies += strcmp(event, "resource_copy_region") == 0;
   unsupported += strcmp(event, "unsupported_resource_copy_region") == 0;
}
bool pvrgpu_context_has_incomplete_replay(const struct pvrgpu_context *ctx)
{ (void)ctx; return false; }
bool pvrgpu_context_has_recorded_geometry(const struct pvrgpu_context *ctx)
{ (void)ctx; return false; }
bool pvrgpu_case_reserves_native_pco_sequence(void) { return false; }
bool pvrgpu_driver_draw_command_has_been_emitted(void) { return false; }
void pvrgpu_note_driver_draw_command_emitted(void) { CHECK(false); }
bool pvrgpu_write_draw_indexed_quad_command(
   const char *path, const struct pvrgpu_draw_indexed_quad_command *cmd,
   char *error, size_t error_size)
{
   (void)path; (void)cmd; (void)error; (void)error_size;
   CHECK(false); return false;
}
void pvrgpu_context_end_frame_at_readback(struct pvrgpu_context *ctx)
{ CHECK(ctx->array_primitive_draw_count == 0); }
void pvrgpu_invalidate_full_depth_clear_for_resource(
   struct pvrgpu_context *ctx, const struct pipe_resource *resource)
{ (void)ctx; (void)resource; }
uint64_t pvrgpu_systemc_submission_generation(void) { return 0; }
bool pvrgpu_is_supported_color_format(enum pipe_format format)
{ (void)format; return false; }
bool pvrgpu_systemc_flush_readback_pixels(
   uint32_t width, uint32_t height, uint32_t bpp, uint32_t attachment,
   uint32_t samples, uint32_t depth_format, uint32_t layers,
   const char *color_format,
   uint8_t *pixels, size_t bytes, bool *written, char *error, size_t error_size)
{
   (void)width; (void)height; (void)bpp; (void)attachment; (void)samples;
   (void)depth_format; (void)layers; (void)pixels; (void)bytes; (void)written;
   (void)error; (void)error_size; (void)color_format;
   CHECK(false); return false;
}

static struct pvrgpu_resource
resource(unsigned width, unsigned height)
{
   struct pvrgpu_resource r = {0};
   r.base.target = PIPE_TEXTURE_2D;
   r.base.format = PIPE_FORMAT_Z24X8_UNORM;
   r.base.width0 = width; r.base.height0 = height;
   r.base.depth0 = r.base.array_size = 1;
   r.base.last_level = 1;
   r.base.bind = PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW;
   pipe_reference_init(&r.base.reference, 1);
   CHECK(pvrgpu_init_resource_storage(&r));
   memset(r.data, 0xa5, r.size);
   return r;
}

static void
test_upload(unsigned mode, unsigned level)
{
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_resource src = resource(16, 12), dst = resource(20, 16);
   struct gl_context *gl = calloc(1, sizeof(*gl));
   CHECK(gl != NULL);
   gl->Pixel.DepthScale = 1.0f;
   struct gl_pixelstore_attrib unpack = {0};
   const GLuint input[5] = {0, 0x40000000, 0x80000000, 0xffffff00, UINT32_MAX};
   GLuint converted[5] = {0};
   _mesa_unpack_depth_span(gl, 5, GL_UNSIGNED_INT, converted, 0xffffff,
                            GL_UNSIGNED_INT, input, &unpack);
   CHECK(converted[0] == 0 && converted[4] == 0xffffff);
   CHECK(converted[1] > 0 && converted[1] < converted[2]);
   CHECK(converted[2] < converted[3] && converted[3] <= converted[4]);
   GLuint rows[3][7];
   for (unsigned y = 0; y < 3; ++y)
   for (unsigned x = 0; x < 7; ++x)
      rows[y][x] = converted[(x + y) % 5] | (mode == 0 ? 0x7b000000 : 0);
   const struct pipe_box box = {.x = 1, .y = 1, .width = 5, .height = 3, .depth = 1};
   if (mode < 2) {
      struct pipe_transfer *transfer = NULL;
      uint8_t *mapped = pvrgpu_transfer_map(&ctx.base, &src.base, level,
         PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE, &box, &transfer);
      CHECK(mapped != NULL && transfer != NULL);
      CHECK(transfer->stride == u_minify(src.base.width0, level) * 4);
      CHECK(mapped == src.data + src.level_offsets[level] + transfer->stride + 4);
      for (unsigned y = 0; y < 3; ++y)
         memcpy(mapped + y * transfer->stride, rows[y], 5 * sizeof(GLuint));
      pvrgpu_transfer_unmap(&ctx.base, transfer);
      CHECK(src.base.reference.count == 1);
   } else {
      pvrgpu_texture_subdata(&ctx.base, &src.base, level, PIPE_MAP_WRITE,
                              &box, rows, sizeof(rows[0]), sizeof(rows));
   }
   const unsigned before = copies;
   pvrgpu_resource_copy_region(&ctx.base, &dst.base, level, 2, 2, 0,
                                &src.base, level, &box);
   CHECK(copies == before + 1 && unsupported == 0);
   CHECK(dst.driver_writes_model_cannot_reproduce);
   for (unsigned y = 0; y < u_minify(dst.base.height0, level); ++y)
   for (unsigned x = 0; x < u_minify(dst.base.width0, level); ++x) {
      GLuint actual;
      memcpy(&actual, dst.data + dst.level_offsets[level] +
                       y * dst.level_strides[level] + x * 4, sizeof(actual));
      const bool inside = x >= 2 && x < 7 && y >= 2 && y < 5;
      CHECK(actual == (inside ? rows[y - 2][x - 2] : 0xa5a5a5a5));
   }
   /* Readback mapping and its unmap are also a direct, lossless view. */
   struct pipe_box read_box = {.x = 2, .y = 2, .width = 5, .height = 3, .depth = 1};
   struct pipe_transfer *read = NULL;
   uint8_t *mapped = pvrgpu_transfer_map(&ctx.base, &dst.base, level,
                                        PIPE_MAP_READ, &read_box, &read);
   CHECK(mapped != NULL && read != NULL);
   for (unsigned y = 0; y < 3; ++y)
      CHECK(memcmp(mapped + y * read->stride, rows[y], 5 * sizeof(GLuint)) == 0);
   pvrgpu_transfer_unmap(&ctx.base, read);
   CHECK(dst.base.reference.count == 1);
   CHECK(ctx.query_statistics_failures == 0);
   free(gl); FREE(src.data); FREE(dst.data);
}

int main(void)
{
   for (unsigned mode = 0; mode < 3; ++mode)
   for (unsigned level = 0; level < 2; ++level)
      test_upload(mode, level);
   printf("Z24X8 UINT upload/copy: %s (%u checks, %u failures)\n",
          failures ? "FAIL" : "PASS", checks, failures);
   return failures ? 1 : 0;
}
