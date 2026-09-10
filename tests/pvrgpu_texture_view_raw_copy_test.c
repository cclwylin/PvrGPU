/* SPDX-License-Identifier: MIT */
/* Capture-independent raw glCopyImageSubData storage test.  The formats in
 * GL's VIEW_CLASS_32_BITS may be copied without conversion; in particular a
 * sampler-only RGB9_E5 cubemap can be staged through an integer renderable
 * texture and read back without a helper shader. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_resource.c"

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
   uint8_t *pixels, size_t bytes, bool *written, char *error,
   size_t error_size)
{
   (void)width; (void)height; (void)bpp; (void)attachment; (void)samples;
   (void)depth_format; (void)layers; (void)color_format; (void)pixels;
   (void)bytes; (void)written; (void)error; (void)error_size;
   CHECK(false); return false;
}

static struct pvrgpu_resource
cube(enum pipe_format format)
{
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_TEXTURE_CUBE;
   resource.base.format = format;
   resource.base.width0 = 17;
   resource.base.height0 = 11;
   resource.base.depth0 = 1;
   resource.base.array_size = 6;
   resource.base.last_level = 4;
   resource.base.bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET;
   pipe_reference_init(&resource.base.reference, 1);
   CHECK(pvrgpu_init_resource_storage(&resource));
   return resource;
}

static uint32_t
pattern(unsigned direction, unsigned level, unsigned face,
        unsigned x, unsigned y)
{
   uint32_t value = UINT32_C(0x9e3779b9) *
      (1u + direction * 131u + level * 29u + face * 7u + y * 19u + x);
   value ^= value >> 16;
   value *= UINT32_C(0x85ebca6b);
   return value ^ (value >> 13);
}

static void
fill_cube(struct pvrgpu_resource *resource, unsigned direction)
{
   for (unsigned level = 0; level < resource->level_count; ++level) {
      const unsigned width = pvrgpu_resource_level_width(&resource->base,
                                                         level);
      const unsigned height = pvrgpu_resource_level_height(&resource->base,
                                                            level);
      for (unsigned face = 0; face < 6; ++face)
      for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x) {
         const uint32_t value = pattern(direction, level, face, x, y);
         uint8_t *pixel = resource->data + resource->level_offsets[level] +
            (size_t)face * resource->level_layer_strides[level] +
            (size_t)y * resource->level_strides[level] + x * sizeof(value);
         memcpy(pixel, &value, sizeof(value));
      }
   }
}

static void
test_cube_faces_and_mips(enum pipe_format src_format,
                         enum pipe_format dst_format,
                         unsigned direction)
{
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_resource src = cube(src_format);
   struct pvrgpu_resource dst = cube(dst_format);
   fill_cube(&src, direction);
   memset(dst.data, 0xa5, dst.size);
   CHECK(src.size == dst.size);
   CHECK(pvrgpu_texture_copy_formats_are_raw_compatible(dst_format,
                                                        src_format));
   const unsigned before = copies;
   for (unsigned level = 0; level < src.level_count; ++level) {
      const struct pipe_box face_box = {
         .x = 0,
         .y = 0,
         .width = (int)pvrgpu_resource_level_width(&src.base, level),
         .height = (int)pvrgpu_resource_level_height(&src.base, level),
         .depth = 1,
      };
      for (unsigned face = 0; face < 6; ++face) {
         struct pipe_box source = face_box;
         source.z = (int)face;
         CHECK(pvrgpu_can_copy_texture_region(&dst.base, level, 0, 0, face,
                                               &src.base, level, &source));
         pvrgpu_resource_copy_region(&ctx.base, &dst.base, level, 0, 0, face,
                                     &src.base, level, &source);
      }
   }
   CHECK(copies == before + src.level_count * 6);
   CHECK(unsupported == 0);
   CHECK(dst.driver_writes_model_cannot_reproduce);
   CHECK(memcmp(src.data, dst.data, src.size) == 0);
   FREE(src.data);
   FREE(dst.data);
}

static void
test_rejections(void)
{
   struct pvrgpu_resource rgb9 = cube(PIPE_FORMAT_R9G9B9E5_FLOAT);
   struct pvrgpu_resource uint32 = cube(PIPE_FORMAT_R32_UINT);
   struct pvrgpu_resource wide = cube(PIPE_FORMAT_R16G16B16A16_FLOAT);
   struct pvrgpu_resource depth = cube(PIPE_FORMAT_Z24X8_UNORM);
   struct pipe_box box = {.width = 4, .height = 3, .depth = 1};

   CHECK(!pvrgpu_texture_copy_formats_are_raw_compatible(
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R9G9B9E5_FLOAT));
   CHECK(!pvrgpu_texture_copy_formats_are_raw_compatible(
      PIPE_FORMAT_Z24X8_UNORM, PIPE_FORMAT_R32_UINT));
   CHECK(!pvrgpu_can_copy_texture_region(&wide.base, 0, 0, 0, 0,
                                          &rgb9.base, 0, &box));
   CHECK(!pvrgpu_can_copy_texture_region(&depth.base, 0, 0, 0, 0,
                                          &uint32.base, 0, &box));
   uint32.base.target = PIPE_TEXTURE_2D_ARRAY;
   CHECK(!pvrgpu_can_copy_texture_region(&uint32.base, 0, 0, 0, 0,
                                          &rgb9.base, 0, &box));
   uint32.base.target = PIPE_TEXTURE_CUBE;
   box.x = 16;
   CHECK(!pvrgpu_can_copy_texture_region(&uint32.base, 0, 0, 0, 0,
                                          &rgb9.base, 0, &box));
   box.x = 0;
   uint32.base.nr_samples = uint32.base.nr_storage_samples = 2;
   CHECK(!pvrgpu_can_copy_texture_region(&uint32.base, 0, 0, 0, 0,
                                          &rgb9.base, 0, &box));

   FREE(rgb9.data);
   FREE(uint32.data);
   FREE(wide.data);
   FREE(depth.data);
}

int main(void)
{
   test_cube_faces_and_mips(PIPE_FORMAT_R9G9B9E5_FLOAT,
                            PIPE_FORMAT_R32_UINT, 0);
   test_cube_faces_and_mips(PIPE_FORMAT_R32_UINT,
                            PIPE_FORMAT_R9G9B9E5_FLOAT, 1);
   test_rejections();
   printf("32-bit texture-view raw copy: %s (%u checks, %u failures)\n",
          failures ? "FAIL" : "PASS", checks, failures);
   return failures ? 1 : 0;
}
