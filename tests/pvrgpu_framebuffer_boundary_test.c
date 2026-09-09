/* SPDX-License-Identifier: MIT */
/* Test actual attachment transport, with only the bridge completion mocked.
 * This does not evaluate shaders or stand in for the native CTS regressions. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_resource.c"

static unsigned checks, failures, boundary_errors, color_calls, depth_calls;
static unsigned color_failure, depth_failure;
static bool replay_incomplete;
static uint64_t completed_generation = 7;
static enum pipe_format expected_color_formats[4];
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); } } while (0)

void pvrgpu_counter_eventf(const char *event, const char *format, ...)
{
   (void)format;
   boundary_errors += !strcmp(event, "framebuffer_boundary_flush_error");
}
bool pvrgpu_context_has_incomplete_replay(const struct pvrgpu_context *ctx)
{ (void)ctx; return replay_incomplete; }
void pvrgpu_context_end_frame_at_readback(struct pvrgpu_context *ctx)
{ ctx->array_primitive_draw_count = 0; }
uint64_t pvrgpu_systemc_submission_generation(void)
{ return completed_generation; }
bool pvrgpu_is_supported_color_format(enum pipe_format format)
{ return format == PIPE_FORMAT_R8G8B8A8_UNORM ||
         format == PIPE_FORMAT_R10G10B10A2_UNORM ||
         format == PIPE_FORMAT_B10G10R10A2_UNORM; }
bool pvrgpu_systemc_flush_readback_pixels(
   uint32_t width, uint32_t height, uint32_t bpp, uint32_t attachment,
   uint32_t samples, uint32_t depth_format, uint32_t layers,
   const char *color_format,
   uint8_t *pixels, size_t bytes, bool *written, char *error, size_t error_size)
{
   (void)depth_format; (void)error; (void)error_size;
   const bool depth = attachment == UINT32_MAX;
   CHECK(depth ? color_format == NULL : color_format &&
         attachment < ARRAY_SIZE(expected_color_formats) &&
         !strcmp(color_format, util_format_name(expected_color_formats[attachment])));
   if (depth) ++depth_calls; else ++color_calls;
   CHECK(bytes == (size_t)width * height * bpp * samples * layers);
   const unsigned failure = depth ? depth_failure : color_failure;
   *written = failure == 0;
   if (*written) memset(pixels, depth ? 0x6b : 0x5a + attachment, bytes);
   return failure != 1;
}

static struct pvrgpu_resource resource(enum pipe_format format, unsigned samples)
{
   struct pvrgpu_resource r = {0};
   r.base.target = PIPE_TEXTURE_2D;
   r.base.format = format;
   r.base.width0 = r.base.height0 = 2;
   r.base.depth0 = r.base.array_size = 1;
   r.base.nr_samples = r.base.nr_storage_samples = samples;
   r.base.bind = util_format_is_depth_or_stencil(format) ?
      PIPE_BIND_DEPTH_STENCIL : PIPE_BIND_RENDER_TARGET;
   CHECK(pvrgpu_init_resource_storage(&r));
   memset(r.data, 0x31, r.size);
   return r;
}

static void
test_mrt_boundary(unsigned samples, bool recorded_draws, bool mixed)
{
   struct pvrgpu_resource colors[4];
   for (unsigned i = 0; i < ARRAY_SIZE(colors); ++i) {
      expected_color_formats[i] = mixed && i == 1 ? PIPE_FORMAT_R10G10B10A2_UNORM :
         mixed && i == 2 ? PIPE_FORMAT_B10G10R10A2_UNORM : PIPE_FORMAT_R8G8B8A8_UNORM;
      colors[i] = resource(expected_color_formats[i], samples);
   }
   struct pvrgpu_resource depth = resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, samples);
   struct pvrgpu_context ctx = {0};
   ctx.framebuffer.width = ctx.framebuffer.height = 2;
   ctx.framebuffer.nr_cbufs = ARRAY_SIZE(colors);
   for (unsigned i = 0; i < ARRAY_SIZE(colors); ++i)
      ctx.framebuffer.cbufs[i] = (struct pipe_surface){
         .texture = &colors[i].base, .format = colors[i].base.format};
   ctx.framebuffer.zsbuf = (struct pipe_surface){
      .texture = &depth.base, .format = depth.base.format};
   const struct pipe_framebuffer_state framebuffer = ctx.framebuffer;
   ctx.color_readback_pending_mask = 15u | PVRGPU_DEPTH_READBACK_PENDING;
   ctx.color_readback_generation = completed_generation = 7;
   ctx.array_primitive_draw_count = recorded_draws ? 2 : 0;
   ctx.array_primitive_sequence_owns_command = recorded_draws;
   replay_incomplete = false;
   boundary_errors = color_calls = depth_calls = color_failure = depth_failure = 0;
   pvrgpu_flush_current_color_attachments(&ctx.base);
   CHECK(ctx.array_primitive_draw_count == 0);
   CHECK(ctx.color_readback_pending_mask == 0);
   CHECK(ctx.query_statistics_failures == 0 && boundary_errors == 0);
   CHECK(color_calls == 4 && depth_calls == 1);
   CHECK(memcmp(&framebuffer, &ctx.framebuffer, sizeof(framebuffer)) == 0);
   for (unsigned target = 0; target < ARRAY_SIZE(colors); ++target) {
      for (unsigned i = 0; i < colors[target].size; ++i)
         CHECK(colors[target].data[i] == 0x5a + target);
   }
   for (unsigned i = 0; i < depth.size; ++i) CHECK(depth.data[i] == 0x6b);
   /* A second flush must neither resubmit nor replace the completed bytes. */
   pvrgpu_flush_current_color_attachments(&ctx.base);
   CHECK(color_calls == 4 && depth_calls == 1);
   for (unsigned target = 0; target < ARRAY_SIZE(colors); ++target) {
      for (unsigned i = 0; i < colors[target].size; ++i)
         CHECK(colors[target].data[i] == 0x5a + target);
      FREE(colors[target].data);
   }
   for (unsigned i = 0; i < depth.size; ++i) CHECK(depth.data[i] == 0x6b);
   FREE(depth.data);
}

int main(void)
{
   for (unsigned samples = 1; samples <= 16; samples *= 2) {
      test_mrt_boundary(samples, false, false);
      test_mrt_boundary(samples, true, false);
      test_mrt_boundary(samples, false, true);
      test_mrt_boundary(samples, true, true);
   }
   expected_color_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   for (unsigned samples = 1; samples <= 16; samples *= 2)
   for (unsigned scenario = 0; scenario < 11; ++scenario) {
      struct pvrgpu_resource color = resource(PIPE_FORMAT_R8G8B8A8_UNORM, samples);
      struct pvrgpu_resource depth = resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, samples);
      struct pvrgpu_context ctx = {0};
      ctx.framebuffer.width = ctx.framebuffer.height = 2;
      ctx.framebuffer.nr_cbufs = 1;
      ctx.framebuffer.cbufs[0] = (struct pipe_surface){
         .texture = &color.base, .format = color.base.format};
      ctx.framebuffer.zsbuf = (struct pipe_surface){
         .texture = &depth.base, .format = depth.base.format};
      ctx.color_readback_pending_mask = 1u | PVRGPU_DEPTH_READBACK_PENDING;
      ctx.color_readback_generation = 7;
      ctx.query_statistics_failures = 5;
      boundary_errors = color_calls = depth_calls = 0;
      color_failure = scenario == 1 ? 1 : scenario == 2 ? 2 : 0;
      depth_failure = scenario == 3 ? 1 : scenario == 4 ? 2 : 0;
      replay_incomplete = scenario == 8;
      completed_generation = scenario == 5 ? 8 : 7;
      if (scenario == 6) ctx.framebuffer.cbufs[0].format = PIPE_FORMAT_R8_UNORM;
      if (scenario == 7) ctx.framebuffer.zsbuf.level = PIPE_MAX_TEXTURE_LEVELS;
      if (scenario == 9) ctx.color_readback_pending_mask = PVRGPU_DEPTH_READBACK_PENDING;
      if (scenario == 10) ctx.color_readback_pending_mask = 1;
      const bool expected_failure = scenario >= 1 && scenario <= 7;
      pvrgpu_flush_current_color_attachments(&ctx.base);
      CHECK(ctx.query_statistics_failures == 5u + expected_failure);
      CHECK(boundary_errors == expected_failure);
      CHECK(replay_incomplete || ctx.color_readback_pending_mask == 0);
      if (scenario == 0) {
         CHECK(color_calls == 1 && depth_calls == 1);
         for (unsigned i = 0; i < color.size; ++i) CHECK(color.data[i] == 0x5a);
         for (unsigned i = 0; i < depth.size; ++i) CHECK(depth.data[i] == 0x6b);
      }
      if (scenario == 1 || scenario == 2 || scenario == 5 || scenario == 6 || scenario == 8)
         for (unsigned i = 0; i < color.size; ++i) CHECK(color.data[i] == 0x31);
      if (scenario == 1 || scenario == 3 || scenario == 4 || scenario == 5 || scenario == 7 || scenario == 8)
         for (unsigned i = 0; i < depth.size; ++i) CHECK(depth.data[i] == 0x31);
      if (scenario == 1) CHECK(depth_calls == 0); /* Failed execution cannot expose stale Z. */
      if (scenario == 8) CHECK(color_calls == 0 && depth_calls == 0);
      if (scenario == 9) CHECK(color_calls == 0 && depth_calls == 1);
      if (scenario == 10) CHECK(color_calls == 1 && depth_calls == 0);
      FREE(color.data); FREE(depth.data);
   }
   printf("framebuffer boundary: %s (%u checks, %u failures)\n",
          failures ? "FAIL" : "PASS", checks, failures);
   return failures ? 1 : 0;
}
