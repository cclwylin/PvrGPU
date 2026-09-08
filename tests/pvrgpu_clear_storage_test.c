/* SPDX-License-Identifier: MIT */
/* Run with script/run_mesa_resource_unit.sh clear. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_clear.c"

static unsigned failures, checks, flush_calls, flush_mode, color_events, depth_events;
static unsigned stencil_events, clear_errors, command_calls;
static struct pvrgpu_clear_color_command last_command;
#define CHECK(condition) do { \
   ++checks; \
   if (!(condition)) { \
      fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #condition); \
      ++failures; \
   } \
} while (0)

/* Only the submission boundary and legacy capsule sink are mocked. The clear
 * entry point, format packers and every byte store are the actual driver. */
void pvrgpu_counter_event(const char *event, const char *detail)
{ (void)detail; clear_errors += !strcmp(event, "clear_error"); }
void pvrgpu_counter_eventf(const char *event, const char *format, ...)
{
   (void)format;
   color_events += !strcmp(event, "clear_color");
   depth_events += !strcmp(event, "clear_depth");
   stencil_events += !strcmp(event, "clear_stencil");
   clear_errors += !strcmp(event, "clear_error");
}
void pvrgpu_flush_current_color_attachments(struct pipe_context *pipe)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ++flush_calls;
   if (flush_mode == 1) {
      ++ctx->query_statistics_failures;
      return;
   }
   if (flush_mode == 2)
      return;
   ctx->array_primitive_draw_count = 0;
   ctx->color_readback_pending_mask = 0;
}
void pvrgpu_invalidate_full_depth_clear(struct pvrgpu_context *ctx)
{ ctx->full_depth_clear_is_one = false; }
void pvrgpu_note_full_depth_clear_one(struct pvrgpu_context *ctx,
                                    const struct pipe_surface *surface,
                                    unsigned width, unsigned height)
{ (void)surface; (void)width; (void)height; ctx->full_depth_clear_is_one = true; }
bool pvrgpu_note_pending_attachment_clear(struct pvrgpu_context *ctx,
   unsigned x, unsigned y, unsigned width, unsigned height, unsigned aspects,
   uint32_t depth_bits, unsigned stencil)
{
   (void)x; (void)y; (void)width; (void)height; (void)aspects;
   (void)depth_bits; (void)stencil; ++ctx->pending_attachment_clear_count; return true;
}
bool pvrgpu_case_reserves_native_pco_sequence(void) { return false; }
bool pvrgpu_driver_draw_command_has_been_emitted(void) { return false; }
bool pvrgpu_write_clear_color_command(const char *path,
   const struct pvrgpu_clear_color_command *command, char *error, size_t error_size)
{
   (void)path; (void)error; (void)error_size;
   ++command_calls; last_command = *command; return true;
}

static struct pvrgpu_resource
test_resource(enum pipe_format format, unsigned samples)
{
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_TEXTURE_2D;
   resource.base.format = format;
   resource.base.width0 = 3;
   resource.base.height0 = 2;
   resource.base.depth0 = resource.base.array_size = 1;
   resource.base.nr_samples = resource.base.nr_storage_samples = samples;
   resource.level_count = 1;
   resource.stride = resource.level_strides[0] =
      3 * samples * util_format_get_blocksize(format);
   resource.layer_stride = resource.level_layer_strides[0] = resource.stride * 2;
   resource.size = resource.layer_stride;
   resource.data = calloc(1, resource.size);
   CHECK(resource.data != NULL);
   return resource;
}

static struct pipe_surface
test_surface(struct pvrgpu_resource *resource)
{
   struct pipe_surface surface = {0};
   surface.texture = &resource->base;
   surface.format = resource->base.format;
   return surface;
}

static void
test_integer_color(enum pipe_format format)
{
   struct pvrgpu_resource resource = test_resource(format, 16);
   struct pipe_surface surface = test_surface(&resource);
   union pipe_color_union color;
   color.ui[0] = UINT32_C(0xffffffff);
   color.ui[1] = UINT32_C(0x80000000);
   color.ui[2] = UINT32_C(0x7fffffff);
   color.ui[3] = UINT32_C(0xdeadbeef);
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 0, 0, 3, 2,
                                              PIPE_MASK_RGBA, &color);
   for (unsigned texel = 0; texel < 6 * 16; ++texel)
      CHECK(memcmp(resource.data + texel * 16, color.ui, 16) == 0);

   memset(resource.data, 0xa5, resource.size);
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 1, 1, 1, 1,
                                              PIPE_MASK_R | PIPE_MASK_A, &color);
   for (unsigned texel = 0; texel < 6 * 16; ++texel) {
      const bool selected = texel >= 4 * 16 && texel < 5 * 16;
      const uint32_t *pixel = (const uint32_t *)resource.data + texel * 4;
      for (unsigned channel = 0; channel < 4; ++channel)
         CHECK(pixel[channel] ==
               (selected && (channel == 0 || channel == 3)
                   ? color.ui[channel] : UINT32_C(0xa5a5a5a5)));
   }
   free(resource.data);
}

static void
test_float_color(void)
{
   struct pvrgpu_resource resource = test_resource(PIPE_FORMAT_R32G32B32A32_FLOAT, 4);
   struct pipe_surface surface = test_surface(&resource);
   union pipe_color_union color = {.f = {-2.0F, 4.0F, 1234.5F, 0.125F}};
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 0, 0, 3, 2,
                                              PIPE_MASK_RGBA, &color);
   for (unsigned texel = 0; texel < 6 * 4; ++texel)
      CHECK(memcmp(resource.data + texel * 16, color.f, 16) == 0);

   uint32_t *words = (uint32_t *)resource.data;
   for (unsigned word = 0; word < resource.size / 4; ++word)
      words[word] = UINT32_C(0x7fc12345);
   struct pipe_framebuffer_state fb = {.width = 3, .height = 2};
   struct pipe_scissor_state scissor = {.minx = 1, .miny = 1, .maxx = 2, .maxy = 2};
   unsigned x, y, width, height;
   CHECK(pvrgpu_clear_rect(&fb, &scissor, &x, &y, &width, &height));
   CHECK(x == 1 && y == 1 && width == 1 && height == 1);
   pvrgpu_fill_surface_rect_with_clear_color(&surface, x, y, width, height,
                                              PIPE_MASK_R, &color);
   for (unsigned texel = 0; texel < 6 * 4; ++texel)
      for (unsigned channel = 0; channel < 4; ++channel)
         CHECK(words[texel * 4 + channel] ==
               (texel >= 4 * 4 && texel < 5 * 4 && channel == 0
                  ? color.ui[0] : UINT32_C(0x7fc12345)));
   const uint32_t first = words[0];
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 0, 0, 3, 2, 0, &color);
   CHECK(words[0] == first);
   free(resource.data);
}

static void
test_depth_stencil(void)
{
   struct pvrgpu_resource resource = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 4);
   struct pipe_surface surface = test_surface(&resource);
   uint32_t *words = (uint32_t *)resource.data;
   for (unsigned word = 0; word < resource.size / 4; ++word)
      words[word] = UINT32_C(0xab123456);
   CHECK(pvrgpu_fill_surface_rect_with_clear_depth(&surface, 1, 1, 1, 1, 0.5));
   for (unsigned sample = 0; sample < 6 * 4; ++sample)
      CHECK(words[sample] == (sample >= 16 && sample < 20
                                 ? UINT32_C(0xab7fffff) : UINT32_C(0xab123456)));
   CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(&surface, 1, 1, 1, 1,
                                                      0x31, 0x0f));
   for (unsigned sample = 0; sample < 6 * 4; ++sample)
      CHECK(words[sample] == (sample >= 16 && sample < 20
                                 ? UINT32_C(0xa17fffff) : UINT32_C(0xab123456)));
   free(resource.data);
}

static void
test_depth_layers(enum pipe_format format, unsigned samples,
                  enum pipe_texture_target target, unsigned layers,
                  unsigned level)
{
   const unsigned block_size = util_format_get_blocksize(format);
   const bool has_stencil =
      util_format_pack_description(format)->pack_s_8uint != NULL;
   struct pvrgpu_resource resource = test_resource(format, samples);
   free(resource.data);
   resource.base.target = target;
   resource.base.width0 = 3u << level;
   resource.base.height0 = 2u << level;
   resource.base.array_size = layers;
   resource.level_count = level + 1;
   /* Keep a prefix, row padding, layer padding and a suffix as guards. */
   resource.level_offsets[level] = 16;
   resource.level_strides[level] = 3 * samples * block_size + 16;
   resource.level_layer_strides[level] =
      resource.level_strides[level] * 2 + 16;
   resource.size = 32 + layers * resource.level_layer_strides[level];
   resource.data = malloc(resource.size);
   uint8_t *expected = malloc(resource.size);
   CHECK(resource.data && expected);
   if (!resource.data || !expected) {
      free(resource.data);
      free(expected);
      return;
   }
   memset(resource.data, 0xa5, resource.size);
   struct pipe_surface surface = test_surface(&resource);
   surface.level = level;
   const float initial_depth = 0.25f;
   const uint8_t initial_stencil = 0xab;
   for (unsigned layer = 0; layer < layers; ++layer) {
      for (unsigned y = 0; y < 2; ++y) {
         for (unsigned x = 0; x < 3 * samples; ++x) {
            uint8_t *pixel = resource.data + 16 +
               layer * resource.level_layer_strides[level] +
               y * resource.level_strides[level] + x * block_size;
            util_format_pack_z_float(format, pixel, &initial_depth, 1);
            if (has_stencil)
               util_format_pack_s_8uint(format, pixel, &initial_stencil, 1);
         }
      }
   }
   memcpy(expected, resource.data, resource.size);

   /* Match ordinary FBO attachment switches: a full clear of each single
    * layer must initialize every sample, including pixels never rasterized. */
   const float full_depth = 0.9375f;
   for (unsigned layer = 0; layer < layers; ++layer) {
      surface.first_layer = surface.last_layer = layer;
      CHECK(pvrgpu_fill_surface_rect_with_clear_depth(
         &surface, 0, 0, 3, 2, full_depth));
      for (unsigned y = 0; y < 2; ++y)
         for (unsigned x = 0; x < 3 * samples; ++x)
            util_format_pack_z_float(format,
               expected + 16 + layer * resource.level_layer_strides[level] +
               y * resource.level_strides[level] + x * block_size,
               &full_depth, 1);
      CHECK(memcmp(resource.data, expected, resource.size) == 0);
   }

   /* A scissored layer range preserves layer 0, all borders, all padding,
    * and the other aspect of packed depth/stencil formats. */
   surface.first_layer = 1;
   surface.last_layer = layers - 1;
   CHECK(pvrgpu_fill_surface_rect_with_clear_depth(&surface, 1, 1, 1, 1, 0.5));
   if (has_stencil) {
      CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
         &surface, 1, 1, 1, 1, 0x31, 0x0f));
      CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
         &surface, 0, 0, 3, 2, 0, 0));
   }
   const float partial_depth = 0.5f;
   const uint8_t partial_stencil = 0xa1;
   for (unsigned layer = 1; layer < layers; ++layer)
      for (unsigned sample = 0; sample < samples; ++sample) {
         uint8_t *pixel = expected + 16 +
            layer * resource.level_layer_strides[level] +
            resource.level_strides[level] + (samples + sample) * block_size;
         util_format_pack_z_float(format, pixel, &partial_depth, 1);
         if (has_stencil)
            util_format_pack_s_8uint(format, pixel, &partial_stencil, 1);
      }
   CHECK(memcmp(resource.data, expected, resource.size) == 0);

   /* Malformed metadata must fail before the first store, including when
    * the logical layer exists but the allocation cannot hold that layer. */
   const struct pvrgpu_resource saved = resource;
   const struct pipe_surface saved_surface = surface;
   for (unsigned invalid = 0; invalid < 8; ++invalid) {
      resource = saved;
      surface = saved_surface;
      switch (invalid) {
      case 0: surface.first_layer = 2; surface.last_layer = 1; break;
      case 1: surface.last_layer = layers; break;
      case 2: resource.level_strides[level] = 3 * samples * block_size - 1; break;
      case 3: resource.level_layer_strides[level] =
                 resource.level_strides[level] * 2 - 1; break;
      case 4: resource.size = resource.level_offsets[level] +
                 layers * resource.level_layer_strides[level] - 1; break;
      case 5: resource.level_offsets[level] = SIZE_MAX; break;
      case 6: surface.level = level + 1; break;
      case 7: resource.level_layer_strides[level] = SIZE_MAX; break;
      }
      CHECK(!pvrgpu_fill_surface_rect_with_clear_depth(
         &surface, 0, 0, 3, 2, 0.0));
      CHECK(!pvrgpu_fill_surface_rect_with_clear_stencil(
         &surface, 0, 0, 3, 2, 0, 0xff));
      CHECK(memcmp(saved.data, expected, saved.size) == 0);
   }
   resource = saved;
   surface = saved_surface;
   CHECK(!pvrgpu_fill_surface_rect_with_clear_depth(
      &surface, 2, 1, 2, 1, 0.0));
   CHECK(memcmp(resource.data, expected, resource.size) == 0);
   free(resource.data);
   free(expected);
}

static void
test_uniform_includes_last_sample(void)
{
   struct pvrgpu_resource resource = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 4);
   struct pipe_surface surface = test_surface(&resource);
   memset(resource.data, 31, resource.size);
   float color[4];
   CHECK(pvrgpu_surface_uniform_color(&surface, 3, 2, color));
   resource.data[resource.size - 1] = 32;
   CHECK(!pvrgpu_surface_uniform_color(&surface, 3, 2, color));
   free(resource.data);
}

static void
test_pending_clear_lifecycle(void)
{
   struct pvrgpu_context ctx = {0};
   const unsigned masks[] = {0, 0x0f, 0xff};
   for (unsigned samples = 1; samples <= 16; samples *= 2) {
      for (unsigned mask_index = 0;
           mask_index < sizeof(masks) / sizeof(masks[0]); ++mask_index) {
         const unsigned mask = masks[mask_index];
         struct pvrgpu_resource resource =
            test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, samples);
         struct pipe_surface surface = test_surface(&resource);
         uint32_t *words = (uint32_t *)resource.data;
         for (unsigned sample = 0; sample < 6 * samples; ++sample)
            words[sample] = UINT32_C(0xa5123456);
         CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
            &surface, 1, 1, 1, 1, 0x12, mask));

         /* Capture the actual native LOAD, then retire rectangles it already
          * contains. Replaying a rectangle with its implicit full-byte mask
          * would change A5/A2 to 12, despite the CPU clear being correct. */
         void *load = malloc(resource.size);
         CHECK(load != NULL);
         memcpy(load, resource.data, resource.size);
         ctx.array_primitive_draw_count = 0;
         ctx.pending_attachment_clear_count = 1;
         pvrgpu_retire_materialized_attachment_clears(&ctx, false, resource.size);
         CHECK(ctx.pending_attachment_clear_count == 0);
         resource.data = load;
         if (ctx.pending_attachment_clear_count)
            CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
               &surface, 1, 1, 1, 1, 0x12, 0xff));
         const uint32_t stencil = (0xa5 & ~mask) | (0x12 & mask);
         for (unsigned sample = 0; sample < 6 * samples; ++sample)
            CHECK(((uint32_t *)load)[sample] ==
                  (sample >= 4 * samples && sample < 5 * samples
                     ? (stencil << 24) | UINT32_C(0x123456)
                     : UINT32_C(0xa5123456)));
         free(words);
         free(load);
      }
   }

   /* No new LOAD: keep the ordered clear between two draws in an incomplete
    * RDC replay. Merely selecting the same framebuffer is not a boundary. */
   ctx.array_primitive_draw_count = 2;
   ctx.pending_attachment_clear_count = 3;
   pvrgpu_retire_materialized_attachment_clears(&ctx, false, 0);
   CHECK(ctx.pending_attachment_clear_count == 3);
   ctx.array_primitive_draw_count = 0;
   pvrgpu_retire_materialized_attachment_clears(&ctx, false, 0);
   CHECK(ctx.pending_attachment_clear_count == 3);

   /* An A -> B attachment-key change invalidates A's unconsumed rectangles,
    * even if A had no draw. B's own contents must not receive A's clear. */
   struct pvrgpu_resource other = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 4);
   struct pipe_surface other_surface = test_surface(&other);
   memset(other.data, 0x3c, other.size);
   pvrgpu_retire_materialized_attachment_clears(&ctx, true, 0);
   CHECK(ctx.pending_attachment_clear_count == 0);
   if (ctx.pending_attachment_clear_count)
      CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
         &other_surface, 1, 1, 1, 1, 0x12, 0xff));
   for (unsigned byte = 0; byte < other.size; ++byte)
      CHECK(other.data[byte] == 0x3c);
   free(other.data);
}

static void
test_mrt_clear_entry(unsigned samples, unsigned targets, unsigned selected,
                     unsigned channel_pattern, unsigned aspects, bool scissored,
                     enum pipe_format format, bool shared_mask)
{
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_resource colors[PIPE_MAX_COLOR_BUFS] = {0};
   struct pvrgpu_resource depth = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, samples);
   ctx.framebuffer.width = 3;
   ctx.framebuffer.height = 2;
   ctx.framebuffer.nr_cbufs = targets;
   ctx.framebuffer.zsbuf = test_surface(&depth);
   memset(depth.data, 0xa5, depth.size);
   ctx.stencil_clear_value = 0xa5;
   ctx.array_primitive_draw_count = 1;
   ctx.color_readback_pending_mask = selected;
   union pipe_color_union color = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   uint8_t packed[16] = {0};
   util_format_pack_rgba(format, packed, &color, 1);
   const unsigned bpp = util_format_get_blocksize(format);
   unsigned buffers = aspects, masks = 0;
   for (unsigned target = 0; target < targets; ++target) {
      colors[target] = test_resource(format, samples);
      memset(colors[target].data, 0x31 + target, colors[target].size);
      ctx.framebuffer.cbufs[target] = test_surface(&colors[target]);
      if (selected & (1u << target))
         buffers |= PIPE_CLEAR_COLOR0 << target;
      masks |= ((channel_pattern + (shared_mask ? 0 : target)) & 15u) << (4 * target);
   }
   const struct pipe_scissor_state scissor = {.minx=1, .miny=1, .maxx=2, .maxy=2};
   const unsigned prior_flush = flush_calls, prior_commands = command_calls;
   const unsigned prior_depth = depth_events, prior_stencil = stencil_events;
   flush_mode = 0;
   pvrgpu_clear(&ctx.base, buffers, masks, 0x0f,
                scissored ? &scissor : NULL, &color, 1.0, 0x32);
   CHECK(flush_calls == prior_flush + 1);
   CHECK(command_calls == prior_commands);
   CHECK(ctx.query_statistics_failures == 0);
   CHECK(ctx.array_primitive_draw_count == 0 && ctx.color_readback_pending_mask == 0);
   CHECK(depth_events == prior_depth + !!(aspects & PIPE_CLEAR_DEPTH));
   CHECK(stencil_events == prior_stencil + !!(aspects & PIPE_CLEAR_STENCIL));
   CHECK(ctx.stencil_clear_value == 0xa5); /* Partial mask must not claim full 0x32. */
   for (unsigned target = 0; target < targets; ++target) {
      const unsigned mask = (masks >> (4 * target)) & 15;
      const bool writes = (selected & (1u << target)) && mask;
      CHECK(colors[target].driver_writes_model_cannot_reproduce == writes);
      for (unsigned pixel = 0; pixel < 6; ++pixel)
         for (unsigned sample = 0; sample < samples; ++sample)
            for (unsigned byte = 0; byte < bpp; ++byte) {
               const unsigned channel = byte / (bpp / 4);
               const bool changed = writes && (!scissored || pixel == 4) && (mask & (1u << channel));
               CHECK(colors[target].data[(pixel * samples + sample) * bpp + byte] ==
                     (changed ? packed[byte] : 0x31 + target));
            }
      free(colors[target].data);
   }
   for (unsigned pixel = 0; pixel < 6; ++pixel)
      for (unsigned sample = 0; sample < samples; ++sample) {
         const bool inside = !scissored || pixel == 4;
         const uint32_t z = inside && (aspects & PIPE_CLEAR_DEPTH) ? 0xffffff : 0xa5a5a5;
         const uint32_t s = inside && (aspects & PIPE_CLEAR_STENCIL) ? 0xa2 : 0xa5;
         CHECK(((uint32_t *)depth.data)[pixel * samples + sample] == (s << 24 | z));
      }
   free(depth.data);
}

static void
test_mrt_clear_atomic_rejection(void)
{
   for (unsigned invalid = 0; invalid < 8; ++invalid) {
      struct pvrgpu_context ctx = {0};
      struct pvrgpu_resource colors[4];
      struct pvrgpu_resource depth = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 1);
      ctx.framebuffer.width = 3;
      ctx.framebuffer.height = 2;
      ctx.framebuffer.nr_cbufs = 4;
      ctx.framebuffer.zsbuf = test_surface(&depth);
      memset(depth.data, 0x5a, depth.size);
      for (unsigned target = 0; target < 4; ++target) {
         colors[target] = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 1);
         memset(colors[target].data, 0x5a, colors[target].size);
         ctx.framebuffer.cbufs[target] = test_surface(&colors[target]);
      }
      const size_t color_size = colors[3].size, depth_size = depth.size;
      double depth_value = 1.0;
      flush_mode = 0;
      switch (invalid) {
      case 0: colors[3].size = 1; break;
      case 1: ctx.framebuffer.cbufs[3].texture = NULL; break;
      case 2: depth.size = 1; break;
      case 3: depth_value = NAN; break;
      case 4: ctx.framebuffer.zsbuf.format = PIPE_FORMAT_Z32_FLOAT; break;
      case 5: colors[3].level_strides[0] = 1; break;
      case 6: flush_mode = 1; break;
      case 7: flush_mode = 2; ctx.array_primitive_draw_count = 2; break;
      }
      const union pipe_color_union color = {.f = {1, 0, 0, 1}};
      const unsigned prior_flush = flush_calls, prior_errors = clear_errors;
      const unsigned prior_colors = color_events, prior_depth = depth_events;
      pvrgpu_clear(&ctx.base, (PIPE_CLEAR_COLOR0 * 15) | PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL,
                   0xffff, 0xff, NULL, &color, depth_value, 0xff);
      CHECK(clear_errors == prior_errors + 1);
      CHECK(ctx.query_statistics_failures == 1);
      CHECK(color_events == prior_colors && depth_events == prior_depth);
      CHECK(flush_calls == prior_flush + (invalid >= 6));
      for (unsigned target = 0; target < 4; ++target) {
         for (unsigned byte = 0; byte < color_size; ++byte)
            CHECK(colors[target].data[byte] == 0x5a);
         free(colors[target].data);
      }
      for (unsigned byte = 0; byte < depth_size; ++byte)
         CHECK(depth.data[byte] == 0x5a);
      free(depth.data);
   }
   flush_mode = 0;
}

static void
test_mrt_clear_layers(void)
{
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_resource resources[5];
   ctx.framebuffer.width = 3;
   ctx.framebuffer.height = 2;
   ctx.framebuffer.nr_cbufs = 4;
   for (unsigned target = 0; target < 5; ++target) {
      struct pvrgpu_resource *r = &resources[target];
      *r = test_resource(target == 4 ? PIPE_FORMAT_Z24_UNORM_S8_UINT :
                                     PIPE_FORMAT_R8G8B8A8_UNORM, 4);
      free(r->data);
      r->base.target = PIPE_TEXTURE_2D_ARRAY;
      r->base.array_size = 3;
      r->level_offsets[0] = 16;
      r->level_strides[0] += 16;
      r->level_layer_strides[0] = r->level_strides[0] * 2 + 16;
      r->size = 32 + 3 * r->level_layer_strides[0];
      r->data = malloc(r->size);
      CHECK(r->data != NULL);
      memset(r->data, 0x2a, r->size);
      struct pipe_surface surface = test_surface(r);
      surface.first_layer = 1;
      surface.last_layer = 2;
      if (target == 4) ctx.framebuffer.zsbuf = surface;
      else ctx.framebuffer.cbufs[target] = surface;
   }
   const union pipe_color_union color = {.f = {0.25, 0.5, 0.75, 1}};
   const struct pipe_scissor_state scissor = {.minx=1, .miny=1, .maxx=2, .maxy=2};
   pvrgpu_clear(&ctx.base, (PIPE_CLEAR_COLOR0 * 15) | PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL,
                0x8421, 0xf0, &scissor, &color, 0.5, 0xf3);
   CHECK(ctx.query_statistics_failures == 0);
   const uint8_t packed[] = {64, 128, 191, 255};
   for (unsigned target = 0; target < 5; ++target) {
      struct pvrgpu_resource *r = &resources[target];
      uint8_t *expected = malloc(r->size);
      CHECK(expected != NULL);
      memset(expected, 0x2a, r->size);
      for (unsigned layer = 1; layer <= 2; ++layer)
         for (unsigned sample = 0; sample < 4; ++sample) {
            const size_t offset = 16 + layer * r->level_layer_strides[0] +
                                  r->level_strides[0] + (4 + sample) * 4;
            if (target == 4) {
               const uint32_t word = UINT32_C(0xfa7fffff);
               memcpy(expected + offset, &word, sizeof(word));
            } else {
               expected[offset + target] = packed[target];
            }
         }
      CHECK(memcmp(r->data, expected, r->size) == 0);
      free(expected);
      free(r->data);
   }
}

static void
test_clear_holes_and_legacy(void)
{
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_resource color = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 1);
   struct pvrgpu_resource depth = test_resource(PIPE_FORMAT_Z32_FLOAT, 1);
   const union pipe_color_union value = {.f = {0.25, 0.5, 0.75, 1}};
   ctx.framebuffer.width = 3;
   ctx.framebuffer.height = 2;
   ctx.framebuffer.nr_cbufs = 8;
   ctx.framebuffer.cbufs[7] = test_surface(&color);
   ctx.framebuffer.zsbuf = test_surface(&depth);
   /* RT6 is selected but channel-masked out; RT0..5 are unselected holes.
    * A zero stencil mask needs no stencil aspect in this depth-only format. */
   pvrgpu_clear(&ctx.base, PIPE_CLEAR_COLOR7 | PIPE_CLEAR_COLOR6 |
                PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL,
                UINT32_C(0xf0000000), 0, NULL, &value, 1, 0xff);
   CHECK(ctx.query_statistics_failures == 0);
   for (unsigned pixel = 0; pixel < 6; ++pixel) {
      CHECK(color.data[pixel * 4] == 64 && color.data[pixel * 4 + 3] == 255);
      CHECK(((float *)depth.data)[pixel] == 1.0f);
   }
   memset(ctx.framebuffer.cbufs, 0, sizeof(ctx.framebuffer.cbufs));
   ctx.framebuffer.nr_cbufs = 1;
   ctx.framebuffer.cbufs[0] = test_surface(&color);
   CHECK(setenv("PVRGPU_DRIVER_COMMAND_OUT", "/unused/mock-clear-command.txt", 1) == 0);
   unsetenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
   const unsigned prior = command_calls;
   pvrgpu_clear(&ctx.base, PIPE_CLEAR_COLOR0 | PIPE_CLEAR_DEPTH, 15, 0,
                NULL, &value, 0.25, 0);
   CHECK(command_calls == prior + 1);
   CHECK(last_command.width == 3 && last_command.height == 2);
   CHECK(memcmp(last_command.clear_color_bits, value.f, 16) == 0);
   CHECK(!color.driver_writes_model_cannot_reproduce);
   unsetenv("PVRGPU_DRIVER_COMMAND_OUT");
   free(color.data);
   free(depth.data);
}

int main(void)
{
   test_integer_color(PIPE_FORMAT_R32G32B32A32_UINT);
   test_integer_color(PIPE_FORMAT_R32G32B32A32_SINT);
   test_float_color();
   test_depth_stencil();
   const enum pipe_format depth_formats[] = {
      PIPE_FORMAT_Z32_FLOAT, PIPE_FORMAT_Z24_UNORM_S8_UINT,
      PIPE_FORMAT_Z32_FLOAT_S8X24_UINT
   };
   for (unsigned format = 0; format < 3; ++format) {
      for (unsigned samples = 1; samples <= 16; samples *= 2)
         test_depth_layers(depth_formats[format], samples,
                           PIPE_TEXTURE_2D_ARRAY, 3, 0);
      test_depth_layers(depth_formats[format], 1, PIPE_TEXTURE_2D_ARRAY, 3, 1);
      test_depth_layers(depth_formats[format], 1, PIPE_TEXTURE_CUBE, 6, 1);
   }
   test_uniform_includes_last_sample();
   test_pending_clear_lifecycle();
   for (unsigned selected = 1; selected < 16; ++selected)
      for (unsigned channels = 0; channels < 16; ++channels)
         for (unsigned scoped = 0; scoped < 2; ++scoped)
            for (unsigned samples = 1; samples <= 4; samples *= 4)
               test_mrt_clear_entry(samples, 4, selected, channels,
                  PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL, scoped, PIPE_FORMAT_R8G8B8A8_UNORM, false);
   for (unsigned aspects = 0; aspects < 4; ++aspects) {
      const unsigned flags = (aspects & 1 ? PIPE_CLEAR_DEPTH : 0) |
                             (aspects & 2 ? PIPE_CLEAR_STENCIL : 0);
      test_mrt_clear_entry(16, 8, 0x85, 9, flags, true, PIPE_FORMAT_R32G32B32A32_UINT, false);
      test_mrt_clear_entry(4, 4, 15, 15, flags, false, PIPE_FORMAT_R32G32B32A32_FLOAT, false);
   }
   for (unsigned samples = 1; samples <= 16; samples *= 2)
      test_mrt_clear_entry(samples, 4, 15, 15, PIPE_CLEAR_DEPTH,
                           false, PIPE_FORMAT_R8G8B8A8_UNORM, true);
   test_mrt_clear_atomic_rejection();
   test_mrt_clear_layers();
   test_clear_holes_and_legacy();
   if (!failures)
      printf("Mesa clear storage tests passed (%u checks)\n", checks);
   return failures ? 1 : 0;
}
