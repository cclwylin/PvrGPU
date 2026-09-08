/* SPDX-License-Identifier: MIT */
/* Exercise the actual CPU-present refusal boundary and the diagnostic copy
 * control. This does not claim native shader execution; model-command writer
 * stubs below must never be called. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"

static unsigned checks, unsupported_events, present_events;
#define CHECK(c) do { ++checks; if (!(c)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

void pvrgpu_counter_eventf(const char *event, const char *format, ...)
{
   (void)format;
   unsupported_events += strcmp(event, "unsupported_draw") == 0;
   present_events += strcmp(event, "present_textured_quad") == 0;
}
bool pvrgpu_case_reserves_native_pco_sequence(void) { return false; }
bool pvrgpu_driver_draw_command_has_been_emitted(void) { return false; }
void pvrgpu_note_driver_draw_command_emitted(void) { CHECK(false); }
void pvrgpu_note_current_color_readback_pending(struct pvrgpu_context *ctx)
{ (void)ctx; CHECK(false); }
bool pvrgpu_write_clear_color_command(const char *path,
   const struct pvrgpu_clear_color_command *cmd, char *error, size_t size)
{ (void)path; (void)cmd; (void)error; (void)size; CHECK(false); return false; }

static void set_optional(const char *name, const char *value)
{ CHECK(value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0); }

static void test_configuration(const char *api, const char *bridge,
                               bool expected_refusal, bool previous_submission)
{
   uint8_t source[8 * 4], target[sizeof(source)], before[sizeof(source)];
   for (unsigned i = 0; i < sizeof(source); ++i) source[i] = i * 7 + 3;
   memset(target, 0xa5, sizeof(target));
   memcpy(before, target, sizeof(target));
   struct pvrgpu_resource src = {0}, dst = {0};
   struct pvrgpu_resource *resources[] = {&src, &dst};
   for (unsigned i = 0; i < 2; ++i) {
      resources[i]->base.target = PIPE_TEXTURE_2D;
      resources[i]->base.format = PIPE_FORMAT_R8G8B8A8_UNORM;
      resources[i]->base.width0 = 4;
      resources[i]->base.height0 = 2;
      resources[i]->base.depth0 = resources[i]->base.array_size = 1;
      resources[i]->data = i ? target : source;
      resources[i]->size = sizeof(source);
      resources[i]->level_count = 1;
      resources[i]->level_strides[0] = 16;
      resources[i]->level_layer_strides[0] = sizeof(source);
   }
   struct pipe_sampler_view view = {0};
   view.texture = &src.base;
   view.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   view.swizzle_r = PIPE_SWIZZLE_X;
   view.swizzle_g = PIPE_SWIZZLE_Y;
   view.swizzle_b = PIPE_SWIZZLE_Z;
   view.swizzle_a = PIPE_SWIZZLE_W;
   struct pvrgpu_sampler_state sampler = {0};
   struct pvrgpu_context ctx = {0};
   ctx.framebuffer.width = 4;
   ctx.framebuffer.height = 2;
   ctx.framebuffer.nr_cbufs = 1;
   ctx.framebuffer.cbufs[0] = (struct pipe_surface){
      .texture = &dst.base, .format = PIPE_FORMAT_R8G8B8A8_UNORM};
   ctx.num_sampler_views[MESA_SHADER_FRAGMENT] = 1;
   ctx.num_samplers[MESA_SHADER_FRAGMENT] = 1;
   ctx.sampler_views[MESA_SHADER_FRAGMENT][0] = &view;
   ctx.samplers[MESA_SHADER_FRAGMENT][0] = &sampler;
   ctx.color_readback_generation = 37;
   ctx.driver_draw_command_emitted = previous_submission;
   ctx.full_depth_clear_is_one = true;
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLE_STRIP};
   struct pipe_draw_start_count_bias draw = {.count = 4};
   set_optional("PVRGPU_SYSTEMC_API_LIB", api);
   set_optional("PVRGPU_SYSTEMC_BRIDGE", bridge);
   unsupported_events = present_events = 0;
   CHECK(pvrgpu_can_cpu_present_textured_quad(&ctx, &info, NULL, &draw, 1));
   const bool refused = pvrgpu_refuse_native_cpu_present(
      &ctx, &info, NULL, &draw, 1);
   CHECK(refused == expected_refusal);
   CHECK(memcmp(target, before, sizeof(target)) == 0);
   CHECK(ctx.unsupported_draws == (unsigned)refused);
   CHECK(ctx.observed_draws == (unsigned)refused);
   CHECK(unsupported_events == (unsigned)refused && present_events == 0);
   CHECK(ctx.color_readback_generation == 37 &&
         ctx.driver_draw_command_emitted == previous_submission);
   CHECK(ctx.full_depth_clear_is_one == !refused);
   if (!refused) {
      CHECK(pvrgpu_cpu_present_textured_quad(&ctx, &info, NULL, &draw, 1));
      CHECK(memcmp(target, source, sizeof(target)) == 0);
      CHECK(present_events == 1 && unsupported_events == 0);
   }
   const unsigned prior = ctx.unsupported_draws;
   draw.count = 3;
   CHECK(!pvrgpu_refuse_native_cpu_present(&ctx, &info, NULL, &draw, 1));
   CHECK(ctx.unsupported_draws == prior);
   CHECK(!pvrgpu_refuse_native_cpu_present(NULL, &info, NULL, &draw, 1));
}

int main(void)
{
   unsetenv("PVRGPU_DRIVER_COMMAND_OUT");
   unsetenv("PVRGPU_RDC_CASE_NAME");
   unsetenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
   const char *values[] = {NULL, "", "/deliberately/not/a/model/library"};
   for (unsigned a = 0; a < 3; ++a)
      for (unsigned b = 0; b < 3; ++b)
         for (unsigned previous = 0; previous < 2; ++previous)
            test_configuration(values[a], values[b], a == 2 || b == 2,
                               previous != 0);
   puts("native CPU-present guard: PASS");
   printf("%u checks; diagnostic-only CPU copy control, no native work claimed\n", checks);
   return 0;
}
