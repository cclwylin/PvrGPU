/* SPDX-License-Identifier: MIT */
/* Run with script/run_mesa_resource_unit.sh flush.
 * Exercise the actual pipe.flush and fence callbacks. Only the synchronous
 * framebuffer materialization boundary is mocked; shader execution is not
 * claimed by this test. Actual attachment transport has a separate fixture. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"
#include "../src/gallium/drivers/pvrgpu/pvrgpu_screen.c"

static unsigned checks, failures, materializations, references;
static unsigned completion_mode;
static bool expected_materialization;
static bool global_emitted;
static struct pipe_fence_handle *old_fence;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); } } while (0)

void
pvrgpu_counter_eventf(const char *event, const char *format, ...)
{ (void)event; (void)format; }

bool
pvrgpu_driver_draw_command_has_been_emitted(void)
{ return global_emitted; }

void
pvrgpu_flush_current_color_attachments(struct pipe_context *pipe)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ++materializations;
   CHECK(references == 0); /* No completion may be advertised first. */
   if (completion_mode == 1) {
      /* Real transport reports errors even after clearing its pending mask. */
      ++ctx->query_statistics_failures;
      ctx->array_primitive_draw_count = 0;
      ctx->color_readback_pending_mask = 0;
   } else if (completion_mode == 0 || completion_mode == 3) {
      ctx->array_primitive_draw_count = 0;
      ctx->color_readback_pending_mask = 0;
      if (completion_mode == 3) {
         ctx->driver_draw_command_emitted = false;
         global_emitted = false;
      }
   }
   /* Mode 2 intentionally leaves uncompleted work, without a new error. */
}

static void
checked_reference(struct pipe_screen *screen, struct pipe_fence_handle **out,
                  struct pipe_fence_handle *fence)
{
   CHECK(out && *out == old_fence);
   CHECK(!expected_materialization || materializations == 1);
   ++references;
   pvrgpu_fence_reference(screen, out, fence);
}

static void
test_flush(unsigned flags, unsigned pending, unsigned mode, bool request_fence)
{
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_screen screen = {0};
   struct pipe_resource resources[3] = {{0}};
   screen.base.fence_reference = checked_reference;
   screen.base.fence_finish = pvrgpu_fence_finish;
   ctx.base.screen = &screen.base;
   ctx.base.flush = pvrgpu_flush;
   ctx.framebuffer.width = 7;
   ctx.framebuffer.height = 11;
   ctx.framebuffer.nr_cbufs = 2;
   for (unsigned i = 0; i < 2; ++i)
      ctx.framebuffer.cbufs[i] = (struct pipe_surface){
         .texture = &resources[i], .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .level = i + 1, .first_layer = i, .last_layer = i + 2};
   ctx.framebuffer.zsbuf = (struct pipe_surface){
      .texture = &resources[2], .format = PIPE_FORMAT_Z24_UNORM_S8_UINT};
   const struct pipe_framebuffer_state framebuffer = ctx.framebuffer;
   ctx.query_statistics_failures = 5;
   ctx.array_primitive_draw_count = pending & 1 ? 2 : 0;
   ctx.color_readback_pending_mask = pending & 2 ?
      3u | (1u << 31) : 0;
   ctx.color_readback_generation = 41;
   completion_mode = mode == 3 ? 0 : mode;
   materializations = references = 0;
   const bool incomplete = mode == 3 && ctx.array_primitive_draw_count;
   expected_materialization = pending != 0 && !incomplete;
   CHECK(setenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS", mode == 3 ? "3" : "0", 1) == 0);
   old_fence = pvrgpu_failed_fence();
   struct pipe_fence_handle *fence = old_fence;
   ctx.base.flush(&ctx.base, request_fence ? &fence : NULL, flags);
   const bool failed = incomplete || (pending && (mode == 1 || mode == 2));
   CHECK(ctx.flushes == 1);
   CHECK(materializations == (unsigned)expected_materialization);
   CHECK(references == (unsigned)request_fence);
   CHECK(memcmp(&ctx.framebuffer, &framebuffer, sizeof(framebuffer)) == 0);
   CHECK(ctx.color_readback_generation == 41); /* No invented submission. */
   if (request_fence) {
      CHECK(fence == (failed ? pvrgpu_failed_fence() : NULL));
      CHECK(screen.base.fence_finish(&screen.base, &ctx.base, fence, 0) == !failed);
      CHECK(screen.base.fence_finish(&screen.base, &ctx.base, fence,
                                     UINT64_MAX) == !failed);
   } else {
      CHECK(fence == old_fence);
   }
   CHECK(ctx.query_statistics_failures == 5u + (unsigned)failed);
   if (!failed) {
      CHECK(ctx.array_primitive_draw_count == 0);
      CHECK(ctx.color_readback_pending_mask == 0);
   }
}

static void
test_fence_lifetime(void)
{
   struct pipe_fence_handle *failed = pvrgpu_failed_fence();
   struct pipe_fence_handle *copy = NULL;
   CHECK(failed != NULL && failed == pvrgpu_failed_fence());
   pvrgpu_fence_reference(NULL, &copy, failed);
   CHECK(copy == failed);
   pvrgpu_fence_reference(NULL, &failed, NULL);
   CHECK(failed == NULL && copy == pvrgpu_failed_fence());
   CHECK(!pvrgpu_fence_finish(NULL, NULL, copy, UINT64_MAX));
   unsigned char unknown;
   CHECK(!pvrgpu_fence_finish(NULL, NULL,
                              (struct pipe_fence_handle *)&unknown, 0));
   pvrgpu_fence_reference(NULL, &copy, NULL);
   CHECK(copy == NULL && pvrgpu_fence_finish(NULL, NULL, copy, 0));
   pvrgpu_fence_reference(NULL, NULL, pvrgpu_failed_fence());
   CHECK(!pvrgpu_fence_finish(NULL, NULL, pvrgpu_failed_fence(), 0));
}

static void
test_compute_dependency_boundary(void)
{
   /* Exercise the same guard launch_grid uses before reading indirect grids,
    * texture/image backing or any SSBO/UBO snapshot. The materializer remains
    * the only mocked boundary; this does not claim compute shader coverage. */
   for (unsigned pending = 0; pending < 4; ++pending)
   for (unsigned mode = 0; mode < 4; ++mode)
   for (unsigned emitted = 0; emitted < 4; ++emitted) {
      struct pvrgpu_context ctx = {0};
      ctx.query_statistics_failures = 9;
      ctx.array_primitive_draw_count = pending & 1 ? 2 : 0;
      ctx.color_readback_pending_mask = pending & 2 ? 3u | (1u << 31) : 0;
      ctx.driver_draw_command_emitted = (emitted & 1) != 0;
      global_emitted = (emitted & 2) != 0;
      completion_mode = mode;
      materializations = references = 0;
      CHECK(setenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS", "0", 1) == 0);
      const bool expected = mode != 1 && (mode != 2 || pending == 0) &&
                            (mode == 3 || emitted == 0);
      CHECK(pvrgpu_materialize_compute_inputs(&ctx) == expected);
      CHECK(materializations == 1);
      CHECK(ctx.query_statistics_failures == 9u + !expected);
   }
   struct pvrgpu_context ctx = {0};
   ctx.array_primitive_draw_count = 2;
   global_emitted = false;
   materializations = references = 0;
   CHECK(setenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS", "3", 1) == 0);
   CHECK(!pvrgpu_materialize_compute_inputs(&ctx));
   CHECK(materializations == 0 && ctx.query_statistics_failures == 1);
   unsetenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
}

int
main(void)
{
   const unsigned flags[] = {0, PIPE_FLUSH_DEFERRED,
      PIPE_FLUSH_ASYNC | PIPE_FLUSH_HINT_FINISH,
      PIPE_FLUSH_TOP_OF_PIPE, PIPE_FLUSH_BOTTOM_OF_PIPE,
      PIPE_FLUSH_END_OF_FRAME};
   test_fence_lifetime();
   test_compute_dependency_boundary();
   for (unsigned f = 0; f < ARRAY_SIZE(flags); ++f)
   for (unsigned pending = 0; pending < 4; ++pending)
   for (unsigned mode = 0; mode < 4; ++mode)
   for (unsigned fence = 0; fence < 2; ++fence)
      test_flush(flags[f], pending, mode, fence != 0);
   unsetenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
   /* A successful later flush must not retroactively signal a failed fence. */
   CHECK(!pvrgpu_fence_finish(NULL, NULL, pvrgpu_failed_fence(), 0));
   printf("synchronous flush/fence: %s (%u checks, %u failures)\n",
          failures ? "FAIL" : "PASS", checks, failures);
   return failures ? 1 : 0;
}
