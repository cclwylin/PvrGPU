/* SPDX-License-Identifier: MIT */
#include "pvrgpu_context.h"
#include "pvrgpu_state.h"
#include "pipe/p_defines.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(bool condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "pvrgpu_stream_output_query_test: %s\n", message);
      exit(1);
   }
}

static void
test_instanced_topology(void)
{
   static const struct {
      enum mesa_prim mode;
      unsigned width, count;
      uint32_t elements[8];
   } cases[] = {
      {MESA_PRIM_LINE_STRIP, 2, 6, {0, 1, 1, 2, 2, 3}},
      {MESA_PRIM_LINE_LOOP, 2, 8, {0, 1, 1, 2, 2, 3, 3, 0}},
      {MESA_PRIM_TRIANGLE_STRIP, 3, 6, {0, 1, 2, 2, 1, 3}},
      {MESA_PRIM_TRIANGLE_FAN, 3, 6, {0, 1, 2, 0, 2, 3}},
   };
   const uint8_t indices8[] = {3, 1, 0, 2};
   const uint16_t indices16[] = {3, 1, 0, 2};
   const uint32_t indices32[] = {3, 1, 0, 2};
   const void *sources[] = {NULL, indices8, indices16, indices32};
   const unsigned sizes[] = {0, 1, 2, 4};
   for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
      for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
         uint32_t *indices = NULL;
         unsigned count = 0;
         enum mesa_prim mode = MESA_PRIM_POINTS;
         check(pvrgpu_expand_instanced_connected_indices(cases[c].mode,
            sources[s], sizes[s], 4, 4, 2, &indices, &count, &mode),
            "connected instancing expansion failed");
         check(count == cases[c].count * 2, "cross-instance primitive count");
         check(mode == (cases[c].width == 2 ? MESA_PRIM_LINES : MESA_PRIM_TRIANGLES),
            "expanded topology is not an independent list");
         for (unsigned i = 0; i < count; ++i) {
            unsigned element = cases[c].elements[i % cases[c].count];
            unsigned expected = (i / cases[c].count) * 4 +
               (s ? indices8[element] : element);
            check(indices[i] == expected, "strip winding/fan anchor/loop closure changed");
         }
         free(indices);
      }
   }

   uint32_t *indices = NULL;
   unsigned count = 0;
   enum mesa_prim mode = MESA_PRIM_POINTS;
   /* Even byte source indices may address above 255 after instance offset. */
   check(pvrgpu_expand_instanced_connected_indices(MESA_PRIM_LINE_STRIP,
      indices8, 1, 4, 256, 2, &indices, &count, &mode), "upcast expansion failed");
   check(indices[6] == 259 && indices[7] == 257, "instance index was truncated");
   free(indices);
   indices = NULL;
   check(!pvrgpu_expand_instanced_connected_indices(MESA_PRIM_LINE_STRIP,
      indices8, 1, 4, 3, 2, &indices, &count, &mode), "out-of-bounds source accepted");
   check(!pvrgpu_expand_instanced_connected_indices(MESA_PRIM_TRIANGLE_STRIP,
      NULL, 0, 4, UINT32_MAX, 2, &indices, &count, &mode), "instance extent overflow accepted");
   check(!pvrgpu_expand_instanced_connected_indices(MESA_PRIM_LINE_LOOP,
      NULL, 0, UINT32_MAX, UINT32_MAX, 1, &indices, &count, &mode), "index count overflow accepted");
   check(!pvrgpu_expand_instanced_connected_indices(MESA_PRIM_TRIANGLE_STRIP,
      indices8, 1, UINT32_MAX, 1, UINT32_MAX, &indices, &count, &mode), "multiply overflow accepted");
}

int main(void)
{
   test_instanced_topology();
   /* Force the real sequence submit entry to reject before touching the
    * recording payload. A failed draw must not become a successful zero
    * primitive query merely because it created no model generation. */
   unsetenv("PVRGPU_DRIVER_COMMAND_OUT");
   unsetenv("PVRGPU_SYSTEMC_JSONL_OUT");
   unsetenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
   unsetenv("PVRGPU_SYSTEMC_API_LIB");
   const unsigned types[] = {
      PIPE_QUERY_PRIMITIVES_GENERATED,
      PIPE_QUERY_PRIMITIVES_EMITTED,
      PIPE_QUERY_SO_STATISTICS,
   };
   for (unsigned i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
      struct pvrgpu_context ctx = {0};
      pvrgpu_init_state_functions(&ctx.base);
      struct pipe_query *query = ctx.base.create_query(&ctx.base, types[i], 0);
      check(query != NULL, "query allocation failed");
      check(ctx.base.begin_query(&ctx.base, query), "query begin failed");
      ctx.array_primitive_draw_count = 1;
      pvrgpu_context_end_frame_at_readback(&ctx);
      check(ctx.query_statistics_failures == 1, "rejected sequence did not invalidate native query statistics");
      check(ctx.array_primitive_draw_count == 0, "rejected recording was not retired");
      check(!ctx.base.end_query(&ctx.base, query), "rejected sequence completed a query");
      union pipe_query_result result = {0};
      check(!ctx.base.get_query_result(&ctx.base, query, false, &result), "rejected query reported ready");
      ctx.base.destroy_query(&ctx.base, query);

      /* A genuinely empty interval remains a successful query with zero. */
      query = ctx.base.create_query(&ctx.base, types[i], 0);
      check(query && ctx.base.begin_query(&ctx.base, query), "empty query begin failed");
      check(ctx.base.end_query(&ctx.base, query), "empty query end failed");
      check(ctx.base.get_query_result(&ctx.base, query, false, &result), "empty query was not ready");
      check(result.u64 == 0, "empty interval reported primitives");
      ctx.base.destroy_query(&ctx.base, query);
   }
   puts("pvrgpu_stream_output_query_test: PASS");
   return 0;
}
