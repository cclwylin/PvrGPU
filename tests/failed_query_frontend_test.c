/* SPDX-License-Identifier: MIT */
/* Compile with pinned Mesa's queryobj.c flags and link its ordinary Mesa
 * archives. Including the actual frontend implementation exposes its static
 * lifecycle helpers; only the Gallium query backend is fault-injected. */
#include "main/queryobj.c"
#include <stdio.h>

static unsigned destroyed, reads, checks;
static bool fail_end;

static void
require(bool condition, const char *message)
{
   ++checks;
   if (!condition) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}

static bool
test_end(struct pipe_context *pipe, struct pipe_query *query)
{
   (void)pipe; (void)query;
   return !fail_end;
}

static void
test_destroy(struct pipe_context *pipe, struct pipe_query *query)
{
   (void)pipe;
   ++destroyed;
   free(query);
}

static bool
test_result(struct pipe_context *pipe, struct pipe_query *query, bool wait,
            union pipe_query_result *result)
{
   (void)pipe; (void)query; (void)wait;
   ++reads;
   result->u64 = 0;
   return true;
}

int
main(void)
{
   struct gl_context *ctx = calloc(1, sizeof(*ctx));
   struct st_context *st = calloc(1, sizeof(*st));
   struct pipe_context pipe = {0};
   struct pipe_screen screen = {0};
   require(ctx && st, "test allocation");
   ctx->API = API_OPENGLES2;
   ctx->Version = 30;
   ctx->pipe = &pipe;
   ctx->st = st;
   st->ctx = ctx;
   st->pipe = &pipe;
   st->bitmap.cache.empty = true;
   st->screen = pipe.screen = &screen;
   pipe.end_query = test_end;
   pipe.destroy_query = test_destroy;
   pipe.get_query_result = test_result;
   _mesa_init_queryobj(ctx);
   const unsigned types[] = {PIPE_QUERY_PRIMITIVES_EMITTED, PIPE_QUERY_TIMESTAMP};
   for (unsigned i = 0; i < 2; ++i) {
      struct gl_query_object *q = new_query_object(ctx, i + 1);
      require(q != NULL, "query allocation");
      q->Target = i ? GL_TIME_ELAPSED : GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN;
      q->type = types[i];
      q->EverBound = GL_TRUE;
      q->Ready = GL_FALSE;
      q->Result = UINT64_C(0x123456789abcdef0);
      q->pq = malloc(1);
      if (i) q->pq_begin = malloc(1);
      st->active_queries = i ? 0 : 1;
      _mesa_HashInsertLocked(&ctx->Query.QueryObjects, q->Id, q);
      ctx->ErrorValue = GL_NO_ERROR;
      fail_end = true;
      end_query(ctx, q);
      require(ctx->ErrorValue == GL_OUT_OF_MEMORY, "EndQuery error preserved");
      require(q->Failed && !q->Ready, "failed query became ready");
      require(!q->pq && !q->pq_begin, "failed pipe queries were not retired");
      require(st->active_queries == 0, "active_queries not balanced");
      require(q->Result == UINT64_C(0x123456789abcdef0), "failure fabricated query result");
      for (unsigned mode = 0; mode < 2; ++mode) {
         GLuint result = 0xfeedcafe;
         ctx->ErrorValue = GL_NO_ERROR;
         get_query_object(ctx, "testGetQueryObject", q->Id,
            mode ? GL_QUERY_RESULT_AVAILABLE : GL_QUERY_RESULT,
            GL_UNSIGNED_INT, NULL, (intptr_t)&result);
         require(ctx->ErrorValue == GL_OUT_OF_MEMORY && result == 0xfeedcafe,
            "failed query wrote user result or lost error");
      }
      ctx->ErrorValue = GL_NO_ERROR;
      _mesa_wait_query(ctx, q);
      require(ctx->ErrorValue == GL_OUT_OF_MEMORY && !q->Ready && reads == 0,
         "failed wait polled driver or reported readiness");
      ctx->ErrorValue = GL_NO_ERROR;
      _mesa_check_query(ctx, q);
      require(ctx->ErrorValue == GL_OUT_OF_MEMORY && !q->Ready && reads == 0,
         "failed availability polled driver");
      _mesa_HashRemoveLocked(&ctx->Query.QueryObjects, q->Id);
      delete_query(ctx, q);
      require(destroyed == (i ? 3 : 1), "query lifetime double-free or leak");
   }
   _mesa_free_queryobj_data(ctx);
   free(st);
   free(ctx);
   printf("PASS: failed query frontend %u checks, %u pipe queries destroyed, %u result polls\n",
      checks, destroyed, reads);
   return 0;
}
