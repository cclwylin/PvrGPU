/* SPDX-License-Identifier: MIT */
/* Run with script/run_mesa_resource_unit.sh compute. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_compute_snapshot.h"

#include <stdio.h>

static unsigned checks, failures;
#define CHECK(condition) do { \
   ++checks; \
   if (!(condition)) { \
      fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #condition); \
      ++failures; \
   } \
} while (0)

static void
init_buffer(struct pvrgpu_resource *resource, uint8_t *bytes, unsigned size)
{
   memset(resource, 0, sizeof(*resource));
   pipe_reference_init(&resource->base.reference, 1);
   resource->base.target = PIPE_BUFFER;
   resource->base.width0 = size;
   resource->data = bytes;
   resource->size = size;
}

static void
test_alias_and_writeback(void)
{
   uint8_t bytes[64], original[64];
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      original[i] = bytes[i] = (uint8_t)(17u * i + 3u);
   struct pvrgpu_resource resource;
   init_buffer(&resource, bytes, sizeof(bytes));
   struct pvrgpu_compute_snapshot snapshot = {0};
   const char *reason = NULL;
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER, 0, 1,
      &resource.base, NULL, 12, 24, &reason));
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 3, 3,
      &resource.base, NULL, 20, 12, &reason));
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 5, 2,
      &resource.base, NULL, 28, 12, &reason));
   CHECK(snapshot.resource_count == 1 && snapshot.binding_count == 3);
   CHECK(snapshot.bindings[0].resource_index == 0 &&
         snapshot.bindings[1].resource_index == 0 &&
         snapshot.bindings[2].resource_index == 0);
   CHECK(resource.base.reference.count == 2);
   CHECK(memcmp(snapshot.resources[0].bytes, original, sizeof(bytes)) == 0);
   bytes[0] = 0xab;
   CHECK(snapshot.resources[0].bytes[0] == original[0]);
   /* Emulate returned raw memory, including an invalid change outside views.
    * The driver must copy only writable ranges, never the whole BO. */
   memset(snapshot.resources[0].bytes, 0x71, sizeof(bytes));
   pvrgpu_compute_snapshot_writeback(&snapshot);
   CHECK(bytes[0] == 0xab);
   CHECK(memcmp(bytes + 1, original + 1, 19) == 0);
   for (unsigned i = 20; i < 40; ++i)
      CHECK(bytes[i] == 0x71);
   CHECK(memcmp(bytes + 40, original + 40, 24) == 0);
   pvrgpu_compute_snapshot_finish(&snapshot);
   CHECK(resource.base.reference.count == 1);
   CHECK(snapshot.resource_count == 0 && snapshot.binding_count == 0);
}

static void
test_abort_and_bound_ranges(void)
{
   uint8_t bytes[32];
   memset(bytes, 0x95, sizeof(bytes));
   struct pvrgpu_resource resource;
   init_buffer(&resource, bytes, sizeof(bytes));
   struct pvrgpu_compute_snapshot snapshot = {0};
   const char *reason = NULL;
   /* A binding is clamped to the real backing-store extent. */
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 31, 2,
      &resource.base, NULL, 24, 64, &reason));
   CHECK(snapshot.bindings[0].bytes_size == 8);
   memset(snapshot.resources[0].bytes, 0x44, sizeof(bytes));
   /* Failed dispatch drops snapshots without exposing partial model stores. */
   pvrgpu_compute_snapshot_finish(&snapshot);
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      CHECK(bytes[i] == 0x95);
   CHECK(resource.base.reference.count == 1);
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 32, 1,
      &resource.base, NULL, 0, 4, &reason));
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 1,
      &resource.base, NULL, 32, 4, &reason));
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 1,
      &resource.base, NULL, UINT64_MAX, 4, &reason));
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 1,
      &resource.base, NULL, 0, UINT64_MAX, &reason));
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 1,
      NULL, NULL, 0, 4, &reason));
   CHECK(snapshot.resource_count == 0 && snapshot.binding_count == 0);
   /* .length-only SSBO has metadata but no shader memory access. */
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 7, 0,
      &resource.base, NULL, 8, 12, &reason));
   CHECK(snapshot.bindings[0].access == 0);
   pvrgpu_compute_snapshot_finish(&snapshot);
}

static void
test_user_uniform_range_start(void)
{
   uint8_t bytes[16];
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      bytes[i] = (uint8_t)(37u * i);
   struct pvrgpu_compute_snapshot snapshot = {0};
   const char *reason = NULL;
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER, 14, 1,
      NULL, bytes + 4, 4096, 8, &reason));
   CHECK(snapshot.resources[0].bytes_size == 8);
   CHECK(snapshot.bindings[0].offset == 0);
   CHECK(memcmp(snapshot.resources[0].bytes, bytes + 4, 8) == 0);
   CHECK(snapshot.owners[0] == NULL);
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER, 15, 1,
      NULL, bytes, 0, 4, &reason));
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 1,
      NULL, bytes, 0, 4, &reason));
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER, 1, 3,
      NULL, bytes, 0, 4, &reason));
   pvrgpu_compute_snapshot_finish(&snapshot);
}

static void
test_aggregate_limit_before_allocation(void)
{
   uint8_t bytes[32] = {0};
   struct pvrgpu_resource resource;
   init_buffer(&resource, bytes, sizeof(bytes));
   struct pvrgpu_compute_snapshot snapshot = {0};
   const char *reason = NULL;
   /* Metadata-only existing entry: no giant allocation or read is needed to
    * prove that adding another BO is refused before malloc/memcpy/refcount. */
   snapshot.resource_count = 1;
   snapshot.resources[0].bytes_size = PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES;
   CHECK(!pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 1,
      &resource.base, NULL, 0, sizeof(bytes), &reason));
   CHECK(reason && strcmp(reason, "compute_snapshot_total_bytes") == 0);
   CHECK(snapshot.resource_count == 1 && snapshot.binding_count == 0);
   CHECK(resource.base.reference.count == 1);
   snapshot.resources[0].bytes_size -= sizeof(bytes);
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 1,
      &resource.base, NULL, 0, sizeof(bytes), &reason));
   CHECK(snapshot.resource_count == 2 && snapshot.binding_count == 1);
   /* At the aggregate cap, a second view of that BO is still legal. */
   CHECK(pvrgpu_compute_snapshot_add(&snapshot,
      PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 1, 2,
      &resource.base, NULL, 8, 8, &reason));
   CHECK(snapshot.resource_count == 2 && snapshot.binding_count == 2);
   CHECK(snapshot.bindings[0].resource_index ==
         snapshot.bindings[1].resource_index);
   pvrgpu_compute_snapshot_finish(&snapshot);
   CHECK(resource.base.reference.count == 1);
}

int main(void)
{
   test_alias_and_writeback();
   test_abort_and_bound_ranges();
   test_user_uniform_range_start();
   test_aggregate_limit_before_allocation();
   printf("compute snapshot checks=%u failures=%u\n", checks, failures);
   return failures ? 1 : 0;
}
