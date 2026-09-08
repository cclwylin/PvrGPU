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

static void
test_image_views_and_padding(void)
{
   uint8_t bytes[160], original[160];
   for (unsigned i = 0; i < sizeof(bytes); ++i) bytes[i] = original[i] = i * 13U + 7;
   struct pvrgpu_resource resource;
   init_buffer(&resource, bytes, sizeof(bytes));
   resource.base.target = PIPE_TEXTURE_2D;
   resource.base.format = PIPE_FORMAT_R32_UINT;
   resource.base.width0 = 5; resource.base.height0 = 3;
   resource.base.depth0 = resource.base.array_size = 1;
   resource.base.last_level = 1;
   resource.level_count = 2;
   resource.level_offsets[0] = 16; resource.level_strides[0] = 32;
   resource.level_layer_strides[0] = 96;
   resource.level_offsets[1] = 128; resource.level_strides[1] = 16;
   resource.level_layer_strides[1] = 16;
   struct pipe_image_view view = {.resource = &resource.base,
      .format = PIPE_FORMAT_R32_UINT, .access = PIPE_IMAGE_ACCESS_READ_WRITE};
   struct pvrgpu_compute_snapshot snapshot = {0};
   const char *reason = NULL;
   CHECK(pvrgpu_compute_snapshot_add_image(&snapshot, 1, 3, &view, &reason));
   view.u.tex.level = 1;
   CHECK(pvrgpu_compute_snapshot_add_image(&snapshot, 3, 2, &view, &reason));
   CHECK(pvrgpu_compute_snapshot_add_image(&snapshot, 0, 1, &view, &reason));
   CHECK(snapshot.resource_count == 1 && snapshot.image_count == 3 &&
         snapshot.binding_count == 0 && resource.base.reference.count == 2);
   CHECK(snapshot.images[0].offset == 16 && snapshot.images[0].width == 5 &&
         snapshot.images[0].height == 3 && snapshot.images[0].row_stride_bytes == 32 &&
         snapshot.images[0].bytes_size == 84);
   CHECK(snapshot.images[1].offset == 128 && snapshot.images[1].width == 2 &&
         snapshot.images[1].height == 1 && snapshot.images[1].resource_index == 0);
   memset(snapshot.resources[0].bytes, 0x6a, sizeof(bytes));
   pvrgpu_compute_snapshot_writeback(&snapshot);
   for (unsigned i = 0; i < sizeof(bytes); ++i) {
      const bool written = (i >= 16 && i < 36) || (i >= 48 && i < 68) ||
                           (i >= 80 && i < 100) || (i >= 128 && i < 136);
      CHECK(bytes[i] == (written ? 0x6a : original[i]));
   }
   CHECK(resource.driver_writes_model_cannot_reproduce);
   pvrgpu_compute_snapshot_finish(&snapshot);
   CHECK(resource.base.reference.count == 1);
   resource.driver_writes_model_cannot_reproduce = false;
   const struct pvrgpu_resource valid = resource;
   const struct pipe_image_view valid_view = view;
   for (unsigned bad = 0; bad < 10; ++bad) {
      resource = valid; view = valid_view;
      if (bad == 0) view.access = PIPE_IMAGE_ACCESS_READ;
      if (bad == 1) view.format = PIPE_FORMAT_R32_SINT;
      if (bad == 2) view.u.tex.level = 2;
      if (bad == 3) view.u.tex.first_layer = 1;
      if (bad == 4) resource.level_strides[1] = 7;
      if (bad == 5) resource.level_offsets[1] = 156;
      if (bad == 6) resource.level_offsets[1] = 127;
      if (bad == 7) resource.level_layer_strides[1] = 4;
      if (bad == 8) resource.base.nr_samples = 4;
      if (bad == 9) resource.base.width0 = 0;
      CHECK(!pvrgpu_compute_snapshot_add_image(&snapshot, 0, 2, &view, &reason));
      CHECK(snapshot.resource_count == 0 && snapshot.image_count == 0 &&
            resource.base.reference.count == 1 && !resource.driver_writes_model_cannot_reproduce);
   }
}

int main(void)
{
   test_alias_and_writeback();
   test_abort_and_bound_ranges();
   test_user_uniform_range_start();
   test_aggregate_limit_before_allocation();
   test_image_views_and_padding();
   printf("compute snapshot checks=%u failures=%u\n", checks, failures);
   return failures ? 1 : 0;
}
