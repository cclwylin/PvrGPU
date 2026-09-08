/* SPDX-License-Identifier: MIT */
/* Pure production accounting/retry-boundary helpers. No model execution or
 * framebuffer result is claimed. Run run_mesa_resource_unit.sh payload-budget. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_payload_budget.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); } } while (0)

static void
test_all_owned_fields(void)
{
   uint8_t byte = 0;
   struct pvrgpu_systemc_driver_command draw = {0};
   struct pvrgpu_systemc_tessellation tess = {0};
   struct pvrgpu_systemc_pco_sequence_texture textures[2] = {{0}};
   struct pvrgpu_systemc_pco_uniform_buffer ubo[5] = {{0}};
   struct pvrgpu_systemc_shader_image images[2] = {{0}};
   struct pvrgpu_systemc_stream_output_target targets[2] = {{0}};
   struct pvrgpu_systemc_stream_output so = {0};
   uint64_t bytes = UINT64_MAX;
   CHECK(pvrgpu_pco_draw_payload_bytes(&draw, NULL, 0, &bytes) && bytes == 0);
   draw.raw_vertex_data_size = 11;
   draw.vertex_pco_size = 13;
   draw.fragment_pco_size = 17;
   draw.geometry_pco_size = 19;
   draw.initial_color_attachment_bytes_size = 23; /* Already all MRT/samples/layers. */
   draw.initial_depth_attachment_bytes_size = 29;
   draw.vertex_shared_count = 31;
   draw.fragment_shared_count = 37;
   draw.geometry_shared_count = 41;
   tess.control_pco_size = 43;
   tess.evaluation_pco_size = 47;
   tess.control_shared_count = 53;
   tess.evaluation_shared_count = 59;
   draw.tessellation = &tess;
   textures[0].source = textures[1].source = PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD;
   textures[0].bytes = textures[1].bytes = &byte; /* Equal pointer does not deduplicate copies. */
   textures[0].bytes_size = 61;
   textures[1].bytes_size = 67;
   draw.sampled_texture_count = 2;
   for (unsigned i = 0; i < 5; ++i) {
      ubo[i].stage = i;
      ubo[i].bytes = &byte;
      ubo[i].bytes_size = 71 + i;
   }
   draw.uniform_buffers = ubo;
   draw.uniform_buffer_count = 5;
   images[0].bytes = images[1].bytes = &byte;
   images[0].bytes_size = 79;
   images[1].bytes_size = 83;
   draw.fragment_images = images;
   draw.fragment_image_count = 2;
   targets[0].bytes = targets[1].bytes = &byte;
   targets[0].bytes_size = 89;
   targets[1].bytes_size = 97;
   so.targets = targets; so.target_count = 2;
   draw.stream_output = &so;
   const uint64_t expected = 11+13+17+19+23+29+43+47 +
      4*(31+37+41+53+59) + 61+67 + 71+72+73+74+75 + 79+83 + 89+97;
   CHECK(pvrgpu_pco_draw_payload_bytes(&draw, textures, 2, &bytes) && bytes == expected);
   /* Attachment aliases have metadata extent, but no owned input byte vector. */
   textures[1].source = PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT;
   textures[1].declared_bytes_size = 1u << 20;
   CHECK(pvrgpu_pco_draw_payload_bytes(&draw, textures, 2, &bytes) && bytes == expected - 67);
   draw.raw_index_data_size = 101;
   draw.attachment_clear_count = 3;
   draw.varying_binding_count = 4;
   CHECK(pvrgpu_pco_draw_payload_bytes(&draw, textures, 2, &bytes) && bytes == expected - 67);
   /* Legacy slot-zero bytes, when present, are a separate owned copy. */
   draw.sampled_texture_count = 1;
   draw.sampled_texture_bytes = &byte;
   draw.sampled_texture_bytes_size = 103;
   CHECK(pvrgpu_pco_draw_payload_bytes(&draw, textures, 1, &bytes) && bytes == expected - 67 + 103);
   draw.geometry_pco_size = 0;
   CHECK(pvrgpu_pco_draw_payload_bytes(&draw, textures, 1, &bytes) && bytes == expected - 67 + 103 - 19 - 4*41);
   draw.uniform_buffers = NULL;
   CHECK(!pvrgpu_pco_draw_payload_bytes(&draw, textures, 1, &bytes));
   draw.uniform_buffers = ubo; draw.uniform_buffer_count = UINT32_MAX;
   CHECK(!pvrgpu_pco_draw_payload_bytes(&draw, textures, 1, &bytes));
   draw.uniform_buffer_count = 5; draw.fragment_images = NULL;
   CHECK(!pvrgpu_pco_draw_payload_bytes(&draw, textures, 1, &bytes));
   draw.fragment_images = images; so.targets = NULL;
   CHECK(!pvrgpu_pco_draw_payload_bytes(&draw, textures, 1, &bytes));
   CHECK(!pvrgpu_pco_draw_payload_bytes(NULL, NULL, 0, &bytes));
   CHECK(!pvrgpu_pco_draw_payload_bytes(&draw, textures, UINT32_MAX, &bytes));
}

static void
test_overflow_and_retry_boundaries(void)
{
   uint64_t bytes = UINT64_MAX - 3;
   CHECK(!pvrgpu_payload_add(&bytes, 4) && bytes == UINT64_MAX - 3);
   CHECK(pvrgpu_payload_add(&bytes, 3) && bytes == UINT64_MAX);
   bytes = 0;
   CHECK(!pvrgpu_payload_add_dwords(&bytes, UINT64_MAX / 4 + 1) && bytes == 0);
   struct pvrgpu_systemc_driver_command draw = {0};
   draw.raw_vertex_data_size = SIZE_MAX;
   draw.vertex_pco_size = 1;
   if (SIZE_MAX == UINT64_MAX)
      CHECK(!pvrgpu_pco_draw_payload_bytes(&draw, NULL, 0, &bytes));
   const uint64_t limit = PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_PAYLOAD_BYTES;
   CHECK(limit == UINT64_C(536870912));
   CHECK(pvrgpu_payload_decide(0, limit, 0, true, limit) == PVRGPU_PAYLOAD_FITS);
   CHECK(pvrgpu_payload_decide(limit-1, 1, 2, true, limit) == PVRGPU_PAYLOAD_FITS);
   CHECK(pvrgpu_payload_decide(limit-1, 2, 2, true, limit) == PVRGPU_PAYLOAD_FLUSH);
   CHECK(pvrgpu_payload_decide(limit-1, 2, 2, false, limit) == PVRGPU_PAYLOAD_REJECT);
   CHECK(pvrgpu_payload_decide(0, limit+1, 0, true, limit) == PVRGPU_PAYLOAD_REJECT);
   CHECK(pvrgpu_payload_decide(1, limit+1, 1, true, limit) == PVRGPU_PAYLOAD_REJECT);
   CHECK(pvrgpu_payload_decide(UINT64_MAX, 1, 1, true, limit) == PVRGPU_PAYLOAD_REJECT);
   /* The fresh first draw includes all attachment LOAD bytes; checking only
    * its pre-flush estimate would incorrectly accept this oversized retry. */
   CHECK(pvrgpu_payload_decide(60, 64, 2, true, 100) == PVRGPU_PAYLOAD_FLUSH);
   CHECK(pvrgpu_payload_decide(0, 64+50, 0, false, 100) == PVRGPU_PAYLOAD_REJECT);
   CHECK(pvrgpu_payload_decide(0, 64+20, 0, false, 100) == PVRGPU_PAYLOAD_FITS);
   /* A cleared draw count alone is not proof: readback can fail after reset,
    * pending attachment bytes or replay ownership can also prevent restart. */
   for (unsigned state = 0; state < 32; ++state)
      CHECK(pvrgpu_payload_flush_completed(7, 7 + ((state & 1) != 0),
         (state & 2) != 0, (state & 4) != 0,
         (state & 8) != 0, (state & 16) != 0) == (state == 0));
}

int main(void)
{
   test_all_owned_fields();
   test_overflow_and_retry_boundaries();
   printf("PCO payload accounting/retry boundary: %s (%u checks, %u failures)\n",
          failures ? "FAIL" : "PASS", checks, failures);
   return failures ? 1 : 0;
}
