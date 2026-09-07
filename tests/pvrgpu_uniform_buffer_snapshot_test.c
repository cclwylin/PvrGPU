/* SPDX-License-Identifier: MIT */
/* Run with script/run_mesa_resource_unit.sh ubo. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_uniform_buffer.h"
#include "../src/gallium/drivers/pvrgpu/pvrgpu_cmd.c"

static unsigned checks, failures;
#define CHECK(condition) do { \
   ++checks; \
   if (!(condition)) { \
      fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #condition); \
      ++failures; \
   } \
} while (0)

static void
test_draw_snapshot_lifetime(void)
{
   uint8_t source[96], original[96];
   for (unsigned i = 0; i < sizeof(source); ++i)
      original[i] = source[i] = (uint8_t)(i * 71u + 23u);
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_BUFFER;
   resource.data = source;
   resource.size = sizeof(source);
   struct pipe_constant_buffer bindings[16] = {0};
   bindings[0] = (struct pipe_constant_buffer){ .user_buffer = source,
                                               .buffer_size = 4 };
   bindings[1] = (struct pipe_constant_buffer){ .buffer = &resource.base,
                        .buffer_offset = 12, .buffer_size = 20 };
   bindings[3] = (struct pipe_constant_buffer){ .buffer = &resource.base,
                        .buffer_offset = 44, .buffer_size = 9 };
   /* This old higher binding must not leak into the active shader's 3 slots. */
   bindings[15] = bindings[1];
   struct pvrgpu_systemc_pco_uniform_buffer entries[30] = {0};
   unsigned count = 0;
   uint32_t vs[32] = {0}, fs[32] = {0};
   CHECK(pvrgpu_snapshot_stage_uniform_buffers(bindings, 0, 3, 0, vs, 32,
                                               entries, &count, 30));
   CHECK(pvrgpu_snapshot_stage_uniform_buffers(bindings, 1, 3, 20, fs, 32,
                                               entries, &count, 30));
   CHECK(count == 4);
   for (unsigned i = 0; i < count; ++i) {
      CHECK(entries[i].stage == i / 2);
      CHECK(entries[i].block_index == (i % 2 ? 2 : 0));
      CHECK(entries[i].bytes_size == (i % 2 ? 9 : 20));
      CHECK(memcmp(entries[i].bytes, original + (i % 2 ? 44 : 12),
                   entries[i].bytes_size) == 0);
   }
   CHECK(vs[2] == 20 && vs[6] == 0 && vs[10] == 9);
   CHECK(fs[22] == 20 && fs[26] == 0 && fs[30] == 9);
   struct pvrgpu_systemc_driver_command cmd = {0};
   cmd.vertex_pco_abi.shareds = cmd.fragment_pco_abi.shareds = 32;
   cmd.vertex_pco_abi.uniform_buffer_descriptor_count = 3;
   cmd.fragment_pco_abi.uniform_buffer_descriptor_start = 20;
   cmd.fragment_pco_abi.uniform_buffer_descriptor_count = 3;
   cmd.vertex_shared = vs; cmd.vertex_shared_count = 32;
   cmd.fragment_shared = fs; cmd.fragment_shared_count = 32;
   cmd.uniform_buffers = entries; cmd.uniform_buffer_count = count;
   char error[256];
   CHECK(pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));

   /* Mutation and rebinding produce a new draw, not changes to recorded input. */
   memset(source, 0xe7, sizeof(source));
   bindings[1].buffer_offset = 60;
   struct pvrgpu_systemc_pco_uniform_buffer next[30] = {0};
   unsigned next_count = 0;
   uint32_t next_shared[4] = {0};
   CHECK(pvrgpu_snapshot_stage_uniform_buffers(bindings, 0, 1, 0,
            next_shared, 4, next, &next_count, 30));
   CHECK(next_count == 1 && next[0].bytes[0] == 0xe7);
   memset(bindings, 0, sizeof(bindings));
   resource.data = NULL;
   for (unsigned i = 0; i < count; ++i)
      CHECK(memcmp(entries[i].bytes, original + (i % 2 ? 44 : 12),
                   entries[i].bytes_size) == 0);

   vs[0] = 1;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   vs[0] = 0; vs[3] = 4;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   vs[3] = 0; vs[6] = 16;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   vs[6] = 0; entries[1].block_index = 0;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   entries[1].block_index = 3;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   entries[1].block_index = 2; entries[1].stage = 4;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   entries[1].stage = 0;
   CHECK(pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   pvrgpu_finish_uniform_buffer_snapshots(entries, &count);
   pvrgpu_finish_uniform_buffer_snapshots(next, &next_count);
   CHECK(count == 0 && next_count == 0 && entries[0].bytes == NULL);
   pvrgpu_finish_uniform_buffer_snapshots(entries, &count);
}

static void
test_ranges_limits_and_raw_bytes(void)
{
   uint32_t raw[] = { UINT32_MAX, 0x80000000, 0x7fffffff, 0x7fc12345,
                      0x40000000, 0xc0800000, 0xdeadbeef, 0x12345678 };
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_BUFFER;
   resource.data = (uint8_t *)raw; resource.size = sizeof(raw);
   struct pipe_constant_buffer binding = { .buffer = &resource.base,
                        .buffer_offset = 4, .buffer_size = 100 };
   struct pvrgpu_systemc_pco_uniform_buffer entry;
   uint32_t descriptor[4];
   CHECK(pvrgpu_snapshot_uniform_buffer(&binding, 0, 0, &entry, descriptor));
   CHECK(entry.bytes_size == sizeof(raw) - 4 && descriptor[2] == sizeof(raw) - 4);
   CHECK(memcmp(entry.bytes, (uint8_t *)raw + 4, entry.bytes_size) == 0);
   free((void *)entry.bytes);
   binding.buffer_offset = sizeof(raw);
   CHECK(!pvrgpu_snapshot_uniform_buffer(&binding, 0, 0, &entry, descriptor));
   binding.buffer_offset = 0; resource.base.target = PIPE_TEXTURE_2D;
   CHECK(!pvrgpu_snapshot_uniform_buffer(&binding, 0, 0, &entry, descriptor));
   binding = (struct pipe_constant_buffer){ .user_buffer = raw,
                  .buffer_size = PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES + 1 };
   CHECK(!pvrgpu_snapshot_uniform_buffer(&binding, 0, 0, &entry, descriptor));
   binding.buffer_size = sizeof(raw);
   binding.buffer_offset = 12; /* user_buffer is already the range start. */
   CHECK(pvrgpu_snapshot_uniform_buffer(&binding, 1, 14, &entry, descriptor));
   CHECK(memcmp(entry.bytes, raw, sizeof(raw)) == 0);
   free((void *)entry.bytes);
   CHECK(!pvrgpu_snapshot_uniform_buffer(&binding, 1, 15, &entry, descriptor));
   CHECK(!pvrgpu_snapshot_uniform_buffer(&binding, 4, 0, &entry, descriptor));
   binding = (struct pipe_constant_buffer){0};
   CHECK(pvrgpu_snapshot_uniform_buffer(&binding, 0, 0, &entry, descriptor));
   CHECK(!entry.bytes && !entry.bytes_size && !descriptor[0] &&
         !descriptor[1] && !descriptor[2] && !descriptor[3]);

   struct pipe_constant_buffer bindings[16] = {0};
   for (unsigned i = 1; i < 16; ++i)
      bindings[i] = (struct pipe_constant_buffer){ .user_buffer = raw,
                                                   .buffer_size = sizeof(raw) };
   struct pvrgpu_systemc_pco_uniform_buffer entries[30] = {0};
   unsigned count = 0;
   uint32_t vs[96] = {0}, fs[256] = {0};
   CHECK(pvrgpu_snapshot_stage_uniform_buffers(bindings, 0, 15, 0, vs, 96,
                                               entries, &count, 30));
   CHECK(pvrgpu_snapshot_stage_uniform_buffers(bindings, 1, 15, 20, fs, 256,
                                               entries, &count, 30));
   CHECK(count == 30);
   struct pvrgpu_draw_pco_triangles_command driver = {0};
   driver.vertex_shared = vs; driver.vertex_shared_count = 96;
   driver.fragment_shared = fs; driver.fragment_shared_count = 256;
   driver.vertex_pco_abi.shareds = 96;
   driver.vertex_pco_abi.uniform_buffer_descriptor_count = 15;
   driver.vertex_pco_abi.push_constant_start = 60;
   driver.vertex_pco_abi.push_constant_count = 36;
   driver.fragment_pco_abi.shareds = 256;
   driver.fragment_pco_abi.uniform_buffer_descriptor_start = 20;
   driver.fragment_pco_abi.uniform_buffer_descriptor_count = 15;
   driver.fragment_pco_abi.push_constant_start = 80;
   driver.fragment_pco_abi.push_constant_count = 176;
   driver.uniform_buffers = entries; driver.uniform_buffer_count = count;
   struct pvrgpu_systemc_driver_command cmd;
   pvrgpu_pco_triangles_command_to_systemc(&driver, &cmd);
   CHECK(cmd.uniform_buffers == entries && cmd.uniform_buffer_count == 30);
   CHECK(cmd.fragment_pco_abi.uniform_buffer_descriptor_start == 20);
   CHECK(cmd.fragment_pco_abi.uniform_buffer_descriptor_count == 15);
   char error[256];
   CHECK(pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   cmd.vertex_pco_abi.shareds = cmd.vertex_shared_count = 97;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   cmd.vertex_pco_abi.shareds = cmd.vertex_shared_count = 96;
   cmd.fragment_pco_abi.push_constant_start = 79;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   cmd.fragment_pco_abi.push_constant_start = 80;
   cmd.uniform_buffer_count = 31;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   cmd.uniform_buffer_count = 30;
   entries[29].bytes_size = 0;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   pvrgpu_finish_uniform_buffer_snapshots(entries, &count);

   /* A failed later range still leaves earlier owned allocations retireable. */
   bindings[2].buffer_size = PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES + 1;
   CHECK(!pvrgpu_snapshot_stage_uniform_buffers(bindings, 0, 2, 0, vs, 96,
                                                entries, &count, 30));
   CHECK(count == 1);
   pvrgpu_finish_uniform_buffer_snapshots(entries, &count);
   CHECK(count == 0);
}

int main(void)
{
   test_draw_snapshot_lifetime();
   test_ranges_limits_and_raw_bytes();
   printf("UBO snapshot/descriptor tests: %u checks, %u failures\n", checks, failures);
   return failures != 0;
}
