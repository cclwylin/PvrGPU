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
   CHECK(!pvrgpu_snapshot_uniform_buffer(&binding, 5, 0, &entry, descriptor));
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
   /* Five graphics stages are now admitted. Exceed the actual global cap;
    * count31 is legal at the pointer/count boundary and would overread our
    * 30-entry allocation before a later semantic rejection. */
   cmd.uniform_buffer_count = 5 * PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE + 1;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   /* A count within the global cap must have real backing storage even for
    * a deliberately invalid entry. Reject its stage without an OOB read. */
   struct pvrgpu_systemc_pco_uniform_buffer invalid_entries[31] = {0};
   memcpy(invalid_entries, entries, sizeof(entries));
   invalid_entries[30].stage = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION + 1;
   cmd.uniform_buffers = invalid_entries;
   cmd.uniform_buffer_count = 31;
   CHECK(!pvrgpu_cmd_validate_uniform_buffers(&cmd, error, sizeof(error)));
   cmd.uniform_buffers = entries;
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

static void
test_geometry_sampler_descriptor_prefix(void)
{
   /* Producer envelope only: all calls stop at a named later attribute
    * sentinel. They never submit these placeholder executable bytes. */
   uint8_t code=1;
   uint32_t shared[256]={0};
   struct pvrgpu_draw_pco_triangles_command cmd={0};
   cmd.case_name="native.geometry.sampler.abi";cmd.frame=1;
   cmd.framebuffer_width=cmd.framebuffer_height=cmd.width=cmd.height=16;
   cmd.format=PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   cmd.geometry_pco=&code;cmd.geometry_pco_size=1;
   cmd.geometry_shared=shared;cmd.geometry_input_primitive_vertices=1;
   cmd.geometry_input_stride_dwords=cmd.vertex_pco_abi.vertex_outputs=4;
   cmd.geometry_max_vertices=cmd.geometry_invocations=cmd.geometry_vertices_per_instance=1;
   cmd.geometry_pco_abi.vertex_inputs=2;cmd.geometry_pco_abi.vertex_outputs=4;
   cmd.vertex_attribute_count=17;
   char error[256];
   for(unsigned textures=0;textures<=8;++textures) {
      for(unsigned ubos=0;ubos<=15;++ubos) {
         cmd.geometry_pco_abi.uniform_buffer_descriptor_start=4+20*textures;
         cmd.geometry_pco_abi.uniform_buffer_descriptor_count=ubos;
         cmd.geometry_pco_abi.push_constant_start=4+20*textures+4*ubos;
         cmd.geometry_pco_abi.push_constant_count=4;
         cmd.geometry_shared_count=cmd.geometry_pco_abi.shareds=8+20*textures+4*ubos;
         CHECK(!pvrgpu_validate_draw_pco_triangles_command("unused",&cmd,error,sizeof(error)));
         CHECK(strstr(error,"vertex attribute count")!=NULL);
         struct pvrgpu_systemc_driver_command api;
         pvrgpu_pco_triangles_command_to_systemc(&cmd,&api);
         CHECK(pvrgpu_cmd_validate_uniform_buffers(&api,error,sizeof(error)));
      }
   }
   for (unsigned i = 0; i < 2; ++i) {
      const unsigned vertex_inputs = i ? 64 : 3;
      cmd.geometry_pco_abi.vertex_inputs = vertex_inputs;
      CHECK(!pvrgpu_validate_draw_pco_triangles_command("unused", &cmd,
                                                        error, sizeof(error)));
      CHECK(strstr(error, "vertex attribute count") != NULL);
   }
   const unsigned invalid_vertex_inputs[] = {0, 1, 65, UINT32_MAX};
   for (unsigned i = 0;
        i < sizeof(invalid_vertex_inputs) / sizeof(invalid_vertex_inputs[0]);
        ++i) {
      cmd.geometry_pco_abi.vertex_inputs = invalid_vertex_inputs[i];
      CHECK(!pvrgpu_validate_draw_pco_triangles_command("unused", &cmd,
                                                        error, sizeof(error)));
      CHECK(strstr(error, "independent geometry program ABI") != NULL);
   }
   cmd.geometry_pco_abi.vertex_inputs = 2;
   const unsigned invalid[]={0,3,5,23,25,165,184,UINT32_MAX};
   for(unsigned i=0;i<sizeof(invalid)/sizeof(invalid[0]);++i) {
      cmd.geometry_pco_abi.uniform_buffer_descriptor_start=invalid[i];
      CHECK(!pvrgpu_validate_draw_pco_triangles_command("unused",&cmd,error,sizeof(error)));
      CHECK(strstr(error,"independent geometry program ABI")!=NULL);
      struct pvrgpu_systemc_driver_command api;
      pvrgpu_pco_triangles_command_to_systemc(&cmd,&api);
      CHECK(!pvrgpu_cmd_validate_uniform_buffers(&api,error,sizeof(error)));
   }
   cmd.geometry_pco_abi.uniform_buffer_descriptor_start=164;
   cmd.geometry_pco_abi.uniform_buffer_descriptor_count=16;
   CHECK(!pvrgpu_validate_draw_pco_triangles_command("unused",&cmd,error,sizeof(error)));
   CHECK(strstr(error,"independent geometry program ABI")!=NULL);
}

static void
test_tessellation_sampler_descriptor_prefix(void)
{
   /* Structural envelopes only, not placeholder-code execution. The live
    * tessellation texture test separately exercises actual compiler bytes. */
   const uint8_t code = 1;
   uint32_t banks[2][256] = {{0}};
   struct pvrgpu_systemc_tessellation tess = {0};
   tess.control_pco = tess.evaluation_pco = &code;
   tess.control_pco_size = tess.evaluation_pco_size = 1;
   tess.control_shared = banks[0]; tess.evaluation_shared = banks[1];
   tess.input_vertices = tess.output_vertices = tess.vertices_per_instance = 3;
   tess.input_stride_dwords = tess.output_vertex_stride_dwords = 4;
   tess.per_vertex_offset_dwords = 6; tess.patch_stride_dwords = 18;
   tess.control_abi.vertex_inputs = 3;
   tess.evaluation_abi.vertex_inputs = 5; tess.evaluation_abi.vertex_outputs = 4;
   struct pvrgpu_systemc_driver_command api = {0};
   api.tessellation = &tess;
   char error[256];
   for (unsigned textures = 0; textures <= 8; ++textures) {
      for (unsigned ubos = 0; ubos <= 15; ++ubos) {
         for (unsigned stage = 0; stage < 2; ++stage) {
            struct pvrgpu_systemc_pco_stage_abi *abi = stage ?
               &tess.evaluation_abi : &tess.control_abi;
            const unsigned prefix = stage ? 4 : 8;
            abi->uniform_buffer_descriptor_start = prefix + 20 * textures;
            abi->uniform_buffer_descriptor_count = ubos;
            abi->push_constant_start = abi->uniform_buffer_descriptor_start + 4 * ubos;
            abi->push_constant_count = 4;
            abi->shareds = abi->push_constant_start + 4;
            if (stage) tess.evaluation_shared_count = abi->shareds;
            else tess.control_shared_count = abi->shareds;
         }
         CHECK(pvrgpu_tessellation_payload_error(&tess) == NULL);
         CHECK(pvrgpu_cmd_validate_uniform_buffers(&api, error, sizeof(error)));
         for (unsigned stage = 0; stage < 2; ++stage) {
            struct pvrgpu_systemc_pco_stage_abi *abi = stage ?
               &tess.evaluation_abi : &tess.control_abi;
            const unsigned prefix = stage ? 4 : 8;
            const unsigned valid_start = abi->uniform_buffer_descriptor_start;
            const unsigned invalid[] = {0, prefix - 1, prefix + 4,
               prefix + 19, prefix + 161, prefix + 180, UINT32_MAX};
            for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
               abi->uniform_buffer_descriptor_start = invalid[i];
               CHECK(pvrgpu_tessellation_payload_error(&tess) != NULL);
               CHECK(!pvrgpu_cmd_validate_uniform_buffers(&api, error, sizeof(error)));
            }
            abi->uniform_buffer_descriptor_start = valid_start;
            banks[stage][prefix - 1] = 1;
            CHECK(pvrgpu_tessellation_payload_error(&tess) != NULL);
            banks[stage][prefix - 1] = 0;
            abi->push_constant_start += 4;
            CHECK(pvrgpu_tessellation_payload_error(&tess) != NULL);
            abi->push_constant_start -= 4;
         }
      }
   }
}

static void
test_compiled_ubo_prefix(void)
{
   for (unsigned declared = 0; declared <= 17; ++declared) {
      for (unsigned compiled = 0; compiled <= 17; ++compiled) {
         CHECK(pvrgpu_uniform_buffer_prefix_count_valid(declared, compiled, false) ==
                  (declared <= 15 && compiled == declared));
         CHECK(pvrgpu_uniform_buffer_prefix_count_valid(declared, compiled, true) ==
                  (declared <= 15 && compiled <= declared));
      }
   }
   CHECK(!pvrgpu_uniform_buffer_prefix_count_valid(UINT32_MAX, 0, true));
   CHECK(!pvrgpu_uniform_buffer_prefix_count_valid(1, UINT32_MAX, true));
   /* An unused suffix is not read or copied, even if stale higher bindings
    * name invalid storage. Original block0 still comes from CB1. */
   uint32_t data[] = {0xdeadbeef, 0x7fc12345, 0x80000000, 0x12345678};
   struct pipe_constant_buffer bindings[16] = {0};
   bindings[1].user_buffer = data;
   bindings[1].buffer_size = sizeof(data);
   struct pvrgpu_resource invalid = {0};
   invalid.base.target = PIPE_TEXTURE_2D;
   bindings[2].buffer = &invalid.base;
   bindings[2].buffer_size = 4;
   uint32_t words[12];
   for (unsigned i = 0; i < 12; ++i) words[i] = 0xbabef00d;
   struct pvrgpu_systemc_pco_uniform_buffer entries[15] = {0};
   unsigned count = 0;
   CHECK(pvrgpu_snapshot_stage_uniform_buffers(bindings, 0, 0, 0, words, 12,
                                                entries, &count, 15));
   CHECK(count == 0);
   for (unsigned i = 0; i < 12; ++i) CHECK(words[i] == 0xbabef00d);
   CHECK(pvrgpu_snapshot_stage_uniform_buffers(bindings, 0, 1, 0, words, 12,
                                                entries, &count, 15));
   CHECK(count == 1 && entries[0].block_index == 0 && entries[0].stage == 0);
   CHECK(entries[0].bytes_size == sizeof(data));
   CHECK(memcmp(entries[0].bytes, data, sizeof(data)) == 0);
   CHECK(words[0] == 0 && words[1] == 0 && words[2] == sizeof(data) && words[3] == 0);
   for (unsigned i = 4; i < 12; ++i) CHECK(words[i] == 0xbabef00d);
   pvrgpu_finish_uniform_buffer_snapshots(entries, &count);
   CHECK(count == 0);

   /* The compiler may append CB0 after the retained real-UBO prefix.  Native
    * block0 still snapshots Gallium CB1; native block1 deliberately aliases
    * Gallium CB0, and both payloads keep their native descriptor indices. */
   uint32_t cb0[] = {0x10203040, 0x7fc00001, 0x80000000, 0xffffffff};
   bindings[0].user_buffer = cb0;
   bindings[0].buffer_size = sizeof(cb0) - 3;
   for (unsigned i = 0; i < 12; ++i) words[i] = 0xbabef00d;
   CHECK(pvrgpu_snapshot_stage_uniform_buffers_mapped(
      bindings, 0, 2, 0, 2, 4, words, 12, entries, &count, 15));
   CHECK(count == 2);
   CHECK(entries[0].block_index == 0 && entries[0].bytes_size == sizeof(data));
   CHECK(entries[1].block_index == 1 && entries[1].bytes_size == sizeof(cb0));
   CHECK(memcmp(entries[0].bytes, data, sizeof(data)) == 0);
   CHECK(memcmp(entries[1].bytes, cb0, 3 * sizeof(uint32_t)) == 0);
   CHECK(((const uint32_t *)entries[1].bytes)[3] == 0);
   CHECK(words[2] == sizeof(data) && words[6] == sizeof(cb0));
   CHECK(!pvrgpu_snapshot_stage_uniform_buffers_mapped(
      bindings, 0, 2, 0, 1, 4, words, 12, entries, &count, 15));
   pvrgpu_finish_uniform_buffer_snapshots(entries, &count);
   CHECK(count == 0);
}

static void
test_graphics_shader_buffer_api(void)
{
   uint8_t bytes[16] = {0};
   uint32_t banks[5][32] = {{0}};
   const unsigned prefix[5] = {0, 0, 4, 8, 4};
   struct pvrgpu_systemc_shader_buffer_resource resources[2] = {{
      .resource_token = UINT64_C(0xabc),
      .bytes = bytes,
      .bytes_size = sizeof(bytes),
   }};
   struct pvrgpu_systemc_shader_buffer_binding bindings[5] = {0};
   struct pvrgpu_systemc_graphics_shader_buffers graphics = {
      .resources = resources,
      .resource_count = 1,
      .bindings = bindings,
      .binding_count = 5,
   };
   for (unsigned stage = 0; stage < 5; ++stage) {
      graphics.storage[stage] = (struct pvrgpu_systemc_storage_buffer_abi){
         .descriptor_start = prefix[stage], .descriptor_count = 1,
         .used_mask = 1, .read_mask = 1,
         .write_mask = 1,
      };
      banks[stage][prefix[stage] + 2] = sizeof(bytes);
      bindings[stage] = (struct pvrgpu_systemc_shader_buffer_binding){
         .stage = stage, .access = PVRGPU_SYSTEMC_SHADER_BUFFER_READ |
                                  PVRGPU_SYSTEMC_SHADER_BUFFER_WRITE,
         .bytes_size = sizeof(bytes),
      };
   }
   struct pvrgpu_systemc_tessellation tess = {0};
   tess.control_shared = banks[3];
   tess.control_shared_count = 12;
   tess.control_abi.uniform_buffer_descriptor_start = 8;
   tess.control_abi.shareds = tess.control_abi.push_constant_start = 12;
   tess.evaluation_shared = banks[4];
   tess.evaluation_shared_count = 8;
   tess.evaluation_abi.uniform_buffer_descriptor_start = 4;
   tess.evaluation_abi.shareds = tess.evaluation_abi.push_constant_start = 8;
   struct pvrgpu_systemc_driver_command cmd = {0};
   cmd.vertex_shared = banks[0]; cmd.vertex_shared_count = 4;
   cmd.fragment_shared = banks[1]; cmd.fragment_shared_count = 4;
   cmd.geometry_shared = banks[2]; cmd.geometry_shared_count = 8;
   cmd.vertex_pco_abi.shareds = cmd.vertex_pco_abi.push_constant_start = 4;
   cmd.fragment_pco_abi.shareds = cmd.fragment_pco_abi.push_constant_start = 4;
   cmd.geometry_pco_abi.uniform_buffer_descriptor_start = 4;
   cmd.geometry_pco_abi.shareds = cmd.geometry_pco_abi.push_constant_start = 8;
   cmd.tessellation = &tess;
   cmd.graphics_buffers = &graphics;
   char error[256] = {0};
   CHECK(pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));

   graphics.storage[0].read_mask = graphics.storage[0].write_mask = 0;
   bindings[0].access = 0;
   CHECK(pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));
   bindings[0].access = PVRGPU_SYSTEMC_SHADER_BUFFER_WRITE;
   CHECK(!pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));
   bindings[0].access = 0;
   graphics.storage[0].read_mask = graphics.storage[0].write_mask = 1;
   bindings[0].access = PVRGPU_SYSTEMC_SHADER_BUFFER_READ |
                        PVRGPU_SYSTEMC_SHADER_BUFFER_WRITE;

   /* One whole-resource token is deliberately shared by all five stages. */
   bindings[4].access = PVRGPU_SYSTEMC_SHADER_BUFFER_READ;
   CHECK(!pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));
   bindings[4].access = PVRGPU_SYSTEMC_SHADER_BUFFER_READ |
                        PVRGPU_SYSTEMC_SHADER_BUFFER_WRITE;
   resources[1] = resources[0];
   graphics.resource_count = 2;
   CHECK(!pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));
   graphics.resource_count = 1;
   bindings[2].offset = 4;
   CHECK(!pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));
   bindings[2].offset = 0;
   banks[1][2] = sizeof(bytes) - 4;
   CHECK(!pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));
   banks[1][2] = sizeof(bytes);
   CHECK(pvrgpu_cmd_validate_graphics_buffers(&cmd, NULL, error, sizeof(error)));

   /* With no UBOs the UBO start remains zero, but storage still follows the
    * actual per-stage texture prefix rather than the empty UBO field. */
   uint32_t texture_counts[5] = {1, 0, 0, 0, 0};
   memset(banks[0], 0, sizeof(banks[0]));
   banks[0][22] = sizeof(bytes);
   graphics.storage[0].descriptor_start = 20;
   cmd.vertex_shared_count = 24;
   cmd.vertex_pco_abi.shareds = cmd.vertex_pco_abi.push_constant_start = 24;
   CHECK(pvrgpu_cmd_validate_graphics_buffers(
      &cmd, texture_counts, error, sizeof(error)));

   /* FS descriptors order texture, empty UBO, image, then storage. */
   texture_counts[0] = 0;
   texture_counts[1] = 1;
   memset(banks[0], 0, sizeof(banks[0]));
   banks[0][2] = sizeof(bytes);
   graphics.storage[0].descriptor_start = 0;
   cmd.vertex_shared_count = 4;
   cmd.vertex_pco_abi.shareds = cmd.vertex_pco_abi.push_constant_start = 4;
   memset(banks[1], 0, sizeof(banks[1]));
   banks[1][30] = sizeof(bytes);
   graphics.storage[1].descriptor_start = 28;
   cmd.fragment_image_descriptor_start = 20;
   cmd.fragment_image_descriptor_count = 1;
   cmd.fragment_shared_count = 32;
   cmd.fragment_pco_abi.shareds =
      cmd.fragment_pco_abi.push_constant_start = 32;
   CHECK(pvrgpu_cmd_validate_graphics_buffers(
      &cmd, texture_counts, error, sizeof(error)));

   /* Stages without an executable, shared bank, or storage metadata stay
    * completely empty.  A single-stage payload must not acquire the native
    * GS/TCS/TES system prefix merely because the outer v38 capsule exists. */
   memset(graphics.storage, 0, sizeof(graphics.storage));
   memset(bindings, 0, sizeof(bindings));
   memset(banks, 0, sizeof(banks));
   memset(texture_counts, 0, sizeof(texture_counts));
   memset(&cmd.vertex_pco_abi, 0, sizeof(cmd.vertex_pco_abi));
   memset(&cmd.fragment_pco_abi, 0, sizeof(cmd.fragment_pco_abi));
   memset(&cmd.geometry_pco_abi, 0, sizeof(cmd.geometry_pco_abi));
   cmd.vertex_shared = cmd.fragment_shared = cmd.geometry_shared = NULL;
   cmd.vertex_shared_count = cmd.fragment_shared_count =
      cmd.geometry_shared_count = 0;
   cmd.fragment_image_descriptor_start = 0;
   cmd.fragment_image_descriptor_count = 0;
   cmd.tessellation = NULL;
   graphics.binding_count = 1;
   graphics.storage[0] = (struct pvrgpu_systemc_storage_buffer_abi){
      .descriptor_start = 0, .descriptor_count = 1,
      .used_mask = 1, .read_mask = 1,
   };
   bindings[0] = (struct pvrgpu_systemc_shader_buffer_binding){
      .stage = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_VERTEX,
      .access = PVRGPU_SYSTEMC_SHADER_BUFFER_READ,
      .bytes_size = sizeof(bytes),
   };
   banks[0][2] = sizeof(bytes);
   cmd.vertex_shared = banks[0];
   cmd.vertex_shared_count = 4;
   cmd.vertex_pco_abi.shareds = cmd.vertex_pco_abi.push_constant_start = 4;
   CHECK(pvrgpu_cmd_validate_graphics_buffers(
      &cmd, texture_counts, error, sizeof(error)));

   graphics.storage[1] = graphics.storage[0];
   memset(&graphics.storage[0], 0, sizeof(graphics.storage[0]));
   bindings[0].stage = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT;
   cmd.vertex_shared = NULL;
   cmd.vertex_shared_count = 0;
   memset(&cmd.vertex_pco_abi, 0, sizeof(cmd.vertex_pco_abi));
   cmd.fragment_shared = banks[0];
   cmd.fragment_shared_count = 4;
   cmd.fragment_pco_abi.shareds =
      cmd.fragment_pco_abi.push_constant_start = 4;
   CHECK(pvrgpu_cmd_validate_graphics_buffers(
      &cmd, texture_counts, error, sizeof(error)));
}

static void
test_uniform_storage_alias_gate(void)
{
   uint8_t bytes[16] = {0};
   struct pvrgpu_resource uniform_resource = {0};
   struct pvrgpu_resource other_resource = {0};
   uniform_resource.base.target = other_resource.base.target = PIPE_BUFFER;
   uniform_resource.data = other_resource.data = bytes;
   uniform_resource.size = other_resource.size = sizeof(bytes);
   struct pipe_constant_buffer uniforms[2] = {0};
   uniforms[1] = (struct pipe_constant_buffer){
      .buffer = &uniform_resource.base,
      .buffer_size = sizeof(bytes),
   };
   struct pvrgpu_compute_snapshot storage = {0};
   storage.resource_count = 1;
   storage.binding_count = 1;
   storage.owners[0] = &other_resource.base;
   storage.bindings[0] = (struct pvrgpu_systemc_compute_binding){
      .kind = PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER,
      .resource_index = 0,
      .access = PVRGPU_SYSTEMC_COMPUTE_ACCESS_WRITE,
      .bytes_size = sizeof(bytes),
   };
   CHECK(pvrgpu_uniform_buffers_disjoint_from_writable_snapshot(
      uniforms, 1, 0, &storage));

   storage.owners[0] = &uniform_resource.base;
   storage.bindings[0].access = PVRGPU_SYSTEMC_COMPUTE_ACCESS_READ;
   CHECK(pvrgpu_uniform_buffers_disjoint_from_writable_snapshot(
      uniforms, 1, 0, &storage));
   storage.bindings[0].access |= PVRGPU_SYSTEMC_COMPUTE_ACCESS_WRITE;
   CHECK(!pvrgpu_uniform_buffers_disjoint_from_writable_snapshot(
      uniforms, 1, 0, &storage));

   uniforms[1].buffer = NULL;
   uniforms[1].user_buffer = bytes;
   CHECK(pvrgpu_uniform_buffers_disjoint_from_writable_snapshot(
      uniforms, 1, 0, &storage));

   uniforms[0].user_buffer = NULL;
   uniforms[0].buffer = &uniform_resource.base;
   uniforms[0].buffer_size = sizeof(bytes);
   CHECK(!pvrgpu_uniform_buffers_disjoint_from_writable_snapshot(
      uniforms, 1, 1, &storage));
}

int main(void)
{
   test_compiled_ubo_prefix();
   test_draw_snapshot_lifetime();
   test_ranges_limits_and_raw_bytes();
   test_geometry_sampler_descriptor_prefix();
   test_tessellation_sampler_descriptor_prefix();
   test_graphics_shader_buffer_api();
   test_uniform_storage_alias_gate();
   printf("UBO snapshot/descriptor tests: %u checks, %u failures\n", checks, failures);
   return failures != 0;
}
