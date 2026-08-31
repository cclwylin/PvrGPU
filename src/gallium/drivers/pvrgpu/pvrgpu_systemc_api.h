/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SYSTEMC_API_H
#define PVRGPU_SYSTEMC_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PVRGPU_SYSTEMC_API_VERSION 1u

struct pvrgpu_systemc_driver_command {
   uint32_t version;
   const char *schema;
   const char *producer;
   const char *command;
   const char *case_name;
   const char *format;
   const char *framebuffer_rgba8_path;
   uint32_t frame;
   uint32_t framebuffer_width;
   uint32_t framebuffer_height;
   uint32_t width;
   uint32_t height;
   uint32_t clear_color_bits[4];
   uint32_t vertex_bits[3][2];
   uint32_t fragment_color_bits[4];
   uint32_t draw_count;
   uint32_t index_count;
   uint32_t unique_vertices;
   uint32_t primitive_count;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint32_t ia_vertices;
   uint32_t ia_primitives;
   uint32_t vs_invocations;
   uint32_t gs_invocations;
   uint32_t gs_primitives;
   uint32_t clip_invocations;
   uint32_t hs_invocations;
   uint32_t ds_invocations;
   uint32_t cs_invocations;
   uint64_t ps_invocations;
   uint64_t semantic_texel_fetches;
};

struct pvrgpu_systemc_submit_info {
   uint32_t version;
   const struct pvrgpu_systemc_driver_command *command;
   const char *jsonl_path;
   const char *stderr_path;
   const char *outdir;
   const char *memory_mode;
};

typedef int (*pvrgpu_systemc_submit_driver_command_fn)(
   const struct pvrgpu_systemc_submit_info *info,
   char *error,
   size_t error_size);

int
pvrgpu_systemc_submit_driver_command(
   const struct pvrgpu_systemc_submit_info *info,
   char *error,
   size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* PVRGPU_SYSTEMC_API_H */
