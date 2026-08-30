/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_CMD_H
#define PVRGPU_CMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PVRGPU_DRIVER_COMMAND_SCHEMA "pvrgpu.driver-command.v1"
#define PVRGPU_DRIVER_COMMAND_PRODUCER "pvrgpu-gallium-driver"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8 "PIPE_FORMAT_R8G8B8A8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBX8 "PIPE_FORMAT_R8G8B8X8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_BGRX8 "PIPE_FORMAT_B8G8R8X8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2 "PIPE_FORMAT_R10G10B10A2_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2 "PIPE_FORMAT_B10G10R10A2_UNORM"

struct pvrgpu_clear_color_command {
   const char *case_name;
   uint32_t frame;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
};

struct pvrgpu_draw_triangle_command {
   const char *case_name;
   uint32_t frame;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
   uint32_t vertex_bits[3][2];
   uint32_t fragment_color_bits[4];
};

struct pvrgpu_draw_indexed_quad_command {
   const char *case_name;
   uint32_t frame;
   uint32_t framebuffer_width;
   uint32_t framebuffer_height;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
   uint32_t draw_count;
   uint32_t index_count;
   uint32_t unique_vertices;
   uint32_t primitive_count;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint64_t semantic_texel_fetches;
};

struct pvrgpu_draw_primitive_sequence_command {
   const char *case_name;
   uint32_t frame;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
   uint32_t draw_count;
   uint32_t ia_vertices;
   uint32_t ia_primitives;
   uint32_t vs_invocations;
   uint32_t clip_invocations;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint64_t ps_invocations;
   uint64_t semantic_texel_fetches;
};

bool
pvrgpu_write_clear_color_command(const char *path,
                                 const struct pvrgpu_clear_color_command *cmd,
                                 char *error,
                                 size_t error_size);

bool
pvrgpu_write_draw_triangle_command(const char *path,
                                   const struct pvrgpu_draw_triangle_command *cmd,
                                   char *error,
                                   size_t error_size);

bool
pvrgpu_write_draw_indexed_quad_command(
   const char *path,
   const struct pvrgpu_draw_indexed_quad_command *cmd,
   char *error,
   size_t error_size);

bool
pvrgpu_write_draw_primitive_sequence_command(
   const char *path,
   const struct pvrgpu_draw_primitive_sequence_command *cmd,
   char *error,
   size_t error_size);

bool
pvrgpu_driver_draw_command_has_been_emitted(void);

void
pvrgpu_note_driver_draw_command_emitted(void);

bool
pvrgpu_driver_counter_sequence_command_has_been_emitted(void);

void
pvrgpu_note_driver_counter_sequence_command_emitted(void);

#ifdef __cplusplus
}
#endif

#endif /* PVRGPU_CMD_H */
