/* SPDX-License-Identifier: MIT */

#include "pvrgpu_cmd.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static bool pvrgpu_global_driver_draw_command_emitted;

bool
pvrgpu_driver_draw_command_has_been_emitted(void)
{
   return pvrgpu_global_driver_draw_command_emitted;
}

void
pvrgpu_note_driver_draw_command_emitted(void)
{
   pvrgpu_global_driver_draw_command_emitted = true;
}

static void
pvrgpu_cmd_error(char *error, size_t error_size, const char *message)
{
   if (!error || error_size == 0)
      return;
   snprintf(error, error_size, "%s", message ? message : "unknown error");
}

static bool
pvrgpu_cmd_format_supported(const char *format)
{
   return format &&
          (strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8) == 0 ||
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2) == 0 ||
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2) == 0);
}

static bool
pvrgpu_cmd_validate_common(const char *path,
                           const char *case_name,
                           uint32_t frame,
                           uint32_t width,
                           uint32_t height,
                           const char *format,
                           const char *command_label,
                           char *error,
                           size_t error_size)
{
   if (!path || path[0] == '\0') {
      pvrgpu_cmd_error(error, error_size, "missing driver command output path");
      return false;
   }
   if (!case_name || case_name[0] == '\0') {
      char message[128];
      snprintf(message, sizeof(message), "missing %s command case name",
               command_label);
      pvrgpu_cmd_error(error, error_size, message);
      return false;
   }
   if (frame != 1) {
      pvrgpu_cmd_error(error, error_size, "driver command frame must be 1");
      return false;
   }
   if (width == 0 || height == 0) {
      char message[128];
      snprintf(message, sizeof(message), "%s command dimensions must be positive",
               command_label);
      pvrgpu_cmd_error(error, error_size, message);
      return false;
   }
   if (!pvrgpu_cmd_format_supported(format)) {
      pvrgpu_cmd_error(error, error_size, "unsupported driver command format");
      return false;
   }
   return true;
}

static bool
pvrgpu_cmd_validate_clear(const char *path,
                          const struct pvrgpu_clear_color_command *cmd,
                          char *error,
                          size_t error_size)
{
   if (!cmd) {
      pvrgpu_cmd_error(error, error_size, "missing clear command");
      return false;
   }
   return pvrgpu_cmd_validate_common(path,
                                     cmd->case_name,
                                     cmd->frame,
                                     cmd->width,
                                     cmd->height,
                                     cmd->format,
                                     "clear",
                                     error,
                                     error_size);
}

static bool
pvrgpu_cmd_validate_draw_triangle(
   const char *path,
   const struct pvrgpu_draw_triangle_command *cmd,
   char *error,
   size_t error_size)
{
   if (!cmd) {
      pvrgpu_cmd_error(error, error_size, "missing draw triangle command");
      return false;
   }
   return pvrgpu_cmd_validate_common(path,
                                     cmd->case_name,
                                     cmd->frame,
                                     cmd->width,
                                     cmd->height,
                                     cmd->format,
                                     "draw triangle",
                                     error,
                                     error_size);
}

static bool
pvrgpu_cmd_validate_draw_indexed_quad(
   const char *path,
   const struct pvrgpu_draw_indexed_quad_command *cmd,
   char *error,
   size_t error_size)
{
   if (!cmd) {
      pvrgpu_cmd_error(error, error_size, "missing draw indexed quad command");
      return false;
   }
   if (!pvrgpu_cmd_validate_common(path,
                                   cmd->case_name,
                                   cmd->frame,
                                   cmd->framebuffer_width,
                                   cmd->framebuffer_height,
                                   cmd->format,
                                   "draw indexed quad",
                                   error,
                                   error_size))
      return false;
   if (cmd->width == 0 || cmd->height == 0 ||
       cmd->width > cmd->framebuffer_width ||
       cmd->height > cmd->framebuffer_height) {
      pvrgpu_cmd_error(error, error_size,
                       "draw indexed quad command expects positive viewport "
                       "width/height within framebuffer_width/height");
      return false;
   }
   if (cmd->draw_count == 0 ||
       cmd->index_count != 6 ||
       cmd->unique_vertices != 4 ||
       cmd->primitive_count != 2) {
      pvrgpu_cmd_error(error, error_size,
                       "draw indexed quad command expects draw_count>0, "
                       "index_count=6, unique_vertices=4, primitive_count=2");
      return false;
   }
   return true;
}

bool
pvrgpu_write_clear_color_command(const char *path,
                                 const struct pvrgpu_clear_color_command *cmd,
                                 char *error,
                                 size_t error_size)
{
   if (!pvrgpu_cmd_validate_clear(path, cmd, error, error_size))
      return false;

   FILE *file = fopen(path, "w");
   if (!file) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "cannot open driver command output: %s",
                  strerror(errno));
      }
      return false;
   }

   const int written = fprintf(
      file,
      "schema=%s\n"
      "producer=%s\n"
      "command=clear_color\n"
      "case=%s\n"
      "frame=%u\n"
      "width=%u\n"
      "height=%u\n"
      "format=%s\n"
      "clear_color_bits=%u,%u,%u,%u\n",
      PVRGPU_DRIVER_COMMAND_SCHEMA,
      PVRGPU_DRIVER_COMMAND_PRODUCER,
      cmd->case_name,
      cmd->frame,
      cmd->width,
      cmd->height,
      cmd->format,
      cmd->clear_color_bits[0],
      cmd->clear_color_bits[1],
      cmd->clear_color_bits[2],
      cmd->clear_color_bits[3]);
   const int close_status = fclose(file);
   if (written < 0 || close_status != 0) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "failed to write driver command: %s",
                  strerror(errno));
      }
      return false;
   }
   return true;
}

bool
pvrgpu_write_draw_indexed_quad_command(
   const char *path,
   const struct pvrgpu_draw_indexed_quad_command *cmd,
   char *error,
   size_t error_size)
{
   if (!pvrgpu_cmd_validate_draw_indexed_quad(path, cmd, error, error_size))
      return false;

   FILE *file = fopen(path, "w");
   if (!file) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "cannot open driver command output: %s",
                  strerror(errno));
      }
      return false;
   }

   const int written = fprintf(
      file,
      "schema=%s\n"
      "producer=%s\n"
      "command=draw_indexed_quad\n"
      "case=%s\n"
      "frame=%u\n"
      "framebuffer_width=%u\n"
      "framebuffer_height=%u\n"
      "width=%u\n"
      "height=%u\n"
      "format=%s\n"
      "clear_color_bits=%u,%u,%u,%u\n"
      "draw_count=%u\n"
      "index_count=%u\n"
      "unique_vertices=%u\n"
      "primitive_count=%u\n"
      "semantic_texel_fetches=%" PRIu64 "\n",
      PVRGPU_DRIVER_COMMAND_SCHEMA,
      PVRGPU_DRIVER_COMMAND_PRODUCER,
      cmd->case_name,
      cmd->frame,
      cmd->framebuffer_width,
      cmd->framebuffer_height,
      cmd->width,
      cmd->height,
      cmd->format,
      cmd->clear_color_bits[0],
      cmd->clear_color_bits[1],
      cmd->clear_color_bits[2],
      cmd->clear_color_bits[3],
      cmd->draw_count,
      cmd->index_count,
      cmd->unique_vertices,
      cmd->primitive_count,
      cmd->semantic_texel_fetches);
   const int close_status = fclose(file);
   if (written < 0 || close_status != 0) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "failed to write driver command: %s",
                  strerror(errno));
      }
      return false;
   }
   return true;
}

bool
pvrgpu_write_draw_triangle_command(
   const char *path,
   const struct pvrgpu_draw_triangle_command *cmd,
   char *error,
   size_t error_size)
{
   if (!pvrgpu_cmd_validate_draw_triangle(path, cmd, error, error_size))
      return false;

   FILE *file = fopen(path, "w");
   if (!file) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "cannot open driver command output: %s",
                  strerror(errno));
      }
      return false;
   }

   const int written = fprintf(
      file,
      "schema=%s\n"
      "producer=%s\n"
      "command=draw_triangle\n"
      "case=%s\n"
      "frame=%u\n"
      "width=%u\n"
      "height=%u\n"
      "format=%s\n"
      "clear_color_bits=%u,%u,%u,%u\n"
      "vertex0_bits=%u,%u\n"
      "vertex1_bits=%u,%u\n"
      "vertex2_bits=%u,%u\n"
      "fragment_color_bits=%u,%u,%u,%u\n",
      PVRGPU_DRIVER_COMMAND_SCHEMA,
      PVRGPU_DRIVER_COMMAND_PRODUCER,
      cmd->case_name,
      cmd->frame,
      cmd->width,
      cmd->height,
      cmd->format,
      cmd->clear_color_bits[0],
      cmd->clear_color_bits[1],
      cmd->clear_color_bits[2],
      cmd->clear_color_bits[3],
      cmd->vertex_bits[0][0],
      cmd->vertex_bits[0][1],
      cmd->vertex_bits[1][0],
      cmd->vertex_bits[1][1],
      cmd->vertex_bits[2][0],
      cmd->vertex_bits[2][1],
      cmd->fragment_color_bits[0],
      cmd->fragment_color_bits[1],
      cmd->fragment_color_bits[2],
      cmd->fragment_color_bits[3]);
   const int close_status = fclose(file);
   if (written < 0 || close_status != 0) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "failed to write driver command: %s",
                  strerror(errno));
      }
      return false;
   }
   return true;
}
