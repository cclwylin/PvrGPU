/* SPDX-License-Identifier: MIT */

#include "pvrgpu_cmd.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_systemc_api.h"

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool pvrgpu_global_driver_draw_command_emitted;
static bool pvrgpu_global_driver_counter_sequence_command_emitted;

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

bool
pvrgpu_driver_counter_sequence_command_has_been_emitted(void)
{
   return pvrgpu_global_driver_counter_sequence_command_emitted;
}

void
pvrgpu_note_driver_counter_sequence_command_emitted(void)
{
   pvrgpu_global_driver_counter_sequence_command_emitted = true;
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
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_RGBX8) == 0 ||
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_BGRX8) == 0 ||
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_R5G6B5) == 0 ||
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_B5G6R5) == 0 ||
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2) == 0 ||
           strcmp(format, PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2) == 0);
}

static const char *
pvrgpu_nonempty_env(const char *name)
{
   const char *value = getenv(name);
   return value && value[0] != '\0' ? value : NULL;
}

static const char *
pvrgpu_safe_text(const char *text)
{
   return text ? text : "";
}

static void
pvrgpu_systemc_submit_info_init(
   struct pvrgpu_systemc_submit_info *info,
   const struct pvrgpu_systemc_driver_command *command)
{
   memset(info, 0, sizeof(*info));
   info->version = PVRGPU_SYSTEMC_API_VERSION;
   info->command = command;
   info->jsonl_path = pvrgpu_nonempty_env("PVRGPU_SYSTEMC_JSONL_OUT");
   info->stderr_path = pvrgpu_nonempty_env("PVRGPU_SYSTEMC_STDERR_OUT");
   info->outdir = pvrgpu_nonempty_env("PVRGPU_SYSTEMC_OUTDIR");
   info->memory_mode = pvrgpu_nonempty_env("PVRGPU_MODEL_MEMORY_MODE");
}

static bool
pvrgpu_submit_systemc_api(const struct pvrgpu_systemc_driver_command *command,
                          char *error,
                          size_t error_size)
{
   const char *library_path = pvrgpu_nonempty_env("PVRGPU_SYSTEMC_API_LIB");
   if (!library_path) {
      pvrgpu_counter_eventf("systemc_api_disabled",
                            "command=%s case=%s",
                            pvrgpu_safe_text(command ? command->command : NULL),
                            pvrgpu_safe_text(command ? command->case_name : NULL));
      return true;
   }

   if (!pvrgpu_nonempty_env("PVRGPU_SYSTEMC_JSONL_OUT") ||
       !pvrgpu_nonempty_env("PVRGPU_SYSTEMC_OUTDIR")) {
      pvrgpu_cmd_error(error, error_size,
                       "SystemC API requires PVRGPU_SYSTEMC_JSONL_OUT and "
                       "PVRGPU_SYSTEMC_OUTDIR");
      pvrgpu_counter_eventf("systemc_api_error",
                            "stage=env command=%s case=%s reason=%s",
                            pvrgpu_safe_text(command ? command->command : NULL),
                            pvrgpu_safe_text(command ? command->case_name : NULL),
                            pvrgpu_safe_text(error && error_size != 0 ?
                                             error : NULL));
      return false;
   }

   static void *handle;
   static pvrgpu_systemc_submit_driver_command_fn submit;
   static const char *loaded_library_path;
   if (!handle || loaded_library_path != library_path) {
      dlerror();
      handle = dlopen(library_path, RTLD_NOW | RTLD_GLOBAL);
      if (!handle) {
         const char *dl_message = dlerror();
         char message[512];
         snprintf(message, sizeof(message),
                  "cannot load SystemC API library: %s",
                  dl_message ? dl_message : library_path);
         pvrgpu_cmd_error(error, error_size, message);
         pvrgpu_counter_eventf("systemc_api_error",
                               "stage=dlopen command=%s case=%s library=%s "
                               "reason=%s",
                               pvrgpu_safe_text(command ? command->command :
                                                NULL),
                               pvrgpu_safe_text(command ? command->case_name :
                                                NULL),
                               pvrgpu_safe_text(library_path),
                               pvrgpu_safe_text(message));
         return false;
      }
      loaded_library_path = library_path;
      dlerror();
      submit = (pvrgpu_systemc_submit_driver_command_fn)
         dlsym(handle, "pvrgpu_systemc_submit_driver_command");
      const char *symbol_error = dlerror();
      if (symbol_error || !submit) {
         char message[512];
         snprintf(message, sizeof(message),
                  "cannot resolve SystemC API submit symbol: %s",
                  symbol_error ? symbol_error :
                                 "pvrgpu_systemc_submit_driver_command");
         pvrgpu_cmd_error(error, error_size, message);
         pvrgpu_counter_eventf("systemc_api_error",
                               "stage=dlsym command=%s case=%s library=%s "
                               "reason=%s",
                               pvrgpu_safe_text(command ? command->command :
                                                NULL),
                               pvrgpu_safe_text(command ? command->case_name :
                                                NULL),
                               pvrgpu_safe_text(library_path),
                               pvrgpu_safe_text(message));
         return false;
      }
   }

   struct pvrgpu_systemc_submit_info info;
   pvrgpu_systemc_submit_info_init(&info, command);
   pvrgpu_counter_eventf("systemc_api_submit",
                         "command=%s case=%s jsonl=%s outdir=%s mode=%s",
                         pvrgpu_safe_text(command ? command->command : NULL),
                         pvrgpu_safe_text(command ? command->case_name : NULL),
                         pvrgpu_safe_text(info.jsonl_path),
                         pvrgpu_safe_text(info.outdir),
                         pvrgpu_safe_text(info.memory_mode));
   if (error && error_size != 0)
      error[0] = '\0';
   const int result = submit(&info, error, error_size);
   if (result != 0) {
      if (error && error_size != 0 && error[0] == '\0')
         snprintf(error, error_size, "SystemC API returned %d", result);
      pvrgpu_counter_eventf("systemc_api_error",
                            "stage=submit command=%s case=%s result=%d "
                            "reason=%s",
                            pvrgpu_safe_text(command ? command->command : NULL),
                            pvrgpu_safe_text(command ? command->case_name :
                                             NULL),
                            result,
                            pvrgpu_safe_text(error && error_size != 0 ?
                                             error : NULL));
      return false;
   }
   pvrgpu_counter_eventf("systemc_api_done",
                         "command=%s case=%s",
                         pvrgpu_safe_text(command ? command->command : NULL),
                         pvrgpu_safe_text(command ? command->case_name : NULL));
   return true;
}

static void
pvrgpu_systemc_command_init(struct pvrgpu_systemc_driver_command *command,
                            const char *command_name,
                            const char *case_name,
                            uint32_t frame,
                            uint32_t framebuffer_width,
                            uint32_t framebuffer_height,
                            uint32_t width,
                            uint32_t height,
                            const char *format,
                            const uint32_t clear_color_bits[4])
{
   memset(command, 0, sizeof(*command));
   command->version = PVRGPU_SYSTEMC_API_VERSION;
   command->schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command->producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command->command = command_name;
   command->case_name = case_name;
   command->format = format;
   command->frame = frame;
   command->framebuffer_width = framebuffer_width ? framebuffer_width : width;
   command->framebuffer_height = framebuffer_height ? framebuffer_height : height;
   command->width = width;
   command->height = height;
   if (clear_color_bits) {
      command->clear_color_bits[0] = clear_color_bits[0];
      command->clear_color_bits[1] = clear_color_bits[1];
      command->clear_color_bits[2] = clear_color_bits[2];
      command->clear_color_bits[3] = clear_color_bits[3];
   }
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
   if (cmd->clip_primitives > cmd->primitive_count ||
       cmd->setup_triangles > cmd->primitive_count) {
      pvrgpu_cmd_error(error, error_size,
                       "draw indexed quad command expects clip_primitives and "
                       "setup_triangles within primitive_count");
      return false;
   }
   return true;
}

static bool
pvrgpu_cmd_validate_draw_primitive_sequence(
   const char *path,
   const struct pvrgpu_draw_primitive_sequence_command *cmd,
   char *error,
   size_t error_size)
{
   if (!cmd) {
      pvrgpu_cmd_error(error, error_size,
                       "missing draw primitive sequence command");
      return false;
   }
   if (!pvrgpu_cmd_validate_common(path,
                                   cmd->case_name,
                                   cmd->frame,
                                   cmd->width,
                                   cmd->height,
                                   cmd->format,
                                   "draw primitive sequence",
                                   error,
                                   error_size))
      return false;
   if (cmd->draw_count == 0 || cmd->setup_triangles > cmd->clip_primitives) {
      pvrgpu_cmd_error(error, error_size,
                       "draw primitive sequence command contains invalid "
                       "semantic counter metadata");
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

   int written = fprintf(
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

   struct pvrgpu_systemc_driver_command api_command;
   pvrgpu_systemc_command_init(&api_command,
                               "clear_color",
                               cmd->case_name,
                               cmd->frame,
                               cmd->width,
                               cmd->height,
                               cmd->width,
                               cmd->height,
                               cmd->format,
                               cmd->clear_color_bits);
   return pvrgpu_submit_systemc_api(&api_command, error, error_size);
}

bool
pvrgpu_write_draw_primitive_sequence_command(
   const char *path,
   const struct pvrgpu_draw_primitive_sequence_command *cmd,
   char *error,
   size_t error_size)
{
   if (!pvrgpu_cmd_validate_draw_primitive_sequence(path, cmd, error,
                                                    error_size))
      return false;

   FILE *file = fopen(path, "w");
   if (!file) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "cannot open driver command output: %s",
                  strerror(errno));
      }
      return false;
   }

   int written = fprintf(
      file,
      "schema=%s\n"
      "producer=%s\n"
      "command=draw_primitive_sequence\n"
      "case=%s\n"
      "frame=%u\n"
      "width=%u\n"
      "height=%u\n"
      "format=%s\n"
      "clear_color_bits=%u,%u,%u,%u\n"
      "draw_count=%u\n"
      "ia_vertices=%u\n"
      "ia_primitives=%u\n"
      "vs_invocations=%u\n"
      "gs_invocations=%u\n"
      "gs_primitives=%u\n"
      "clip_invocations=%u\n"
      "clip_primitives=%u\n"
      "setup_triangles=%u\n"
      "ps_invocations=%" PRIu64 "\n"
      "hs_invocations=%u\n"
      "ds_invocations=%u\n"
      "cs_invocations=%u\n"
      "semantic_texel_fetches=%" PRIu64 "\n",
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
      cmd->draw_count,
      cmd->ia_vertices,
      cmd->ia_primitives,
      cmd->vs_invocations,
      cmd->gs_invocations,
      cmd->gs_primitives,
      cmd->clip_invocations,
      cmd->clip_primitives,
      cmd->setup_triangles,
      cmd->ps_invocations,
      cmd->hs_invocations,
      cmd->ds_invocations,
      cmd->cs_invocations,
      cmd->semantic_texel_fetches);
   if (written >= 0 &&
       cmd->framebuffer_rgba8_path &&
       cmd->framebuffer_rgba8_path[0] != '\0') {
      const int extra_written = fprintf(file,
                                        "framebuffer_rgba8_path=%s\n",
                                        cmd->framebuffer_rgba8_path);
      if (extra_written < 0)
         written = extra_written;
   }
   const int close_status = fclose(file);
   if (written < 0 || close_status != 0) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "failed to write driver command: %s",
                  strerror(errno));
      }
      return false;
   }

   struct pvrgpu_systemc_driver_command api_command;
   pvrgpu_systemc_command_init(&api_command,
                               "draw_primitive_sequence",
                               cmd->case_name,
                               cmd->frame,
                               cmd->width,
                               cmd->height,
                               cmd->width,
                               cmd->height,
                               cmd->format,
                               cmd->clear_color_bits);
   api_command.framebuffer_rgba8_path = cmd->framebuffer_rgba8_path;
   api_command.draw_count = cmd->draw_count;
   api_command.ia_vertices = cmd->ia_vertices;
   api_command.ia_primitives = cmd->ia_primitives;
   api_command.vs_invocations = cmd->vs_invocations;
   api_command.gs_invocations = cmd->gs_invocations;
   api_command.gs_primitives = cmd->gs_primitives;
   api_command.clip_invocations = cmd->clip_invocations;
   api_command.clip_primitives = cmd->clip_primitives;
   api_command.setup_triangles = cmd->setup_triangles;
   api_command.ps_invocations = cmd->ps_invocations;
   api_command.hs_invocations = cmd->hs_invocations;
   api_command.ds_invocations = cmd->ds_invocations;
   api_command.cs_invocations = cmd->cs_invocations;
   api_command.semantic_texel_fetches = cmd->semantic_texel_fetches;
   return pvrgpu_submit_systemc_api(&api_command, error, error_size);
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
      "clip_primitives=%u\n"
      "setup_triangles=%u\n"
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
      cmd->clip_primitives,
      cmd->setup_triangles,
      cmd->semantic_texel_fetches);
   const int close_status = fclose(file);
   if (written < 0 || close_status != 0) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "failed to write driver command: %s",
                  strerror(errno));
      }
      return false;
   }

   struct pvrgpu_systemc_driver_command api_command;
   pvrgpu_systemc_command_init(&api_command,
                               "draw_indexed_quad",
                               cmd->case_name,
                               cmd->frame,
                               cmd->framebuffer_width,
                               cmd->framebuffer_height,
                               cmd->width,
                               cmd->height,
                               cmd->format,
                               cmd->clear_color_bits);
   api_command.draw_count = cmd->draw_count;
   api_command.index_count = cmd->index_count;
   api_command.unique_vertices = cmd->unique_vertices;
   api_command.primitive_count = cmd->primitive_count;
   api_command.clip_primitives = cmd->clip_primitives;
   api_command.setup_triangles = cmd->setup_triangles;
   api_command.semantic_texel_fetches = cmd->semantic_texel_fetches;
   return pvrgpu_submit_systemc_api(&api_command, error, error_size);
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

   struct pvrgpu_systemc_driver_command api_command;
   pvrgpu_systemc_command_init(&api_command,
                               "draw_triangle",
                               cmd->case_name,
                               cmd->frame,
                               cmd->width,
                               cmd->height,
                               cmd->width,
                               cmd->height,
                               cmd->format,
                               cmd->clear_color_bits);
   for (unsigned vertex = 0; vertex < 3; ++vertex) {
      api_command.vertex_bits[vertex][0] = cmd->vertex_bits[vertex][0];
      api_command.vertex_bits[vertex][1] = cmd->vertex_bits[vertex][1];
   }
   api_command.fragment_color_bits[0] = cmd->fragment_color_bits[0];
   api_command.fragment_color_bits[1] = cmd->fragment_color_bits[1];
   api_command.fragment_color_bits[2] = cmd->fragment_color_bits[2];
   api_command.fragment_color_bits[3] = cmd->fragment_color_bits[3];
   return pvrgpu_submit_systemc_api(&api_command, error, error_size);
}
