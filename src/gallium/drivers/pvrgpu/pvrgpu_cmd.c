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
#include <sys/stat.h>

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

bool
pvrgpu_case_reserves_native_pco_sequence(void)
{
   const char *case_name = getenv("PVRGPU_RDC_CASE_NAME");
   return case_name &&
          (strcmp(case_name, "refract.refract.capture.1") == 0 ||
           strcmp(case_name, "shadow.shadow.capture.1") == 0 ||
           strcmp(case_name, "terrain.terrain.capture.1") == 0);
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

static bool
pvrgpu_pco_single_draw_resolution_supported(uint32_t framebuffer_width,
                                            uint32_t framebuffer_height,
                                            uint32_t width,
                                            uint32_t height)
{
   /*
    * The model's rasterizer is resolution independent, so the single-draw
    * profile only needs a full-surface render target inside the addressable
    * extent the nested PCO sequence header already enforces.
    */
   return width == framebuffer_width && height == framebuffer_height &&
          framebuffer_width != 0 && framebuffer_height != 0 &&
          framebuffer_width <= 4096 && framebuffer_height <= 4096;
}

static void
pvrgpu_pco_viewport_bits(uint32_t framebuffer_width,
                         uint32_t framebuffer_height,
                         uint32_t bits[3])
{
   const float values[3] = {
      (float)framebuffer_width * 0.5f,
      (float)framebuffer_height * 0.5f,
      0.5f,
   };
   memcpy(bits, values, sizeof(values));
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
      const bool native_sequence =
         command && command->command &&
         strcmp(command->command, "draw_pco_sequence") == 0;
      if (native_sequence) {
         pvrgpu_cmd_error(error, error_size,
                          "draw_pco_sequence requires "
                          "PVRGPU_SYSTEMC_API_LIB");
         pvrgpu_counter_eventf(
            "systemc_api_error",
            "stage=env command=%s case=%s reason=%s",
            pvrgpu_safe_text(command->command),
            pvrgpu_safe_text(command->case_name),
            pvrgpu_safe_text(error && error_size != 0 ? error : NULL));
         return false;
      }
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

static void
pvrgpu_systemc_copy_pco_stage_abi(
   struct pvrgpu_systemc_pco_stage_abi *destination,
   const struct pvrgpu_draw_pco_stage_abi *source)
{
   destination->temps = source->temps;
   destination->vertex_inputs = source->vertex_inputs;
   destination->vertex_outputs = source->vertex_outputs;
   destination->coefficients = source->coefficients;
   destination->shareds = source->shareds;
   destination->push_constant_start = source->push_constant_start;
   destination->push_constant_count = source->push_constant_count;
   destination->entry_offset = source->entry_offset;
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
pvrgpu_cmd_texture_sidecar_size_matches(
   const char *command_path,
   const struct pvrgpu_draw_textured_triangles_command *cmd,
   char *error,
   size_t error_size)
{
   static const char suffix[] = ".texture.rgba8";
   if (!cmd->texture_rgba8_path || cmd->texture_rgba8_path[0] == '\0' ||
       strchr(cmd->texture_rgba8_path, '\n') ||
       strchr(cmd->texture_rgba8_path, '\r')) {
      pvrgpu_cmd_error(error, error_size,
                       "missing or invalid texture RGBA8 sidecar path");
      return false;
   }

   const size_t command_path_length = strlen(command_path);
   const size_t sidecar_path_length = strlen(cmd->texture_rgba8_path);
   const size_t suffix_length = sizeof(suffix) - 1u;
   if (command_path_length > SIZE_MAX - suffix_length ||
       sidecar_path_length != command_path_length + suffix_length ||
       memcmp(cmd->texture_rgba8_path,
              command_path,
              command_path_length) != 0 ||
       memcmp(cmd->texture_rgba8_path + command_path_length,
              suffix,
              suffix_length) != 0) {
      pvrgpu_cmd_error(error, error_size,
                       "texture RGBA8 sidecar must be next to the driver "
                       "command as <command>.texture.rgba8");
      return false;
   }

   const size_t row_bytes = (size_t)cmd->texture_width * 4u;
   if (row_bytes / 4u != cmd->texture_width) {
      pvrgpu_cmd_error(error, error_size,
                       "texture RGBA8 sidecar row size overflows size_t");
      return false;
   }
   if (cmd->texture_height > SIZE_MAX / row_bytes) {
      pvrgpu_cmd_error(error, error_size,
                       "texture RGBA8 sidecar size overflows size_t");
      return false;
   }
   const size_t expected_size = row_bytes * (size_t)cmd->texture_height;

   struct stat sidecar_stat;
   if (stat(cmd->texture_rgba8_path, &sidecar_stat) != 0 ||
       !S_ISREG(sidecar_stat.st_mode) ||
       sidecar_stat.st_size < 0 ||
       (uint64_t)sidecar_stat.st_size != (uint64_t)expected_size) {
      pvrgpu_cmd_error(error, error_size,
                       "texture RGBA8 sidecar size does not match "
                       "texture_width*texture_height*4");
      return false;
   }
   return true;
}

static bool
pvrgpu_cmd_validate_draw_textured_triangles(
   const char *path,
   const struct pvrgpu_draw_textured_triangles_command *cmd,
   char *error,
   size_t error_size)
{
   if (!cmd) {
      pvrgpu_cmd_error(error, error_size,
                       "missing draw textured triangles command");
      return false;
   }
   if (!pvrgpu_cmd_validate_common(path,
                                   cmd->case_name,
                                   cmd->frame,
                                   cmd->framebuffer_width,
                                   cmd->framebuffer_height,
                                   cmd->format,
                                   "draw textured triangles",
                                   error,
                                   error_size))
      return false;
   if (cmd->width != cmd->framebuffer_width ||
       cmd->height != cmd->framebuffer_height) {
      pvrgpu_cmd_error(error, error_size,
                       "draw textured triangles command requires viewport "
                       "and framebuffer dimensions to match");
      return false;
   }
   if (cmd->texture_width == 0 || cmd->texture_height == 0) {
      pvrgpu_cmd_error(error, error_size,
                       "draw textured triangles texture dimensions must be "
                       "positive");
      return false;
   }
   return pvrgpu_cmd_texture_sidecar_size_matches(path,
                                                   cmd,
                                                   error,
                                                   error_size);
}

static bool
pvrgpu_cmd_validate_draw_pco_triangles(
   const char *path,
   const struct pvrgpu_draw_pco_triangles_command *cmd,
   char *error,
   size_t error_size)
{
   if (!cmd) {
      pvrgpu_cmd_error(error, error_size,
                       "missing draw PCO triangles command");
      return false;
   }
   if (!pvrgpu_cmd_validate_common(path,
                                   cmd->case_name,
                                   cmd->frame,
                                   cmd->framebuffer_width,
                                   cmd->framebuffer_height,
                                   cmd->format,
                                   "draw PCO triangles",
                                   error,
                                   error_size))
      return false;

   if (!pvrgpu_pco_single_draw_resolution_supported(
          cmd->framebuffer_width,
          cmd->framebuffer_height,
          cmd->width,
          cmd->height) ||
       strcmp(cmd->format, PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8) != 0) {
      pvrgpu_cmd_error(error, error_size,
                       "draw PCO triangles requires a full-surface RGBA8 "
                       "render target within the model extent");
      return false;
   }
   if (cmd->clear_color_bits[0] != 0 ||
       cmd->clear_color_bits[1] != 0 ||
       cmd->clear_color_bits[2] != 0 ||
       cmd->clear_color_bits[3] != UINT32_C(0x3f800000)) {
      pvrgpu_cmd_error(error, error_size,
                       "draw PCO triangles expects opaque black clear color");
      return false;
   }
   const uint64_t end_vertex =
      (uint64_t)cmd->first_vertex + cmd->vertex_count;
   const uint64_t expected_vertex_bytes =
      end_vertex * (uint64_t)cmd->vertex_stride;
   const bool conditionals_layout =
      cmd->vertex_stride == 12 && cmd->vertex_pco_abi.vertex_inputs == 4;
   const bool color_layout =
      cmd->vertex_stride == 24 && cmd->vertex_pco_abi.vertex_inputs == 8 &&
      cmd->vertex_pco_abi.shareds == 0;
   const bool lit_mesh_layout =
      cmd->vertex_stride == 24 && cmd->vertex_pco_abi.vertex_inputs == 8 &&
      !color_layout;
   const bool texture_layout =
      cmd->vertex_stride == 32 && cmd->vertex_pco_abi.vertex_inputs == 12;
   const bool ideas_case =
      strcmp(cmd->case_name, "ideas.ideas.capture.1") == 0;
   const bool ideas_position_layout =
      ideas_case && cmd->vertex_stride == 16 &&
      cmd->vertex_pco_abi.vertex_inputs == 4;
   const bool ideas_two_attribute_layout =
      ideas_case && cmd->vertex_stride == 32 &&
      cmd->vertex_pco_abi.vertex_inputs == 8;
   const bool ideas_layout =
      ideas_position_layout || ideas_two_attribute_layout;
   const bool ideas_topology =
      (cmd->primitive_mode == 5 &&
       (cmd->vertex_count == 18 || cmd->vertex_count == 26)) ||
      (cmd->primitive_mode == 6 && cmd->vertex_count == 12);
   /*
    * Non-indexed triangle topologies the SystemC submitter expands itself:
    * a triangle list of whole triangles, or a strip/fan of three or more
    * vertices.
    */
   const bool ordinary_topology =
      (cmd->primitive_mode == 4 && cmd->vertex_count % 3 == 0 &&
       cmd->vertex_count >= 3) ||
      ((cmd->primitive_mode == 5 || cmd->primitive_mode == 6) &&
       cmd->vertex_count >= 3);
   if (!cmd->raw_vertex_data || cmd->vertex_count == 0 ||
       (!ideas_layout && !ordinary_topology) ||
       (ideas_layout && !ideas_topology) ||
       (!conditionals_layout && !lit_mesh_layout && !texture_layout &&
        !ideas_layout && !color_layout) ||
       expected_vertex_bytes == 0 || expected_vertex_bytes > SIZE_MAX ||
       cmd->raw_vertex_data_size != (size_t)expected_vertex_bytes ||
       cmd->first_vertex != 0 || cmd->instance_count != 1 ||
       cmd->indexed != 0) {
      pvrgpu_cmd_error(error, error_size,
                       "draw PCO triangles has malformed nonindexed vertex "
                       "payload/topology");
      return false;
   }

   if (!texture_layout) {
      if (cmd->sampled_texture_count != 0 || cmd->sampled_texture_bytes ||
          cmd->sampled_texture_bytes_size != 0 ||
          cmd->sampled_texture_width != 0 ||
          cmd->sampled_texture_height != 0 ||
          cmd->sampled_texture_row_pitch != 0 ||
          cmd->sampled_texture_format ||
          cmd->sampled_texture_mip_count != 0) {
         pvrgpu_cmd_error(error, error_size,
                          "draw PCO triangles has texture state on an "
                          "untextured profile");
         return false;
      }
   } else {
      const uint64_t tight_row_pitch =
         (uint64_t)cmd->sampled_texture_width * 4u;
      const uint64_t expected_texture_bytes =
         (uint64_t)cmd->sampled_texture_row_pitch *
         cmd->sampled_texture_height;
      if (cmd->sampled_texture_count != 1 ||
          !cmd->sampled_texture_bytes ||
          cmd->sampled_texture_width != 512 ||
          cmd->sampled_texture_height != 512 ||
          tight_row_pitch != 2048 ||
          cmd->sampled_texture_row_pitch != tight_row_pitch ||
          cmd->sampled_texture_mip_count != 1 ||
          !cmd->sampled_texture_format ||
          strcmp(cmd->sampled_texture_format,
                 PVRGPU_DRIVER_COMMAND_FORMAT_RGBX8) != 0 ||
          expected_texture_bytes == 0 ||
          expected_texture_bytes > SIZE_MAX ||
          cmd->sampled_texture_bytes_size !=
             (size_t)expected_texture_bytes ||
          cmd->sampled_texture_bytes_size != UINT64_C(1048576)) {
         pvrgpu_cmd_error(error, error_size,
                          "draw PCO triangles has incompatible sampled "
                          "texture payload");
         return false;
      }
   }

   const bool empty_sequence_counters =
      cmd->draw_count == 0 && cmd->ia_vertices == 0 &&
      cmd->ia_primitives == 0 && cmd->vs_invocations == 0 &&
      cmd->gs_invocations == 0 && cmd->gs_primitives == 0 &&
      cmd->clip_invocations == 0 && cmd->clip_primitives == 0 &&
      cmd->hs_invocations == 0 && cmd->ds_invocations == 0 &&
      cmd->cs_invocations == 0 && cmd->ps_invocations == 0 &&
      cmd->setup_triangles == 0 && cmd->semantic_texel_fetches == 0;
   const bool ideas_sequence_counters =
      cmd->draw_count == 180 && cmd->ia_vertices == 3370 &&
      cmd->ia_primitives == 3010 && cmd->vs_invocations == 3370 &&
      cmd->gs_invocations == 0 && cmd->gs_primitives == 0 &&
      cmd->clip_invocations == 3010 && cmd->clip_primitives == 3004 &&
      cmd->hs_invocations == 0 && cmd->ds_invocations == 0 &&
      cmd->cs_invocations == 0 &&
      cmd->ps_invocations ==
         (cmd->framebuffer_width == 80 ? UINT64_C(1553)
                                       : UINT64_C(155163)) &&
      cmd->setup_triangles == 3004 && cmd->semantic_texel_fetches == 0;
   if ((!ideas_layout && !empty_sequence_counters) ||
       (ideas_layout && !empty_sequence_counters &&
        !ideas_sequence_counters)) {
      pvrgpu_cmd_error(error, error_size,
                       "draw PCO triangles has incompatible sequence "
                       "counter metadata");
      return false;
   }

   if (!cmd->vertex_pco || cmd->vertex_pco_size == 0 ||
       cmd->vertex_pco_size > UINT32_MAX || !cmd->fragment_pco ||
       cmd->fragment_pco_size == 0 || cmd->fragment_pco_size > UINT32_MAX ||
       (cmd->vertex_shared_count != 0 && !cmd->vertex_shared) ||
       (cmd->fragment_shared_count != 0 && !cmd->fragment_shared) ||
       cmd->vertex_shared_count != cmd->vertex_pco_abi.shareds ||
       cmd->fragment_shared_count != cmd->fragment_pco_abi.shareds) {
      pvrgpu_cmd_error(error, error_size,
                       "draw PCO triangles has incompatible binary/shared "
                       "payload");
      return false;
   }
   /*
    * A pass-through vertex shader that only forwards position and colour uses
    * no temporaries, exactly as the color layout's fragment stage may.
    */
   if ((!color_layout && cmd->vertex_pco_abi.temps == 0) ||
       cmd->vertex_pco_abi.temps > 256 ||
       cmd->vertex_pco_abi.shareds > 64 ||
       cmd->vertex_pco_abi.coefficients != 0 ||
       cmd->vertex_pco_abi.push_constant_start != 0 ||
       cmd->vertex_pco_abi.push_constant_count !=
          cmd->vertex_pco_abi.shareds ||
       cmd->vertex_pco_abi.entry_offset != 0 ||
       (!ideas_position_layout && !color_layout &&
        cmd->fragment_pco_abi.temps == 0) ||
       cmd->fragment_pco_abi.temps > 256 ||
       cmd->fragment_pco_abi.shareds > 64 ||
       cmd->fragment_pco_abi.vertex_inputs != 0 ||
       cmd->fragment_pco_abi.vertex_outputs != 0 ||
       cmd->fragment_pco_abi.push_constant_start != 0 ||
       cmd->fragment_pco_abi.entry_offset != 0 ||
       cmd->position_output_start != 0 ||
       cmd->position_output_count != 4 ||
       cmd->vertex_pco_abi.vertex_outputs !=
          cmd->position_output_count + cmd->varying_output_count ||
       (cmd->varying_output_count != 0 &&
        cmd->varying_output_start != cmd->position_output_count) ||
       cmd->fragment_position_start != 0 ||
       (cmd->fragment_varying_count != 0 &&
        cmd->fragment_varying_start != cmd->fragment_position_count) ||
       cmd->fragment_pco_abi.coefficients !=
          cmd->fragment_position_count + cmd->fragment_varying_count ||
       ((conditionals_layout) &&
        (cmd->varying_output_count != 0 ||
         cmd->fragment_position_count != 0 ||
         cmd->fragment_varying_count != 0)) ||
       ((lit_mesh_layout) &&
        (cmd->varying_output_count == 0 ||
         cmd->varying_output_count > 4 ||
         cmd->fragment_position_count != 4 ||
         cmd->fragment_varying_count != cmd->varying_output_count * 4)) ||
       ((color_layout) &&
        (cmd->varying_output_count != 4 ||
         cmd->fragment_position_count != 4 ||
         cmd->fragment_varying_count != 16)) ||
       (ideas_position_layout &&
        (cmd->vertex_pco_abi.vertex_outputs != 4 ||
         cmd->vertex_pco_abi.shareds != 32 ||
         cmd->varying_output_start != 0 ||
         cmd->varying_output_count != 0 ||
         cmd->fragment_position_count != 0 ||
         cmd->fragment_varying_start != 0 ||
         cmd->fragment_varying_count != 0 ||
         cmd->fragment_pco_abi.coefficients != 0 ||
         (cmd->fragment_pco_abi.shareds != 0 &&
          cmd->fragment_pco_abi.shareds != 4))) ||
       (ideas_two_attribute_layout &&
        (cmd->vertex_pco_abi.vertex_outputs != 14 ||
         cmd->vertex_pco_abi.shareds != 44 ||
         cmd->fragment_pco_abi.shareds != 12 ||
         cmd->varying_output_start != 4 ||
         cmd->varying_output_count != 10 ||
         cmd->fragment_position_count != 4 ||
         cmd->fragment_varying_start != 4 ||
         cmd->fragment_varying_count != 40 ||
         cmd->fragment_pco_abi.coefficients != 44)) ||
       ((!texture_layout) &&
        cmd->fragment_pco_abi.push_constant_count !=
           cmd->fragment_pco_abi.shareds) ||
       (texture_layout &&
        (cmd->vertex_count != 36 ||
         cmd->vertex_pco_abi.vertex_outputs != 7 ||
         cmd->vertex_pco_abi.shareds != 32 ||
         cmd->vertex_pco_abi.push_constant_count != 32 ||
         cmd->fragment_pco_abi.coefficients != 16 ||
         cmd->fragment_pco_abi.shareds != 20 ||
         cmd->fragment_pco_abi.push_constant_count != 0 ||
         cmd->varying_output_start != 4 ||
         cmd->varying_output_count != 3 ||
         cmd->fragment_position_count != 4 ||
         cmd->fragment_varying_start != 4 ||
         cmd->fragment_varying_count != 12))) {
      snprintf(error, error_size,
               "draw PCO triangles has incompatible PCO ABI metadata: "
               "vs_outputs=%u (pos=%u+var=%u) vs_shared=%u "
               "fs_coeffs=%u (pos=%u+var=%u) fs_shared=%u "
               "vs_temps=%u fs_temps=%u lit_mesh=%d cond=%d",
               cmd->vertex_pco_abi.vertex_outputs,
               cmd->position_output_count, cmd->varying_output_count,
               cmd->vertex_pco_abi.shareds,
               cmd->fragment_pco_abi.coefficients,
               cmd->fragment_position_count, cmd->fragment_varying_count,
               cmd->fragment_pco_abi.shareds,
               cmd->vertex_pco_abi.temps, cmd->fragment_pco_abi.temps,
               lit_mesh_layout ? 1 : 0, conditionals_layout ? 1 : 0);
      return false;
   }

   uint32_t viewport_bits[3];
   pvrgpu_pco_viewport_bits(cmd->framebuffer_width,
                            cmd->framebuffer_height,
                            viewport_bits);
   const bool ideas_depth_state_matches =
      cmd->depth_format != 0 &&
      ((cmd->depth_enable == 0 && cmd->depth_write == 0 &&
        cmd->depth_func == 0) ||
       (cmd->depth_enable == 1 && cmd->depth_write == 1 &&
        cmd->depth_func == 3));
   const bool depth_state_matches =
      ideas_layout ?
         ideas_depth_state_matches :
         (color_layout ?
          ((cmd->depth_enable == 0 && cmd->depth_write == 0) ||
           (cmd->depth_enable == 1 && cmd->depth_write == 1 &&
            cmd->depth_func == 3)) :
          (cmd->depth_enable == 1 && cmd->depth_write == 1 &&
           cmd->depth_func == 3 && cmd->depth_format != 0));
   if (memcmp(cmd->viewport_scale_bits,
              viewport_bits,
              sizeof(viewport_bits)) != 0 ||
       memcmp(cmd->viewport_translate_bits,
              viewport_bits,
              sizeof(viewport_bits)) != 0) {
      pvrgpu_cmd_error(error, error_size,
                       "draw PCO triangles has incompatible viewport state");
      return false;
   }
   if (cmd->front_ccw != 0 ||
       ((!ideas_layout && !color_layout && cmd->cull_face != 2) ||
        ((ideas_layout || color_layout) && cmd->cull_face != 0 &&
         cmd->cull_face != 2)) ||
       cmd->fill_front != 0 || cmd->fill_back != 0 || cmd->scissor != 0 ||
       cmd->rasterizer_discard != 0 || cmd->multisample != 0 ||
       cmd->half_pixel_center != 1 || cmd->bottom_edge_rule != 0 ||
       cmd->clip_halfz != 0 || cmd->depth_clip_near != 1 ||
       cmd->depth_clip_far != 1 || cmd->depth_clamp != 0 ||
       cmd->sample_mask != UINT32_MAX || cmd->color_mask != 0xf ||
       cmd->blend_enable != 0 || cmd->dither != 1 ||
       cmd->depth_clear_bits != UINT32_C(0x3f800000) ||
       !depth_state_matches) {
      pvrgpu_cmd_error(error, error_size,
                       "draw PCO triangles has incompatible raster/depth state");
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

   int written = fprintf(
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
pvrgpu_write_draw_textured_triangles_command(
   const char *path,
   const struct pvrgpu_draw_textured_triangles_command *cmd,
   char *error,
   size_t error_size)
{
   if (!pvrgpu_cmd_validate_draw_textured_triangles(path,
                                                    cmd,
                                                    error,
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
      "command=draw_textured_triangles\n"
      "case=%s\n"
      "frame=%u\n"
      "framebuffer_width=%u\n"
      "framebuffer_height=%u\n"
      "width=%u\n"
      "height=%u\n"
      "format=%s\n"
      "clear_color_bits=%u,%u,%u,%u\n",
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
      cmd->clear_color_bits[3]);
   for (unsigned vertex = 0;
        written >= 0 &&
        vertex < PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT;
        ++vertex) {
      written = fprintf(file,
                        "vertex%u_bits=%u,%u\n",
                        vertex,
                        cmd->vertex_bits[vertex][0],
                        cmd->vertex_bits[vertex][1]);
   }
   for (unsigned vertex = 0;
        written >= 0 &&
        vertex < PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT;
        ++vertex) {
      written = fprintf(file,
                        "texcoord%u_bits=%u,%u\n",
                        vertex,
                        cmd->texcoord_bits[vertex][0],
                        cmd->texcoord_bits[vertex][1]);
   }
   if (written >= 0) {
      written = fprintf(file,
                        "texture_width=%u\n"
                        "texture_height=%u\n"
                        "texture_rgba8_path=%s\n",
                        cmd->texture_width,
                        cmd->texture_height,
                        cmd->texture_rgba8_path);
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
                               "draw_textured_triangles",
                               cmd->case_name,
                               cmd->frame,
                               cmd->framebuffer_width,
                               cmd->framebuffer_height,
                               cmd->width,
                               cmd->height,
                               cmd->format,
                               cmd->clear_color_bits);
   for (unsigned vertex = 0;
        vertex < PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT;
        ++vertex) {
      api_command.vertex_bits[vertex][0] = cmd->vertex_bits[vertex][0];
      api_command.vertex_bits[vertex][1] = cmd->vertex_bits[vertex][1];
      api_command.texcoord_bits[vertex][0] = cmd->texcoord_bits[vertex][0];
      api_command.texcoord_bits[vertex][1] = cmd->texcoord_bits[vertex][1];
   }
   api_command.texture_width = cmd->texture_width;
   api_command.texture_height = cmd->texture_height;
   api_command.texture_rgba8_path = cmd->texture_rgba8_path;
   return pvrgpu_submit_systemc_api(&api_command, error, error_size);
}

bool
pvrgpu_write_draw_pco_triangles_command(
   const char *path,
   const struct pvrgpu_draw_pco_triangles_command *cmd,
   char *error,
   size_t error_size)
{
   if (!pvrgpu_cmd_validate_draw_pco_triangles(path,
                                               cmd,
                                               error,
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
      "command=draw_pco_triangles\n"
      "case=%s\n"
      "frame=%u\n"
      "framebuffer_width=%u\n"
      "framebuffer_height=%u\n"
      "width=%u\n"
      "height=%u\n"
      "format=%s\n"
      "clear_color_bits=%u,%u,%u,%u\n"
      "raw_vertex_data_size=%zu\n"
      "vertex_stride=%u\n"
      "vertex_count=%u\n"
      "first_vertex=%u\n"
      "instance_count=%u\n"
      "primitive_mode=%u\n"
      "indexed=%u\n"
      "draw_count=%u\n"
      "ia_vertices=%u\n"
      "ia_primitives=%u\n"
      "vs_invocations=%u\n"
      "gs_invocations=%u\n"
      "gs_primitives=%u\n"
      "clip_invocations=%u\n"
      "clip_primitives=%u\n"
      "hs_invocations=%u\n"
      "ds_invocations=%u\n"
      "cs_invocations=%u\n"
      "ps_invocations=%" PRIu64 "\n"
      "setup_triangles=%u\n"
      "semantic_texel_fetches=%" PRIu64 "\n"
      "vertex_pco_size=%zu\n"
      "fragment_pco_size=%zu\n"
      "vertex_shared_count=%zu\n",
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
      cmd->raw_vertex_data_size,
      cmd->vertex_stride,
      cmd->vertex_count,
      cmd->first_vertex,
      cmd->instance_count,
      cmd->primitive_mode,
      cmd->indexed,
      cmd->draw_count,
      cmd->ia_vertices,
      cmd->ia_primitives,
      cmd->vs_invocations,
      cmd->gs_invocations,
      cmd->gs_primitives,
      cmd->clip_invocations,
      cmd->clip_primitives,
      cmd->hs_invocations,
      cmd->ds_invocations,
      cmd->cs_invocations,
      cmd->ps_invocations,
      cmd->setup_triangles,
      cmd->semantic_texel_fetches,
      cmd->vertex_pco_size,
      cmd->fragment_pco_size,
      cmd->vertex_shared_count);
   if (written >= 0)
      written = fprintf(file, "vertex_shared_words=");
   for (size_t word = 0; written >= 0 && word < cmd->vertex_shared_count;
        ++word) {
      written = fprintf(file,
                        "%s%u",
                        word ? "," : "",
                        cmd->vertex_shared[word]);
   }
   if (written >= 0) {
      written = fprintf(file,
                        "\nfragment_shared_count=%zu\n"
                        "fragment_shared_words=",
                        cmd->fragment_shared_count);
   }
   for (size_t word = 0; written >= 0 && word < cmd->fragment_shared_count;
        ++word) {
      written = fprintf(file,
                        "%s%u",
                        word ? "," : "",
                        cmd->fragment_shared[word]);
   }
   if (written >= 0 && cmd->sampled_texture_count == 1) {
      written = fprintf(
         file,
         "\nsampled_texture_count=1\n"
         "sampled_texture_bytes_size=%zu\n"
         "sampled_texture_width=%u\n"
         "sampled_texture_height=%u\n"
         "sampled_texture_row_pitch=%u\n"
         "sampled_texture_format=%s\n"
         "sampled_texture_mip_count=%u",
         cmd->sampled_texture_bytes_size,
         cmd->sampled_texture_width,
         cmd->sampled_texture_height,
         cmd->sampled_texture_row_pitch,
         cmd->sampled_texture_format,
         cmd->sampled_texture_mip_count);
   }
   if (written >= 0) {
      written = fprintf(
         file,
      "\n"
      "vertex_pco_abi=%u,%u,%u,%u,%u,%u,%u,%u\n"
      "fragment_pco_abi=%u,%u,%u,%u,%u,%u,%u,%u\n"
      "position_linkage=%u,%u,%u,%u\n"
      "varying_linkage=%u,%u,%u,%u\n"
      "viewport_scale_bits=%u,%u,%u\n"
      "viewport_translate_bits=%u,%u,%u\n"
      "raster_state=%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n"
      "sample_mask=%u\n"
      "color_state=%u,%u,%u\n"
      "depth_state=%u,%u,%u,%u,%u\n",
      cmd->vertex_pco_abi.temps,
      cmd->vertex_pco_abi.vertex_inputs,
      cmd->vertex_pco_abi.vertex_outputs,
      cmd->vertex_pco_abi.coefficients,
      cmd->vertex_pco_abi.shareds,
      cmd->vertex_pco_abi.push_constant_start,
      cmd->vertex_pco_abi.push_constant_count,
      cmd->vertex_pco_abi.entry_offset,
      cmd->fragment_pco_abi.temps,
      cmd->fragment_pco_abi.vertex_inputs,
      cmd->fragment_pco_abi.vertex_outputs,
      cmd->fragment_pco_abi.coefficients,
      cmd->fragment_pco_abi.shareds,
      cmd->fragment_pco_abi.push_constant_start,
      cmd->fragment_pco_abi.push_constant_count,
      cmd->fragment_pco_abi.entry_offset,
      cmd->position_output_start,
      cmd->position_output_count,
      cmd->fragment_position_start,
      cmd->fragment_position_count,
      cmd->varying_output_start,
      cmd->varying_output_count,
      cmd->fragment_varying_start,
      cmd->fragment_varying_count,
      cmd->viewport_scale_bits[0],
      cmd->viewport_scale_bits[1],
      cmd->viewport_scale_bits[2],
      cmd->viewport_translate_bits[0],
      cmd->viewport_translate_bits[1],
      cmd->viewport_translate_bits[2],
      cmd->front_ccw,
      cmd->cull_face,
      cmd->fill_front,
      cmd->fill_back,
      cmd->scissor,
      cmd->rasterizer_discard,
      cmd->multisample,
      cmd->half_pixel_center,
      cmd->bottom_edge_rule,
      cmd->clip_halfz,
      cmd->depth_clip_near,
      cmd->depth_clip_far,
      cmd->depth_clamp,
      cmd->sample_mask,
      cmd->color_mask,
      cmd->blend_enable,
      cmd->dither,
      cmd->depth_enable,
      cmd->depth_write,
      cmd->depth_func,
      cmd->depth_clear_bits,
      cmd->depth_format);
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
                               "draw_pco_triangles",
                               cmd->case_name,
                               cmd->frame,
                               cmd->framebuffer_width,
                               cmd->framebuffer_height,
                               cmd->width,
                               cmd->height,
                               cmd->format,
                               cmd->clear_color_bits);
   api_command.raw_vertex_data = cmd->raw_vertex_data;
   api_command.raw_vertex_data_size = cmd->raw_vertex_data_size;
   api_command.vertex_stride = cmd->vertex_stride;
   api_command.vertex_count = cmd->vertex_count;
   api_command.first_vertex = cmd->first_vertex;
   api_command.instance_count = cmd->instance_count;
   api_command.primitive_mode = cmd->primitive_mode;
   api_command.indexed = cmd->indexed;
   api_command.draw_count = cmd->draw_count;
   api_command.ia_vertices = cmd->ia_vertices;
   api_command.ia_primitives = cmd->ia_primitives;
   api_command.vs_invocations = cmd->vs_invocations;
   api_command.gs_invocations = cmd->gs_invocations;
   api_command.gs_primitives = cmd->gs_primitives;
   api_command.clip_invocations = cmd->clip_invocations;
   api_command.clip_primitives = cmd->clip_primitives;
   api_command.hs_invocations = cmd->hs_invocations;
   api_command.ds_invocations = cmd->ds_invocations;
   api_command.cs_invocations = cmd->cs_invocations;
   api_command.ps_invocations = cmd->ps_invocations;
   api_command.setup_triangles = cmd->setup_triangles;
   api_command.semantic_texel_fetches = cmd->semantic_texel_fetches;
   api_command.vertex_pco = cmd->vertex_pco;
   api_command.vertex_pco_size = cmd->vertex_pco_size;
   api_command.fragment_pco = cmd->fragment_pco;
   api_command.fragment_pco_size = cmd->fragment_pco_size;
   api_command.vertex_shared = cmd->vertex_shared;
   api_command.vertex_shared_count = cmd->vertex_shared_count;
   api_command.fragment_shared = cmd->fragment_shared;
   api_command.fragment_shared_count = cmd->fragment_shared_count;
   api_command.sampled_texture_count = cmd->sampled_texture_count;
   api_command.sampled_texture_bytes = cmd->sampled_texture_bytes;
   api_command.sampled_texture_bytes_size = cmd->sampled_texture_bytes_size;
   api_command.sampled_texture_width = cmd->sampled_texture_width;
   api_command.sampled_texture_height = cmd->sampled_texture_height;
   api_command.sampled_texture_row_pitch = cmd->sampled_texture_row_pitch;
   api_command.sampled_texture_format = cmd->sampled_texture_format;
   api_command.sampled_texture_mip_count = cmd->sampled_texture_mip_count;
   pvrgpu_systemc_copy_pco_stage_abi(&api_command.vertex_pco_abi,
                                     &cmd->vertex_pco_abi);
   pvrgpu_systemc_copy_pco_stage_abi(&api_command.fragment_pco_abi,
                                     &cmd->fragment_pco_abi);
   api_command.position_output_start = cmd->position_output_start;
   api_command.position_output_count = cmd->position_output_count;
   api_command.fragment_position_start = cmd->fragment_position_start;
   api_command.fragment_position_count = cmd->fragment_position_count;
   api_command.varying_output_start = cmd->varying_output_start;
   api_command.varying_output_count = cmd->varying_output_count;
   api_command.fragment_varying_start = cmd->fragment_varying_start;
   api_command.fragment_varying_count = cmd->fragment_varying_count;
   memcpy(api_command.viewport_scale_bits,
          cmd->viewport_scale_bits,
          sizeof(api_command.viewport_scale_bits));
   memcpy(api_command.viewport_translate_bits,
          cmd->viewport_translate_bits,
          sizeof(api_command.viewport_translate_bits));
   api_command.front_ccw = cmd->front_ccw;
   api_command.cull_face = cmd->cull_face;
   api_command.fill_front = cmd->fill_front;
   api_command.fill_back = cmd->fill_back;
   api_command.scissor = cmd->scissor;
   api_command.rasterizer_discard = cmd->rasterizer_discard;
   api_command.multisample = cmd->multisample;
   api_command.half_pixel_center = cmd->half_pixel_center;
   api_command.bottom_edge_rule = cmd->bottom_edge_rule;
   api_command.clip_halfz = cmd->clip_halfz;
   api_command.depth_clip_near = cmd->depth_clip_near;
   api_command.depth_clip_far = cmd->depth_clip_far;
   api_command.depth_clamp = cmd->depth_clamp;
   api_command.sample_mask = cmd->sample_mask;
   api_command.color_mask = cmd->color_mask;
   api_command.blend_enable = cmd->blend_enable;
   api_command.dither = cmd->dither;
   api_command.depth_enable = cmd->depth_enable;
   api_command.depth_write = cmd->depth_write;
   api_command.depth_func = cmd->depth_func;
   api_command.depth_clear_bits = cmd->depth_clear_bits;
   api_command.depth_format = cmd->depth_format;
   return pvrgpu_submit_systemc_api(&api_command, error, error_size);
}

bool
pvrgpu_write_draw_pco_sequence_command(
   const char *path,
   const struct pvrgpu_systemc_driver_command *cmd,
   char *error,
   size_t error_size)
{
   if (!cmd || cmd->version != PVRGPU_SYSTEMC_API_VERSION ||
       !cmd->command || strcmp(cmd->command, "draw_pco_sequence") != 0 ||
       !cmd->case_name || cmd->case_name[0] == '\0' ||
       !cmd->format || cmd->format[0] == '\0' ||
       !pvrgpu_cmd_format_supported(cmd->format) ||
       cmd->frame == 0 || cmd->framebuffer_width == 0 ||
       cmd->framebuffer_height == 0 || cmd->width == 0 || cmd->height == 0 ||
       cmd->pco_sequence_command_count == 0 ||
       cmd->pco_sequence_command_count >
          PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_COMMANDS ||
       !cmd->pco_sequence_commands ||
       cmd->pco_sequence_texture_count >
          PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_TEXTURES ||
       (cmd->pco_sequence_texture_count != 0 &&
        !cmd->pco_sequence_textures)) {
      pvrgpu_cmd_error(error, error_size,
                       "invalid API-v6 draw PCO sequence command");
      return false;
   }
   if (!path || path[0] == '\0') {
      pvrgpu_cmd_error(error, error_size,
                       "missing draw PCO sequence command path");
      return false;
   }

   for (uint32_t ordinal = 0;
        ordinal < cmd->pco_sequence_command_count;
        ++ordinal) {
      const struct pvrgpu_systemc_driver_command *nested =
         &cmd->pco_sequence_commands[ordinal];
      if (nested->version != PVRGPU_SYSTEMC_API_VERSION ||
          !nested->command ||
          strcmp(nested->command, "draw_pco_triangles") != 0 ||
          !nested->case_name || strcmp(nested->case_name, cmd->case_name) != 0 ||
          nested->pco_sequence_command_count != 0 ||
          nested->pco_sequence_commands ||
          nested->pco_sequence_texture_count != 0 ||
          nested->pco_sequence_textures) {
         pvrgpu_cmd_error(error, error_size,
                          "invalid nested API-v6 PCO sequence draw");
         return false;
      }
   }

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
      "command=draw_pco_sequence\n"
      "case=%s\n"
      "frame=%u\n"
      "framebuffer_width=%u\n"
      "framebuffer_height=%u\n"
      "width=%u\n"
      "height=%u\n"
      "format=%s\n"
      "clear_color_bits=%u,%u,%u,%u\n"
      "draw_count=%u\n"
      "ia_vertices=%u\n"
      "ia_primitives=%u\n"
      "vs_invocations=%u\n"
      "clip_invocations=%u\n"
      "clip_primitives=%u\n"
      "ps_invocations=%" PRIu64 "\n"
      "setup_triangles=%u\n"
      "semantic_texel_fetches=%" PRIu64 "\n"
      "pco_sequence_command_count=%u\n"
      "pco_sequence_texture_count=%u\n",
      cmd->schema && cmd->schema[0] ? cmd->schema :
                                      PVRGPU_DRIVER_COMMAND_SCHEMA,
      cmd->producer && cmd->producer[0] ? cmd->producer :
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
      cmd->ia_vertices,
      cmd->ia_primitives,
      cmd->vs_invocations,
      cmd->clip_invocations,
      cmd->clip_primitives,
      cmd->ps_invocations,
      cmd->setup_triangles,
      cmd->semantic_texel_fetches,
      cmd->pco_sequence_command_count,
      cmd->pco_sequence_texture_count);
   const int close_status = fclose(file);
   if (written < 0 || close_status != 0) {
      if (error && error_size != 0) {
         snprintf(error, error_size, "failed to write driver command: %s",
                  strerror(errno));
      }
      return false;
   }

   return pvrgpu_submit_systemc_api(cmd, error, error_size);
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
