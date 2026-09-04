#include "model_runner.h"
#include "pco_sequence_profiles.h"
#include "pvrgpu_systemc_api.h"
#include "shader/pco_iss.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::mutex g_bridge_mutex;

struct PendingSubmit {
  pvrgpu::stub::Options options;
  std::string jsonl_path;
  std::string stderr_path;
  bool valid = false;
  bool executed = false;
};

PendingSubmit g_pending_submit;
bool g_atexit_registered = false;

bool IsIdeasPcoSequenceCase(const char *case_name) {
  if (!case_name)
    return false;
  const std::string_view name(case_name);
  return name == "ideas" || name.rfind("ideas.", 0) == 0 ||
         name.find(".ideas.") != std::string_view::npos;
}

template <typename Command>
bool IdeasDepthStateMatchesOrdinal(const Command &command,
                                   std::size_t ordinal) {
  const bool depth_enabled =
      ordinal >= pvrgpu::stub::kDriverPcoIdeasDepthEnabledFirstCommand &&
      ordinal < pvrgpu::stub::kDriverPcoIdeasDepthEnabledEndCommand;
  return command.depth_clear_bits == UINT32_C(0x3f800000) &&
         command.depth_format != 0 &&
         command.depth_enable == (depth_enabled ? 1U : 0U) &&
         command.depth_write == (depth_enabled ? 1U : 0U) &&
         command.depth_func == (depth_enabled ? 3U : 0U);
}

template <typename Command>
bool IdeasDepthStateIsSupported(const Command &command) {
  return IdeasDepthStateMatchesOrdinal(command, 0U) ||
         IdeasDepthStateMatchesOrdinal(
             command,
             pvrgpu::stub::kDriverPcoIdeasDepthEnabledFirstCommand);
}

bool IsIdeasPcoSequenceRoot(
    const pvrgpu::stub::DriverCommand &command) {
  return command.enabled && command.command == "draw_pco_triangles" &&
         IsIdeasPcoSequenceCase(command.test_case.c_str()) &&
         command.draw_count ==
             pvrgpu::stub::kDriverPcoIdeasSequenceCommands &&
         command.ia_vertices != 0U &&
         command.ia_primitives != 0U && command.vs_invocations != 0U &&
         command.clip_invocations != 0U &&
         IdeasDepthStateMatchesOrdinal(command, 0U);
}

void CopyError(char *error, std::size_t error_size, const std::string &message) {
  if (!error || error_size == 0)
    return;
  const std::size_t count = message.size() < error_size - 1 ? message.size()
                                                            : error_size - 1;
  for (std::size_t i = 0; i < count; ++i)
    error[i] = message[i];
  error[count] = '\0';
}

template <typename Abi>
bool PcoStageAbiMatches(const Abi &actual,
                        const pvrgpu::stub::DriverPcoStageAbi &expected) {
  return actual.temps == expected.temps &&
         actual.vertex_inputs == expected.vertex_inputs &&
         actual.vertex_outputs == expected.vertex_outputs &&
         actual.coefficients == expected.coefficients &&
         actual.shareds == expected.shareds &&
         actual.push_constant_start == expected.push_constant_start &&
         actual.push_constant_count == expected.push_constant_count &&
         actual.entry_offset == expected.entry_offset;
}

template <typename Abi>
bool PcoStageAbiIsBounded(const Abi &abi, bool allow_zero_temps = false,
                          bool fragment_stage = false) {
  const std::size_t maximum_shared =
      fragment_stage ? pvrgpu::stub::kPcoMaximumFragmentSharedCount
                     : pvrgpu::stub::kPcoMaximumVertexSharedCount;
  return (allow_zero_temps || abi.temps != 0) &&
         abi.temps <= pvrgpu::stub::kPcoTemporaryCount &&
         abi.vertex_inputs <= pvrgpu::stub::kPcoVertexInputCount &&
         abi.vertex_outputs <= pvrgpu::stub::kPcoVertexOutputCount &&
         abi.coefficients <=
             pvrgpu::stub::kPcoMaximumVaryingCoefficientCount &&
         abi.shareds <= maximum_shared &&
         abi.push_constant_start <= abi.shareds &&
         abi.push_constant_count <= abi.shareds - abi.push_constant_start &&
         abi.entry_offset == 0;
}

template <typename Abi>
std::string PcoStageAbiText(const Abi &abi) {
  std::ostringstream text;
  text << abi.temps << ',' << abi.vertex_inputs << ',' << abi.vertex_outputs
       << ',' << abi.coefficients << ',' << abi.shareds << ','
       << abi.push_constant_start << ',' << abi.push_constant_count << ','
       << abi.entry_offset;
  return text.str();
}

bool RawFloatVerticesAreFinite(const std::uint8_t *data,
                               std::uint64_t vertex_count,
                               std::uint32_t stride,
                               std::uint32_t component_count) {
  if (!data || component_count == 0 ||
      component_count * sizeof(float) > stride) {
    return false;
  }
  for (std::uint64_t vertex = 0; vertex < vertex_count; ++vertex) {
    const std::size_t offset = static_cast<std::size_t>(vertex * stride);
    for (std::uint32_t component = 0; component < component_count;
         ++component) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, data + offset + component * sizeof(bits),
                  sizeof(bits));
      float value = 0.0F;
      std::memcpy(&value, &bits, sizeof(value));
      if (!std::isfinite(value))
        return false;
    }
  }
  return true;
}

// Non-indexed triangle topologies the submitter can expand into a triangle
// list: a whole-triangle list, or a strip/fan of three or more vertices.
// An indexed draw carries a whole number of 8/16/32-bit indices covering
// first_index + index_count; a non-indexed one carries no index payload.
bool DriverPcoIndexPayloadIsValid(
    const pvrgpu_systemc_driver_command &source) {
  if (source.indexed == 0) {
    return source.raw_index_data == nullptr &&
           source.raw_index_data_size == 0 && source.index_size == 0 &&
           source.index_count == 0 && source.first_index == 0 &&
           source.base_vertex == 0;
  }
  if (source.index_size != 1 && source.index_size != 2 &&
      source.index_size != 4)
    return false;
  if (!source.raw_index_data || source.index_count == 0)
    return false;
  const std::uint64_t index_end =
      static_cast<std::uint64_t>(source.first_index) + source.index_count;
  return index_end * source.index_size == source.raw_index_data_size;
}

bool DriverPcoArrayTopologyIsExpandable(std::uint32_t primitive_mode,
                                        std::uint32_t vertex_count) {
  switch (primitive_mode) {
    case 0U:  // points
      return vertex_count >= 1U;
    case 1U:  // lines
      return vertex_count >= 2U && vertex_count % 2U == 0U;
    case 2U:  // line loop
    case 3U:  // line strip
      return vertex_count >= 2U;
    case 4U:  // triangles
      return vertex_count >= 3U && vertex_count % 3U == 0U;
    case 5U:  // triangle strip
    case 6U:  // triangle fan
      return vertex_count >= 3U;
    default:
      return false;
  }
}

bool PcoSingleDrawResolutionSupported(std::uint32_t framebuffer_width,
                                      std::uint32_t framebuffer_height,
                                      std::uint32_t width,
                                      std::uint32_t height) {
  return width == framebuffer_width && height == framebuffer_height &&
         ((framebuffer_width == 80 && framebuffer_height == 60) ||
          (framebuffer_width == 800 && framebuffer_height == 600));
}

std::array<std::uint32_t, 3> PcoViewportBits(
    std::uint32_t framebuffer_width, std::uint32_t framebuffer_height) {
  const std::array<float, 3> values = {
      static_cast<float>(framebuffer_width) * 0.5F,
      static_cast<float>(framebuffer_height) * 0.5F,
      0.5F,
  };
  std::array<std::uint32_t, 3> bits{};
  static_assert(sizeof(values) == sizeof(bits));
  std::memcpy(bits.data(), values.data(), sizeof(bits));
  return bits;
}

void CopyPcoPayloadFields(
    const pvrgpu_systemc_driver_command &source,
    pvrgpu::stub::DriverCommand *destination) {
  destination->raw_vertex_data.assign(
      source.raw_vertex_data,
      source.raw_vertex_data + source.raw_vertex_data_size);
  if (source.raw_index_data && source.raw_index_data_size != 0) {
    destination->raw_index_data.assign(
        source.raw_index_data,
        source.raw_index_data + source.raw_index_data_size);
  } else {
    destination->raw_index_data.clear();
  }
  destination->declared_raw_index_data_size = source.raw_index_data_size;
  destination->index_size = source.index_size;
  destination->index_count = source.index_count;
  destination->first_index = source.first_index;
  destination->base_vertex = source.base_vertex;
  destination->vertex_pco.assign(source.vertex_pco,
                                 source.vertex_pco + source.vertex_pco_size);
  destination->fragment_pco.assign(
      source.fragment_pco,
      source.fragment_pco + source.fragment_pco_size);
  destination->vertex_shared.clear();
  if (source.vertex_shared_count != 0) {
    destination->vertex_shared.assign(
        source.vertex_shared,
        source.vertex_shared + source.vertex_shared_count);
  }
  destination->fragment_shared.clear();
  if (source.fragment_shared_count != 0) {
    destination->fragment_shared.assign(
        source.fragment_shared,
        source.fragment_shared + source.fragment_shared_count);
  }
  destination->sampled_texture_count = source.sampled_texture_count;
  destination->sampled_texture_bytes.clear();
  if (source.sampled_texture_count == 1 && source.sampled_texture_bytes) {
    destination->sampled_texture_bytes.assign(
        source.sampled_texture_bytes,
        source.sampled_texture_bytes + source.sampled_texture_bytes_size);
  }
  destination->declared_sampled_texture_bytes_size =
      source.sampled_texture_bytes_size;
  destination->sampled_texture_width = source.sampled_texture_width;
  destination->sampled_texture_height = source.sampled_texture_height;
  destination->sampled_texture_row_pitch = source.sampled_texture_row_pitch;
  destination->sampled_texture_format =
      source.sampled_texture_format ? source.sampled_texture_format : "";
  destination->sampled_texture_mip_count = source.sampled_texture_mip_count;
  destination->declared_raw_vertex_data_size = source.raw_vertex_data_size;
  destination->declared_vertex_pco_size = source.vertex_pco_size;
  destination->declared_fragment_pco_size = source.fragment_pco_size;
  destination->vertex_stride = source.vertex_stride;
  destination->vertex_count = source.vertex_count;
  destination->first_vertex = source.first_vertex;
  destination->instance_count = source.instance_count;
  destination->primitive_mode = source.primitive_mode;
  destination->indexed = source.indexed;
  destination->vertex_pco_abi = {
      source.vertex_pco_abi.temps,
      source.vertex_pco_abi.vertex_inputs,
      source.vertex_pco_abi.vertex_outputs,
      source.vertex_pco_abi.coefficients,
      source.vertex_pco_abi.shareds,
      source.vertex_pco_abi.push_constant_start,
      source.vertex_pco_abi.push_constant_count,
      source.vertex_pco_abi.entry_offset,
  };
  destination->fragment_pco_abi = {
      source.fragment_pco_abi.temps,
      source.fragment_pco_abi.vertex_inputs,
      source.fragment_pco_abi.vertex_outputs,
      source.fragment_pco_abi.coefficients,
      source.fragment_pco_abi.shareds,
      source.fragment_pco_abi.push_constant_start,
      source.fragment_pco_abi.push_constant_count,
      source.fragment_pco_abi.entry_offset,
  };
  destination->position_output_start = source.position_output_start;
  destination->position_output_count = source.position_output_count;
  destination->fragment_position_start = source.fragment_position_start;
  destination->fragment_position_count = source.fragment_position_count;
  destination->varying_output_start = source.varying_output_start;
  destination->varying_output_count = source.varying_output_count;
  destination->fragment_varying_start = source.fragment_varying_start;
  destination->fragment_varying_count = source.fragment_varying_count;
  std::copy_n(source.viewport_scale_bits, 3,
              destination->viewport_scale_bits.begin());
  std::copy_n(source.viewport_translate_bits, 3,
              destination->viewport_translate_bits.begin());
  destination->front_ccw = source.front_ccw;
  destination->cull_face = source.cull_face;
  destination->fill_front = source.fill_front;
  destination->fill_back = source.fill_back;
  destination->scissor = source.scissor;
  destination->rasterizer_discard = source.rasterizer_discard;
  destination->multisample = source.multisample;
  destination->half_pixel_center = source.half_pixel_center;
  destination->bottom_edge_rule = source.bottom_edge_rule;
  destination->clip_halfz = source.clip_halfz;
  destination->depth_clip_near = source.depth_clip_near;
  destination->depth_clip_far = source.depth_clip_far;
  destination->depth_clamp = source.depth_clamp;
  destination->sample_mask = source.sample_mask;
  destination->color_mask = source.color_mask;
  destination->blend_enable = source.blend_enable;
  destination->blend_rgb_equation = source.blend_rgb_equation;
  destination->blend_alpha_equation = source.blend_alpha_equation;
  destination->blend_source_rgb_factor = source.blend_source_rgb_factor;
  destination->blend_destination_rgb_factor =
      source.blend_destination_rgb_factor;
  destination->blend_source_alpha_factor = source.blend_source_alpha_factor;
  destination->blend_destination_alpha_factor =
      source.blend_destination_alpha_factor;
  destination->dither = source.dither;
  destination->depth_enable = source.depth_enable;
  destination->depth_write = source.depth_write;
  destination->depth_func = source.depth_func;
  destination->depth_clear_bits = source.depth_clear_bits;
  destination->depth_format = source.depth_format;
  destination->color_attachment_source_command_index =
      source.color_attachment_source_command_index;
  destination->depth_attachment_source_command_index =
      source.depth_attachment_source_command_index;
}

bool SetMemoryMode(const char *text, pvrgpu::stub::Options *options,
                   std::string *error) {
  if (!options || !error)
    return false;
  const std::string mode = text && text[0] ? text : "cache";
  if (mode == "direct") {
    options->memory_mode = pvrgpu::stub::MemoryMode::kDirect;
    options->cache_bypass = false;
    return true;
  }
  if (mode == "bypass") {
    options->memory_mode = pvrgpu::stub::MemoryMode::kBypass;
    options->cache_bypass = true;
    return true;
  }
  if (mode == "cache") {
    options->memory_mode = pvrgpu::stub::MemoryMode::kCache;
    options->cache_bypass = false;
    return true;
  }
  *error = "invalid memory mode for SystemC API: " + mode;
  return false;
}

bool CopyTextureSidecarBytes(
    const pvrgpu_systemc_driver_command &source,
    std::vector<std::uint8_t> *destination, std::string *error) {
  if (!destination || !error)
    return false;
  if (source.texture_width == 0 || source.texture_height == 0 ||
      source.texture_width > 16384U || source.texture_height > 16384U ||
      !source.texture_rgba8_path || !source.texture_rgba8_path[0]) {
    *error = "invalid SystemC API textured-triangle sidecar metadata";
    return false;
  }
  const std::uint64_t byte_count =
      static_cast<std::uint64_t>(source.texture_width) *
      source.texture_height * 4U;
  if (byte_count == 0 ||
      byte_count > std::numeric_limits<std::uint32_t>::max()) {
    *error = "SystemC API texture sidecar size is unsupported";
    return false;
  }
  std::ifstream input(source.texture_rgba8_path, std::ios::binary);
  if (!input) {
    *error = "cannot open SystemC API texture sidecar";
    return false;
  }
  destination->assign(static_cast<std::size_t>(byte_count), 0);
  input.read(reinterpret_cast<char *>(destination->data()),
             static_cast<std::streamsize>(destination->size()));
  if (input.gcount() != static_cast<std::streamsize>(destination->size())) {
    *error = "SystemC API texture sidecar is truncated";
    return false;
  }
  char extra = 0;
  if (input.read(&extra, 1) || input.gcount() != 0) {
    *error = "SystemC API texture sidecar has extra bytes";
    return false;
  }
  return true;
}

bool CopyPcoTrianglePayload(
    const pvrgpu_systemc_driver_command &source,
    pvrgpu::stub::DriverCommand *destination, std::string *error) {
  if (!destination || !error)
    return false;
  if (!PcoSingleDrawResolutionSupported(
          source.framebuffer_width, source.framebuffer_height,
          source.width, source.height)) {
    *error = "SystemC API PCO triangle resolution requires a "
             "framebuffer-sized 80x60 or 800x600 viewport";
    return false;
  }
  const std::array<std::uint32_t, 3> viewport_bits =
      PcoViewportBits(source.framebuffer_width, source.framebuffer_height);
  const bool ideas_sequence = IsIdeasPcoSequenceCase(source.case_name);
  const bool conditionals_layout =
      source.vertex_stride == pvrgpu::stub::kDriverPcoPositionVertexStride;
  // Untextured position/colour layout: six floats for a vec2 position, eight
  // for a vec4 one.
  const bool color_layout =
      (source.vertex_stride ==
           pvrgpu::stub::kDriverPcoPositionNormalVertexStride ||
       source.vertex_stride == 8U * sizeof(float)) &&
      source.vertex_pco_abi.shareds == 0;
  const bool lit_mesh_layout =
      source.vertex_stride ==
          pvrgpu::stub::kDriverPcoPositionNormalVertexStride &&
      !color_layout;
  const bool texture_layout =
      !ideas_sequence &&
      source.vertex_stride ==
          pvrgpu::stub::kDriverPcoPositionNormalTexcoordVertexStride;
  const bool ideas_position_layout =
      ideas_sequence && source.vertex_stride == 4U * sizeof(float);
  const bool ideas_two_attribute_layout =
      ideas_sequence && source.vertex_stride == 8U * sizeof(float);
  const bool ideas_layout =
      ideas_position_layout || ideas_two_attribute_layout;
  const bool ideas_topology =
      ideas_sequence &&
      ((source.primitive_mode == 5U &&
        (source.vertex_count == 18U || source.vertex_count == 26U)) ||
       (source.primitive_mode == 6U && source.vertex_count == 12U));
  const std::uint64_t end_vertex =
      static_cast<std::uint64_t>(source.first_vertex) + source.vertex_count;
  const std::uint64_t expected_vertex_bytes =
      end_vertex * static_cast<std::uint64_t>(source.vertex_stride);
  if (!source.format ||
      std::string(source.format) != "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      source.clear_color_bits[0] != 0 || source.clear_color_bits[1] != 0 ||
      source.clear_color_bits[2] != 0 ||
      source.clear_color_bits[3] != UINT32_C(0x3f800000) ||
      (!conditionals_layout && !lit_mesh_layout && !texture_layout &&
       !ideas_layout && !color_layout) ||
      source.vertex_count == 0 ||
      (!ideas_sequence &&
       !DriverPcoArrayTopologyIsExpandable(source.primitive_mode,
                                           source.indexed != 0
                                               ? source.index_count
                                               : source.vertex_count)) ||
      source.first_vertex != 0 ||
      source.instance_count != 1 || (ideas_sequence && !ideas_topology) ||
      !DriverPcoIndexPayloadIsValid(source) || expected_vertex_bytes == 0 ||
      expected_vertex_bytes > std::numeric_limits<std::uint32_t>::max() ||
      source.raw_vertex_data_size != expected_vertex_bytes ||
      !source.raw_vertex_data ||
      !RawFloatVerticesAreFinite(source.raw_vertex_data, end_vertex,
                                 source.vertex_stride,
                                 ideas_layout
                                     ? source.vertex_stride / sizeof(float)
                                     : texture_layout
                                           ? 8U
                                           : ((lit_mesh_layout || color_layout)
                                                  ? 6U
                                                  : 3U))) {
    *error = "invalid SystemC API PCO triangle VBO/topology payload";
    return false;
  }

  if (!texture_layout) {
    if (source.sampled_texture_count != 0 ||
        source.sampled_texture_bytes ||
        source.sampled_texture_bytes_size != 0 ||
        source.sampled_texture_width != 0 ||
        source.sampled_texture_height != 0 ||
        source.sampled_texture_row_pitch != 0 ||
        source.sampled_texture_format ||
        source.sampled_texture_mip_count != 0) {
      *error = "SystemC API PCO untextured profile has sampled texture state";
      return false;
    }
  } else {
    const std::uint64_t tight_row_pitch =
        static_cast<std::uint64_t>(source.sampled_texture_width) * 4U;
    const std::uint64_t expected_texture_bytes =
        static_cast<std::uint64_t>(source.sampled_texture_row_pitch) *
        source.sampled_texture_height;
    if (source.sampled_texture_count != 1 ||
        !source.sampled_texture_bytes ||
        source.sampled_texture_width != pvrgpu::stub::kDriverPcoTextureWidth ||
        source.sampled_texture_height !=
            pvrgpu::stub::kDriverPcoTextureHeight ||
        tight_row_pitch != pvrgpu::stub::kDriverPcoTextureRowPitch ||
        source.sampled_texture_row_pitch != tight_row_pitch ||
        source.sampled_texture_mip_count != 1 ||
        !source.sampled_texture_format ||
        std::string(source.sampled_texture_format) !=
            "PIPE_FORMAT_R8G8B8X8_UNORM" ||
        expected_texture_bytes == 0 ||
        expected_texture_bytes > std::numeric_limits<std::size_t>::max() ||
        expected_texture_bytes != pvrgpu::stub::kDriverPcoTextureBytes ||
        source.sampled_texture_bytes_size != expected_texture_bytes) {
      *error = "SystemC API PCO sampled texture payload mismatch";
      return false;
    }
  }

  const bool common_abi_invalid =
      !PcoStageAbiIsBounded(source.vertex_pco_abi) ||
      !PcoStageAbiIsBounded(source.fragment_pco_abi,
                            ideas_position_layout || color_layout, true) ||
      source.vertex_pco_abi.coefficients != 0 ||
      source.fragment_pco_abi.vertex_inputs != 0 ||
      source.fragment_pco_abi.vertex_outputs != 0 ||
      source.position_output_start != 0 || source.position_output_count != 4 ||
      source.vertex_pco_abi.vertex_outputs !=
          source.position_output_count + source.varying_output_count ||
      (source.varying_output_count != 0 &&
       source.varying_output_start != source.position_output_count) ||
      source.fragment_position_start != 0 ||
      (source.fragment_varying_count != 0 &&
       source.fragment_varying_start != source.fragment_position_count) ||
      source.fragment_pco_abi.coefficients !=
          source.fragment_position_count + source.fragment_varying_count;
  const bool ideas_abi_invalid =
      ideas_sequence &&
      (source.vertex_pco_abi.vertex_inputs !=
           source.vertex_stride / sizeof(float) ||
       source.vertex_pco_abi.push_constant_start != 0 ||
       source.vertex_pco_abi.push_constant_count !=
           source.vertex_pco_abi.shareds ||
       source.fragment_pco_abi.push_constant_start != 0 ||
       source.fragment_pco_abi.push_constant_count !=
           source.fragment_pco_abi.shareds ||
       source.varying_output_count >
           pvrgpu::stub::kDriverPcoMaximumVaryingComponents ||
       (ideas_position_layout
            ? source.vertex_pco_abi.vertex_outputs != 4 ||
                  source.vertex_pco_abi.shareds != 32 ||
                  source.varying_output_start != 0 ||
                  source.varying_output_count != 0 ||
                  source.fragment_position_count != 0 ||
                  source.fragment_varying_start != 0 ||
                  source.fragment_varying_count != 0 ||
                  source.fragment_pco_abi.coefficients != 0 ||
                  (source.fragment_pco_abi.shareds != 0 &&
                   source.fragment_pco_abi.shareds != 4)
            : source.vertex_pco_abi.vertex_outputs != 14 ||
                  source.vertex_pco_abi.shareds != 44 ||
                  source.fragment_pco_abi.shareds != 12 ||
                  source.varying_output_start != 4 ||
                  source.varying_output_count != 10 ||
                  source.fragment_position_count != 4 ||
                  source.fragment_varying_start != 4 ||
                  source.fragment_varying_count != 40 ||
                  source.fragment_pco_abi.coefficients != 44));
  const bool single_abi_invalid =
      !ideas_sequence &&
      ((conditionals_layout &&
        (!PcoStageAbiMatches(source.vertex_pco_abi,
                             pvrgpu::stub::kConditionalsVertexPcoAbi) ||
         !PcoStageAbiMatches(source.fragment_pco_abi,
                             pvrgpu::stub::kConditionalsFragmentPcoAbi) ||
         source.varying_output_start != 0 ||
         source.varying_output_count != 0 ||
         source.fragment_position_count != 0 ||
         source.fragment_varying_start != 0 ||
         source.fragment_varying_count != 0)) ||
       (lit_mesh_layout &&
        (source.vertex_pco_abi.vertex_inputs != 8 ||
         source.vertex_pco_abi.push_constant_count !=
             source.vertex_pco_abi.shareds ||
         source.fragment_pco_abi.push_constant_count !=
             source.fragment_pco_abi.shareds ||
         source.varying_output_start != 4 ||
         source.varying_output_count == 0 ||
         source.varying_output_count >
             pvrgpu::stub::kDriverPcoMaximumVaryingComponents ||
         source.fragment_position_count != 4 ||
         source.fragment_varying_start != 4 ||
         source.fragment_varying_count != source.varying_output_count * 4U)) ||
       (color_layout &&
        (source.vertex_pco_abi.vertex_inputs != 8 ||
         source.vertex_pco_abi.push_constant_count !=
             source.vertex_pco_abi.shareds ||
         source.fragment_pco_abi.push_constant_count !=
             source.fragment_pco_abi.shareds ||
         source.varying_output_start != 4 ||
         source.varying_output_count != 4 ||
         source.fragment_position_count != 4 ||
         source.fragment_varying_start != 4 ||
         source.fragment_varying_count != 16)) ||
       (texture_layout &&
        (source.vertex_count != 36 ||
         source.vertex_pco_abi.vertex_inputs != 12 ||
         source.vertex_pco_abi.vertex_outputs != 7 ||
         source.vertex_pco_abi.shareds != 32 ||
         source.vertex_pco_abi.push_constant_start != 0 ||
         source.vertex_pco_abi.push_constant_count != 32 ||
         source.fragment_pco_abi.coefficients != 16 ||
         source.fragment_pco_abi.shareds != 20 ||
         source.fragment_pco_abi.push_constant_start != 0 ||
         source.fragment_pco_abi.push_constant_count != 0 ||
         source.varying_output_start != 4 ||
         source.varying_output_count != 3 ||
         source.fragment_position_count != 4 ||
         source.fragment_varying_start != 4 ||
         source.fragment_varying_count != 12)));
  const bool viewport_scale_invalid =
      !std::equal(viewport_bits.begin(), viewport_bits.end(),
                  source.viewport_scale_bits);
  const bool viewport_translate_invalid =
      !std::equal(viewport_bits.begin(), viewport_bits.end(),
                  source.viewport_translate_bits);
  if (common_abi_invalid || ideas_abi_invalid || single_abi_invalid ||
      viewport_scale_invalid || viewport_translate_invalid) {
    std::ostringstream detail;
    detail << "SystemC API PCO triangle ABI/viewport metadata mismatch: ";
    if (!PcoStageAbiIsBounded(source.vertex_pco_abi)) {
      detail << "vertex ABI exceeds model bounds (actual="
             << PcoStageAbiText(source.vertex_pco_abi) << ", limits temps="
             << pvrgpu::stub::kPcoTemporaryCount << " vtxin="
             << pvrgpu::stub::kPcoVertexInputCount << " vtxout="
             << pvrgpu::stub::kPcoVertexOutputCount << " coeff="
             << pvrgpu::stub::kPcoMaximumVaryingCoefficientCount
             << " shared=" << pvrgpu::stub::kPcoMaximumVertexSharedCount
             << ')';
    } else if (!PcoStageAbiIsBounded(source.fragment_pco_abi,
                                     ideas_position_layout || color_layout,
                                     true)) {
      detail << "fragment ABI exceeds model bounds (actual="
             << PcoStageAbiText(source.fragment_pco_abi) << ", limits temps="
             << pvrgpu::stub::kPcoTemporaryCount << " vtxin="
             << pvrgpu::stub::kPcoVertexInputCount << " vtxout="
             << pvrgpu::stub::kPcoVertexOutputCount << " coeff="
             << pvrgpu::stub::kPcoMaximumVaryingCoefficientCount
             << " shared=" << pvrgpu::stub::kPcoMaximumFragmentSharedCount
             << ')';
    } else if (common_abi_invalid) {
      detail << "common linkage (vs="
             << PcoStageAbiText(source.vertex_pco_abi) << " fs="
             << PcoStageAbiText(source.fragment_pco_abi) << " position="
             << source.position_output_start << ','
             << source.position_output_count << " varying="
             << source.varying_output_start << ','
             << source.varying_output_count << " fragment_position="
             << source.fragment_position_start << ','
             << source.fragment_position_count << " fragment_varying="
             << source.fragment_varying_start << ','
             << source.fragment_varying_count << ')';
    } else if (ideas_abi_invalid) {
      detail << "Ideas profile contract (layout="
             << (ideas_position_layout ? "position" : "two-attribute")
             << " vs=" << PcoStageAbiText(source.vertex_pco_abi) << " fs="
             << PcoStageAbiText(source.fragment_pco_abi) << " position="
             << source.position_output_start << ','
             << source.position_output_count << " varying="
             << source.varying_output_start << ','
             << source.varying_output_count << " fragment_position="
             << source.fragment_position_start << ','
             << source.fragment_position_count << " fragment_varying="
             << source.fragment_varying_start << ','
             << source.fragment_varying_count << ')';
    } else if (single_abi_invalid) {
      detail << "single-draw profile contract (vs="
             << PcoStageAbiText(source.vertex_pco_abi) << " fs="
             << PcoStageAbiText(source.fragment_pco_abi) << ')';
    } else if (viewport_scale_invalid) {
      detail << "viewport scale=" << source.viewport_scale_bits[0] << ','
             << source.viewport_scale_bits[1] << ','
             << source.viewport_scale_bits[2];
    } else {
      detail << "viewport translate=" << source.viewport_translate_bits[0]
             << ',' << source.viewport_translate_bits[1] << ','
             << source.viewport_translate_bits[2];
    }
    *error = detail.str();
    return false;
  }
  const bool common_raster_invalid =
      source.front_ccw != 0 ||
      source.fill_front != 0 || source.fill_back != 0 ||
      source.scissor != 0 || source.rasterizer_discard != 0 ||
      source.multisample != 0 || source.half_pixel_center != 1 ||
      source.bottom_edge_rule != 0 || source.clip_halfz != 0 ||
      source.depth_clip_near != 1 || source.depth_clip_far != 1 ||
      source.depth_clamp != 0 || source.sample_mask != UINT32_MAX ||
      source.color_mask != 0x0f || source.blend_enable != 0 ||
      source.dither != 1;
  const bool ideas_raster_invalid =
      ideas_sequence &&
      ((source.cull_face != 0 && source.cull_face != 2) ||
       !IdeasDepthStateIsSupported(source));
  const bool single_raster_invalid =
      !ideas_sequence &&
      (color_layout
           ? (source.cull_face != 0 && source.cull_face != 2)
           : (source.cull_face != 2 || source.depth_enable != 1 ||
              source.depth_write != 1 || source.depth_func != 3 ||
              source.depth_clear_bits != UINT32_C(0x3f800000) ||
              source.depth_format == 0));
  if (common_raster_invalid || ideas_raster_invalid ||
      single_raster_invalid) {
    *error = "SystemC API PCO triangle raster/depth metadata mismatch";
    return false;
  }
  if (!source.vertex_pco || !source.fragment_pco ||
      source.vertex_pco_size == 0 || source.fragment_pco_size == 0 ||
      source.vertex_pco_size > pvrgpu::stub::kDriverPcoMaximumBinaryBytes ||
      source.fragment_pco_size >
          pvrgpu::stub::kDriverPcoMaximumBinaryBytes ||
      (source.vertex_shared_count != 0 && !source.vertex_shared) ||
      (source.fragment_shared_count != 0 && !source.fragment_shared) ||
      source.vertex_shared_count != source.vertex_pco_abi.shareds ||
      source.fragment_shared_count != source.fragment_pco_abi.shareds) {
    *error = "SystemC API PCO triangle binary/shared payload mismatch";
    return false;
  }
  if (conditionals_layout) {
    const std::vector<std::uint8_t> &expected_vertex_pco =
        pvrgpu::stub::ConditionalsVertexPcoBinary();
    const std::vector<std::uint8_t> &expected_fragment_pco =
        pvrgpu::stub::ConditionalsFragmentPcoBinary();
    if (source.vertex_pco_size != expected_vertex_pco.size() ||
        source.fragment_pco_size != expected_fragment_pco.size() ||
        !std::equal(expected_vertex_pco.begin(), expected_vertex_pco.end(),
                    source.vertex_pco) ||
        !std::equal(expected_fragment_pco.begin(), expected_fragment_pco.end(),
                    source.fragment_pco)) {
      *error = "SystemC API PCO triangle shader binary/profile mismatch";
      return false;
    }
  }

  CopyPcoPayloadFields(source, destination);
  return true;
}

bool PcoSequenceTailIsEmpty(
    const pvrgpu_systemc_driver_command &command) {
  return command.pco_sequence_command_count == 0 &&
         command.pco_sequence_commands == nullptr &&
         command.pco_sequence_texture_count == 0 &&
         command.pco_sequence_textures == nullptr;
}

bool CopyPcoSequenceDraw(
    const pvrgpu_systemc_driver_command &source,
    std::size_t ordinal, pvrgpu::stub::DriverCommand *destination,
    std::string *error) {
  if (!destination || !error ||
      source.version != PVRGPU_SYSTEMC_API_VERSION ||
      !source.command || std::string_view(source.command) !=
                             "draw_pco_triangles" ||
      !source.case_name || !source.case_name[0] || !source.format ||
      std::string_view(source.format) !=
          "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      !PcoSequenceTailIsEmpty(source)) {
    if (error)
      *error = "SystemC API nested PCO sequence draw header is invalid";
    return false;
  }
  const std::uint64_t end_vertex =
      static_cast<std::uint64_t>(source.first_vertex) + source.vertex_count;
  const bool triangles = source.primitive_mode == 4;
  const bool strip_or_fan =
      source.primitive_mode == 5 || source.primitive_mode == 6;
  const bool line_or_point = source.primitive_mode <= 3;
  if (source.framebuffer_width == 0 || source.framebuffer_height == 0 ||
      source.width == 0 || source.height == 0 ||
      source.width > source.framebuffer_width ||
      source.height > source.framebuffer_height ||
      source.framebuffer_width > 4096 || source.framebuffer_height > 4096 ||
      source.vertex_stride < 2U * sizeof(float) ||
      source.vertex_stride > 256 || source.vertex_stride % sizeof(float) != 0 ||
      source.vertex_count == 0 || source.first_vertex != 0 ||
      source.instance_count != 1 ||
      (!triangles && !strip_or_fan && !line_or_point) ||
      // An indexed draw assembles primitives from its indices.
      !DriverPcoArrayTopologyIsExpandable(source.primitive_mode,
                                          source.indexed != 0
                                              ? source.index_count
                                              : source.vertex_count) ||
      !DriverPcoIndexPayloadIsValid(source) || end_vertex == 0 ||
      end_vertex > std::numeric_limits<std::uint32_t>::max() ||
      end_vertex > std::numeric_limits<std::uint64_t>::max() /
                       source.vertex_stride ||
      source.raw_vertex_data_size != end_vertex * source.vertex_stride ||
      !source.raw_vertex_data ||
      !RawFloatVerticesAreFinite(source.raw_vertex_data, end_vertex,
                                 source.vertex_stride,
                                 source.vertex_stride / sizeof(float))) {
    *error = "SystemC API nested PCO sequence VBO/topology is invalid";
    return false;
  }
  if (!source.vertex_pco || !source.fragment_pco ||
      source.vertex_pco_size == 0 || source.fragment_pco_size == 0 ||
      source.vertex_pco_size > pvrgpu::stub::kDriverPcoMaximumBinaryBytes ||
      source.fragment_pco_size >
          pvrgpu::stub::kDriverPcoMaximumBinaryBytes ||
      (source.vertex_shared_count != 0 && !source.vertex_shared) ||
      (source.fragment_shared_count != 0 && !source.fragment_shared) ||
      /* A pass-through VS forwarding position and colour uses no temps. */
      !PcoStageAbiIsBounded(source.vertex_pco_abi, true) ||
      !PcoStageAbiIsBounded(source.fragment_pco_abi, true, true) ||
      source.vertex_shared_count != source.vertex_pco_abi.shareds ||
      source.fragment_shared_count != source.fragment_pco_abi.shareds ||
      source.vertex_pco_abi.coefficients != 0 ||
      source.fragment_pco_abi.vertex_inputs != 0 ||
      source.fragment_pco_abi.vertex_outputs != 0 ||
      source.position_output_start != 0 ||
      source.position_output_count != 4 ||
      source.vertex_pco_abi.vertex_outputs !=
          source.position_output_count + source.varying_output_count ||
      source.varying_output_start != source.position_output_count ||
      source.varying_output_count == 0 ||
      source.varying_output_count >
          pvrgpu::stub::kDriverPcoMaximumVaryingComponents ||
      source.fragment_position_start != 0 ||
      source.fragment_position_count != 4 ||
      source.fragment_varying_start != source.fragment_position_count ||
      source.fragment_varying_count != source.varying_output_count * 4U ||
      source.fragment_pco_abi.coefficients !=
          source.fragment_position_count + source.fragment_varying_count) {
    *error = "SystemC API nested PCO sequence ABI/payload is invalid";
    return false;
  }
  float depth_clear = 0.0F;
  std::memcpy(&depth_clear, &source.depth_clear_bits, sizeof(depth_clear));
  const bool blend_enums_valid =
      source.blend_rgb_equation <=
          PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_MAX &&
      source.blend_alpha_equation <=
          PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_MAX &&
      source.blend_source_rgb_factor <=
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA &&
      source.blend_destination_rgb_factor <=
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA &&
      source.blend_source_alpha_factor <=
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA &&
      source.blend_destination_alpha_factor <=
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA;
  const bool disabled_blend_is_canonical =
      source.blend_enable != 0 ||
      (source.blend_rgb_equation ==
           PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD &&
       source.blend_alpha_equation ==
           PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD &&
       source.blend_source_rgb_factor ==
           PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE &&
       source.blend_destination_rgb_factor ==
           PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO &&
       source.blend_source_alpha_factor ==
           PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE &&
       source.blend_destination_alpha_factor ==
           PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO);
  const bool depth_format_supported =
      (source.depth_format == 0 && source.depth_enable == 0 &&
       source.depth_write == 0 &&
       source.depth_attachment_source_command_index ==
           PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR) ||
      source.depth_format ==
          pvrgpu::stub::kDriverPcoDepthFormatZ16Unorm ||
      source.depth_format ==
          pvrgpu::stub::kDriverPcoDepthFormatZ32Unorm ||
      source.depth_format ==
          pvrgpu::stub::kDriverPcoDepthFormatZ24X8Unorm;
  const bool color_attachment_source_valid =
      source.color_attachment_source_command_index ==
          PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR ||
      source.color_attachment_source_command_index < ordinal;
  const bool depth_attachment_source_valid =
      source.depth_attachment_source_command_index ==
          PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR ||
      source.depth_attachment_source_command_index < ordinal;
  if (source.sampled_texture_count >
          2U * pvrgpu::stub::kPcoMaximumTextureDescriptorSets ||
      source.sampled_texture_bytes ||
      source.sampled_texture_bytes_size != 0 ||
      source.sampled_texture_width != 0 ||
      source.sampled_texture_height != 0 ||
      source.sampled_texture_row_pitch != 0 ||
      source.sampled_texture_format ||
      source.sampled_texture_mip_count != 0 || source.front_ccw > 1 ||
      source.cull_face > 3 || source.fill_front != 0 ||
      source.fill_back != 0 || source.scissor != 0 ||
      source.rasterizer_discard != 0 || source.multisample != 0 ||
      source.half_pixel_center != 1 || source.bottom_edge_rule != 0 ||
      source.clip_halfz != 0 || source.depth_clip_near != 1 ||
      source.depth_clip_far != 1 || source.depth_clamp != 0 ||
      source.sample_mask != UINT32_MAX || source.color_mask > 0x0f ||
      source.blend_enable > 1 || !blend_enums_valid ||
      !disabled_blend_is_canonical || source.dither != 1 ||
      source.depth_enable > 1 || source.depth_write > 1 ||
      (source.depth_write != 0 && source.depth_enable == 0) ||
      source.depth_func > 7 || !std::isfinite(depth_clear) ||
      depth_clear < 0.0F || depth_clear > 1.0F ||
      !depth_format_supported || !color_attachment_source_valid ||
      !depth_attachment_source_valid) {
    *error = "SystemC API nested PCO sequence raster/resource state is invalid";
    return false;
  }

  pvrgpu::stub::DriverCommand command;
  command.enabled = true;
  command.schema = source.schema && source.schema[0]
                       ? source.schema
                       : "pvrgpu.driver-command.v1";
  command.producer = source.producer && source.producer[0]
                         ? source.producer
                         : "pvrgpu-gallium-driver";
  command.command = source.command;
  command.test_case = source.case_name;
  command.frame = source.frame;
  command.framebuffer_width = source.framebuffer_width;
  command.framebuffer_height = source.framebuffer_height;
  command.width = source.width;
  command.height = source.height;
  command.format = source.format;
  command.clear_color_bits = {source.clear_color_bits[0],
                              source.clear_color_bits[1],
                              source.clear_color_bits[2],
                              source.clear_color_bits[3]};
  CopyPcoPayloadFields(source, &command);
  command.draw_count = source.draw_count;
  command.index_count = source.index_count;
  command.unique_vertices = source.unique_vertices;
  command.primitive_count = source.primitive_count;
  command.clip_primitives = source.clip_primitives;
  command.setup_triangles = source.setup_triangles;
  command.semantic_texel_fetches = source.semantic_texel_fetches;
  command.ia_vertices = source.ia_vertices;
  command.ia_primitives = source.ia_primitives;
  command.vs_invocations = source.vs_invocations;
  command.gs_invocations = source.gs_invocations;
  command.gs_primitives = source.gs_primitives;
  command.clip_invocations = source.clip_invocations;
  command.ps_invocations = source.ps_invocations;
  command.hs_invocations = source.hs_invocations;
  command.ds_invocations = source.ds_invocations;
  command.cs_invocations = source.cs_invocations;
  *destination = std::move(command);
  return true;
}

bool CopyCommand(const pvrgpu_systemc_driver_command &source,
                 pvrgpu::stub::DriverCommand *destination,
                 std::string *error) {
  if (!destination || !error)
    return false;
  if (source.version != PVRGPU_SYSTEMC_API_VERSION) {
    *error = "unsupported SystemC API command version";
    return false;
  }
  if (!source.command || !source.command[0]) {
    *error = "missing SystemC API command";
    return false;
  }
  if (!source.case_name || !source.case_name[0]) {
    *error = "missing SystemC API case name";
    return false;
  }
  if (!source.format || !source.format[0]) {
    *error = "missing SystemC API format";
    return false;
  }

  pvrgpu::stub::DriverCommand command;
  command.enabled = true;
  command.schema = source.schema && source.schema[0]
                       ? source.schema
                       : "pvrgpu.driver-command.v1";
  command.producer = source.producer && source.producer[0]
                         ? source.producer
                         : "pvrgpu-gallium-driver";
  command.command = source.command;
  command.test_case = source.case_name;
  command.frame = source.frame;
  command.framebuffer_width = source.framebuffer_width != 0
                                  ? source.framebuffer_width
                                  : source.width;
  command.framebuffer_height = source.framebuffer_height != 0
                                   ? source.framebuffer_height
                                   : source.height;
  command.width = source.width;
  command.height = source.height;
  command.format = source.format;
  command.clear_color_bits = {source.clear_color_bits[0],
                              source.clear_color_bits[1],
                              source.clear_color_bits[2],
                              source.clear_color_bits[3]};
  for (std::size_t vertex = 0; vertex < command.vertex_bits.size(); ++vertex) {
    command.vertex_bits[vertex][0] = source.vertex_bits[vertex][0];
    command.vertex_bits[vertex][1] = source.vertex_bits[vertex][1];
    command.texcoord_bits[vertex][0] = source.texcoord_bits[vertex][0];
    command.texcoord_bits[vertex][1] = source.texcoord_bits[vertex][1];
  }
  command.fragment_color_bits = {source.fragment_color_bits[0],
                                 source.fragment_color_bits[1],
                                 source.fragment_color_bits[2],
                                 source.fragment_color_bits[3]};
  command.texture_width = source.texture_width;
  command.texture_height = source.texture_height;
  if (source.texture_rgba8_path && source.texture_rgba8_path[0])
    command.texture_rgba8_path = source.texture_rgba8_path;
  if (command.command == "draw_textured_triangles" &&
      !CopyTextureSidecarBytes(source, &command.texture_rgba8_bytes, error)) {
    return false;
  }
  if (command.command == "draw_pco_triangles" &&
      !CopyPcoTrianglePayload(source, &command, error)) {
    return false;
  }
  command.draw_count = source.draw_count;
  command.index_count = source.index_count;
  command.unique_vertices = source.unique_vertices;
  command.primitive_count = source.primitive_count;
  command.clip_primitives = source.clip_primitives;
  command.setup_triangles = source.setup_triangles;
  command.semantic_texel_fetches = source.semantic_texel_fetches;
  command.ia_vertices = source.ia_vertices;
  command.ia_primitives = source.ia_primitives;
  command.vs_invocations = source.vs_invocations;
  command.gs_invocations = source.gs_invocations;
  command.gs_primitives = source.gs_primitives;
  command.clip_invocations = source.clip_invocations;
  command.ps_invocations = source.ps_invocations;
  command.hs_invocations = source.hs_invocations;
  command.ds_invocations = source.ds_invocations;
  command.cs_invocations = source.cs_invocations;
  if (source.framebuffer_rgba8_path && source.framebuffer_rgba8_path[0])
    command.framebuffer_rgba8_path = source.framebuffer_rgba8_path;

  *destination = command;
  return true;
}

std::uint64_t CommandOwnedPayloadBytes(
    const pvrgpu::stub::DriverCommand &command) {
  std::uint64_t byte_vectors =
      static_cast<std::uint64_t>(command.raw_vertex_data.size()) +
      command.vertex_pco.size() + command.fragment_pco.size() +
      command.sampled_texture_bytes.size() + command.texture_rgba8_bytes.size();
  for (const pvrgpu::stub::DriverPcoSampledTexture &texture :
       command.sampled_textures) {
    if (texture.bytes.size() >
        std::numeric_limits<std::uint64_t>::max() - byte_vectors) {
      throw std::overflow_error("SystemC API sequence payload size overflow");
    }
    byte_vectors += texture.bytes.size();
  }
  const std::uint64_t dword_count =
      static_cast<std::uint64_t>(command.vertex_shared.size()) +
      command.fragment_shared.size();
  if (dword_count >
      (std::numeric_limits<std::uint64_t>::max() - byte_vectors) /
          sizeof(std::uint32_t)) {
    throw std::overflow_error("SystemC API sequence payload size overflow");
  }
  return byte_vectors + dword_count * sizeof(std::uint32_t);
}

bool CopyPcoSequenceTexture(
    const pvrgpu_systemc_pco_sequence_texture &source,
    std::size_t consumer_command_index,
    pvrgpu::stub::DriverPcoSampledTexture *destination,
    std::string *error) {
  if (!destination || !error)
    return false;
  const std::string_view format = source.format ? source.format : "";
  const std::uint32_t bytes_per_texel =
      format == "PIPE_FORMAT_R8G8B8A8_UNORM" ||
              format == "PIPE_FORMAT_R8G8B8X8_UNORM" ||
              format == "PIPE_FORMAT_Z32_UNORM"
          ? 4U
          : 0U;
  if (source.source >
          PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_DEPTH_ATTACHMENT ||
      source.stage > PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT ||
      source.descriptor_set >=
          pvrgpu::stub::kPcoMaximumTextureDescriptorSets ||
      source.binding != 0 || bytes_per_texel == 0 ||
      source.declared_bytes_size == 0 ||
      source.declared_bytes_size >
          pvrgpu::stub::kDriverPcoMaximumSequencePayloadBytes ||
      source.mip_count == 0 ||
      source.mip_count > PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS ||
      source.min_filter > PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR ||
      source.mag_filter > PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR ||
      source.mip_filter > PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_LINEAR ||
      source.wrap_u > PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_REPEAT ||
      source.wrap_v > PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_REPEAT ||
      source.normalized_coordinates != 1 ||
      source.min_lod_u4_6 > source.max_lod_u4_6) {
    *error = "SystemC API PCO sequence texture metadata is invalid";
    return false;
  }

  std::uint64_t previous_end = 0;
  for (std::size_t level = 0; level < source.mip_count; ++level) {
    const pvrgpu_systemc_pco_texture_mip &mip = source.mip[level];
    const std::uint64_t tight_pitch =
        static_cast<std::uint64_t>(mip.width) * bytes_per_texel;
    const std::uint64_t level_bytes =
        static_cast<std::uint64_t>(mip.row_pitch) * mip.height;
    const std::uint64_t level_end =
        static_cast<std::uint64_t>(mip.offset) + level_bytes;
    if (mip.width == 0 || mip.height == 0 ||
        tight_pitch > std::numeric_limits<std::uint32_t>::max() ||
        mip.row_pitch < tight_pitch || mip.offset != previous_end ||
        level_bytes == 0 || level_end < mip.offset ||
        level_end > source.declared_bytes_size) {
      *error = "SystemC API PCO sequence mip layout is invalid";
      return false;
    }
    previous_end = level_end;
  }
  for (std::size_t level = source.mip_count;
       level < PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS; ++level) {
    const pvrgpu_systemc_pco_texture_mip &mip = source.mip[level];
    if (mip.width != 0 || mip.height != 0 || mip.row_pitch != 0 ||
        mip.offset != 0) {
      *error = "SystemC API PCO sequence unused mip metadata is nonzero";
      return false;
    }
  }
  if (previous_end != source.declared_bytes_size) {
    *error = "SystemC API PCO sequence mip bytes do not cover the resource";
    return false;
  }

  const bool external =
      source.source == PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD;
  if (external) {
    if (source.producer_command_index != 0 || !source.bytes ||
        source.bytes_size != source.declared_bytes_size) {
      *error = "SystemC API external PCO sequence texture payload is invalid";
      return false;
    }
  } else if (source.producer_command_index >= consumer_command_index ||
             source.bytes || source.bytes_size != 0) {
    *error = "SystemC API PCO sequence attachment dependency is invalid";
    return false;
  }

  pvrgpu::stub::DriverPcoSampledTexture texture;
  texture.source = static_cast<pvrgpu::stub::DriverPcoTextureSource>(
      source.source);
  texture.stage = static_cast<pvrgpu::stub::DriverPcoShaderStage>(
      source.stage);
  texture.producer_command_index = source.producer_command_index;
  texture.descriptor_set = source.descriptor_set;
  texture.binding = source.binding;
  texture.format = source.format;
  if (external)
    texture.bytes.assign(source.bytes, source.bytes + source.bytes_size);
  texture.declared_bytes_size = source.declared_bytes_size;
  texture.mip_count = source.mip_count;
  for (std::size_t level = 0; level < source.mip_count; ++level) {
    texture.mip[level] = {
        source.mip[level].width,
        source.mip[level].height,
        source.mip[level].row_pitch,
        source.mip[level].offset,
    };
  }
  texture.min_filter = source.min_filter;
  texture.mag_filter = source.mag_filter;
  texture.mip_filter = source.mip_filter;
  texture.wrap_u = source.wrap_u;
  texture.wrap_v = source.wrap_v;
  texture.normalized_coordinates = source.normalized_coordinates;
  texture.min_lod_u4_6 = source.min_lod_u4_6;
  texture.max_lod_u4_6 = source.max_lod_u4_6;
  *destination = std::move(texture);
  return true;
}


bool CopyPcoSequence(const pvrgpu_systemc_driver_command &source,
                     pvrgpu::stub::Options *options, std::string *error) {
  if (!options || !error || source.version != PVRGPU_SYSTEMC_API_VERSION ||
      !source.command ||
      std::string_view(source.command) != "draw_pco_sequence" ||
      source.pco_sequence_command_count == 0 ||
      source.pco_sequence_command_count >
          PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_COMMANDS ||
      !source.pco_sequence_commands ||
      source.pco_sequence_texture_count >
          PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_TEXTURES ||
      (source.pco_sequence_texture_count != 0 &&
       !source.pco_sequence_textures)) {
    if (error)
      *error = "SystemC API PCO sequence header is invalid";
    return false;
  }

  pvrgpu_systemc_driver_command logical_source = source;
  logical_source.pco_sequence_command_count = 0;
  logical_source.pco_sequence_commands = nullptr;
  logical_source.pco_sequence_texture_count = 0;
  logical_source.pco_sequence_textures = nullptr;
  if (!CopyCommand(logical_source, &options->driver_command, error))
    return false;

  std::vector<pvrgpu::stub::DriverCommand> commands;
  commands.reserve(source.pco_sequence_command_count);
  std::size_t texture_offset = 0;
  std::uint64_t payload_bytes = 0;
  for (std::size_t ordinal = 0;
       ordinal < source.pco_sequence_command_count; ++ordinal) {
    pvrgpu::stub::DriverCommand command;
    if (!CopyPcoSequenceDraw(source.pco_sequence_commands[ordinal], ordinal,
                             &command, error)) {
      return false;
    }
    const auto attachment_matches = [&](std::uint32_t source_ordinal,
                                        bool depth) {
      if (source_ordinal == pvrgpu::stub::kDriverPcoNewAttachment)
        return true;
      const pvrgpu::stub::DriverCommand &producer = commands[source_ordinal];
      return producer.framebuffer_width == command.framebuffer_width &&
             producer.framebuffer_height == command.framebuffer_height &&
             (depth ? producer.depth_format == command.depth_format
                    : producer.format == command.format);
    };
    if (!attachment_matches(command.color_attachment_source_command_index,
                            false) ||
        !attachment_matches(command.depth_attachment_source_command_index,
                            true)) {
      *error =
          "SystemC API PCO sequence attachment alias format/extent mismatch";
      return false;
    }
    const std::size_t texture_count = command.sampled_texture_count;
    if (texture_count > source.pco_sequence_texture_count -
                            std::min(texture_offset,
                                     static_cast<std::size_t>(
                                         source.pco_sequence_texture_count))) {
      *error = "SystemC API PCO sequence texture slices are truncated";
      return false;
    }
    command.sampled_textures.reserve(texture_count);
    std::array<bool, pvrgpu::stub::kPcoMaximumTextureDescriptorSets>
        vertex_sets{};
    std::array<bool, pvrgpu::stub::kPcoMaximumTextureDescriptorSets>
        fragment_sets{};
    bool saw_fragment_texture = false;
    for (std::size_t texture_index = 0; texture_index < texture_count;
         ++texture_index) {
      pvrgpu::stub::DriverPcoSampledTexture texture;
      if (!CopyPcoSequenceTexture(
              source.pco_sequence_textures[texture_offset + texture_index],
              ordinal, &texture, error)) {
        return false;
      }
      auto &sets =
          texture.stage == pvrgpu::stub::DriverPcoShaderStage::kVertex
              ? vertex_sets
              : fragment_sets;
      if (texture.stage == pvrgpu::stub::DriverPcoShaderStage::kFragment) {
        saw_fragment_texture = true;
      } else if (saw_fragment_texture) {
        *error =
            "SystemC API PCO sequence texture stages are not VS-then-FS";
        return false;
      }
      if (sets[texture.descriptor_set]) {
        *error = "SystemC API PCO sequence descriptor set is duplicated";
        return false;
      }
      sets[texture.descriptor_set] = true;
      if (texture.stage == pvrgpu::stub::DriverPcoShaderStage::kVertex)
        ++command.vertex_sampled_texture_count;
      else
        ++command.fragment_sampled_texture_count;
      command.sampled_textures.push_back(std::move(texture));
    }
    const auto sets_are_dense = [](const auto &sets, std::size_t count) {
      return std::all_of(sets.begin(), sets.begin() + count,
                         [](bool present) { return present; }) &&
             std::none_of(sets.begin() + count, sets.end(),
                          [](bool present) { return present; });
    };
    if (!sets_are_dense(vertex_sets,
                        command.vertex_sampled_texture_count) ||
        !sets_are_dense(fragment_sets,
                        command.fragment_sampled_texture_count)) {
      *error =
          "SystemC API PCO sequence descriptor sets are not stage-dense";
      return false;
    }
    const std::size_t descriptor_dwords =
        pvrgpu::stub::kPcoTextureDescriptorDwordCount;
    if (command.vertex_shared.size() <
            command.vertex_sampled_texture_count * descriptor_dwords ||
        command.fragment_shared.size() <
            command.fragment_sampled_texture_count * descriptor_dwords) {
      *error =
          "SystemC API PCO sequence descriptor prefix exceeds stage shareds";
      return false;
    }
    std::sort(command.sampled_textures.begin(),
              command.sampled_textures.end(),
              [](const auto &left, const auto &right) {
                if (left.stage != right.stage)
                  return left.stage < right.stage;
                return left.descriptor_set < right.descriptor_set;
              });
    for (const auto &texture : command.sampled_textures) {
      if (texture.source ==
          pvrgpu::stub::DriverPcoTextureSource::kExternalPayload) {
        continue;
      }
      const pvrgpu::stub::DriverCommand &producer =
          commands.at(texture.producer_command_index);
      const auto &base = texture.mip[0];
      if (base.width != producer.framebuffer_width ||
          base.height != producer.framebuffer_height) {
        *error =
            "SystemC API PCO sequence sampled attachment extent mismatch";
        return false;
      }
      if (texture.source ==
          pvrgpu::stub::DriverPcoTextureSource::kPreviousColorAttachment) {
        if (texture.format != producer.format ||
            base.row_pitch_bytes != producer.framebuffer_width * 4U) {
          *error =
              "SystemC API PCO sequence sampled color attachment mismatch";
          return false;
        }
      } else {
        const bool depth_matches =
            producer.depth_format ==
                pvrgpu::stub::kDriverPcoDepthFormatZ32Unorm &&
            texture.format == "PIPE_FORMAT_Z32_UNORM" &&
            texture.mip_count == 1 &&
            base.row_pitch_bytes == producer.framebuffer_width * 4U &&
            texture.declared_bytes_size ==
                static_cast<std::uint64_t>(base.row_pitch_bytes) *
                    base.height;
        if (!depth_matches) {
          *error =
              "SystemC API PCO sequence sampled depth attachment mismatch";
          return false;
        }
      }
    }
    texture_offset += texture_count;
    try {
      const std::uint64_t command_bytes = CommandOwnedPayloadBytes(command);
      if (command_bytes >
              pvrgpu::stub::kDriverPcoMaximumSequencePayloadBytes -
                  std::min(payload_bytes,
                           pvrgpu::stub::kDriverPcoMaximumSequencePayloadBytes)) {
        *error = "SystemC API PCO sequence payload limit exceeded";
        return false;
      }
      payload_bytes += command_bytes;
    } catch (const std::exception &exception) {
      *error = exception.what();
      return false;
    }
    commands.push_back(std::move(command));
  }
  if (texture_offset != source.pco_sequence_texture_count) {
    *error = "SystemC API PCO sequence has unconsumed textures";
    return false;
  }

  options->driver_commands = std::move(commands);
  std::string profile_error;
  if (!pvrgpu::stub::DriverPcoSequenceSupported(*options, &profile_error)) {
    *error = "SystemC API PCO sequence profile is unsupported: " +
             profile_error;
    return false;
  }
  return true;
}

bool CommandsShareSequenceTarget(
    const pvrgpu::stub::DriverCommand &first,
    const pvrgpu::stub::DriverCommand &next) {
  return first.enabled && next.enabled &&
         first.command == "draw_pco_triangles" &&
         next.command == "draw_pco_triangles" && first.schema == next.schema &&
         first.producer == next.producer && first.test_case == next.test_case &&
         first.frame == next.frame &&
         first.framebuffer_width == next.framebuffer_width &&
         first.framebuffer_height == next.framebuffer_height &&
         first.width == next.width && first.height == next.height &&
         first.format == next.format &&
         first.clear_color_bits == next.clear_color_bits;
}

void AdoptCapturedCounterMetadata(
    const pvrgpu::stub::DriverCommand &source,
    pvrgpu::stub::DriverCommand *destination) {
  if (!destination || source.draw_count <= destination->draw_count)
    return;
  destination->draw_count = source.draw_count;
  destination->index_count = source.index_count;
  destination->unique_vertices = source.unique_vertices;
  destination->primitive_count = source.primitive_count;
  destination->clip_primitives = source.clip_primitives;
  destination->setup_triangles = source.setup_triangles;
  destination->semantic_texel_fetches = source.semantic_texel_fetches;
  destination->ia_vertices = source.ia_vertices;
  destination->ia_primitives = source.ia_primitives;
  destination->vs_invocations = source.vs_invocations;
  destination->gs_invocations = source.gs_invocations;
  destination->gs_primitives = source.gs_primitives;
  destination->clip_invocations = source.clip_invocations;
  destination->ps_invocations = source.ps_invocations;
  destination->hs_invocations = source.hs_invocations;
  destination->ds_invocations = source.ds_invocations;
  destination->cs_invocations = source.cs_invocations;
}

int RunModelToFiles(const pvrgpu::stub::Options &options,
                    const std::string &jsonl_path,
                    const std::string &stderr_path,
                    std::string *error) {
  std::ofstream jsonl(jsonl_path, std::ios::out | std::ios::trunc);
  if (!jsonl) {
    if (error)
      *error = "cannot open SystemC API jsonl_path: " + jsonl_path;
    return 1;
  }
  std::ofstream stderr_file;
  if (!stderr_path.empty()) {
    stderr_file.open(stderr_path, std::ios::out | std::ios::trunc);
    if (!stderr_file) {
      if (error)
        *error = "cannot open SystemC API stderr_path: " + stderr_path;
      return 1;
    }
  }

  std::streambuf *old_stdout = std::cout.rdbuf(jsonl.rdbuf());
  std::streambuf *old_stderr =
      stderr_file ? std::cerr.rdbuf(stderr_file.rdbuf()) : nullptr;
  const int result = pvrgpu::stub::RunConfiguredModel(options);
  std::cout.rdbuf(old_stdout);
  if (old_stderr)
    std::cerr.rdbuf(old_stderr);
  jsonl.close();
  if (stderr_file)
    stderr_file.close();

  if (result != 0 && error) {
    *error =
        "SystemC model returned non-zero status: " + std::to_string(result);
  }
  return result;
}

int FlushPendingSubmitLocked(std::string *error) {
  if (!g_pending_submit.valid || g_pending_submit.executed)
    return 0;
  g_pending_submit.executed = true;
  if (IsIdeasPcoSequenceRoot(g_pending_submit.options.driver_command) &&
      g_pending_submit.options.driver_commands.size() !=
          pvrgpu::stub::kDriverPcoIdeasSequenceCommands) {
    if (error)
      *error = "Ideas PCO profile requires exactly 180 ordered draws";
    return 2;
  }
  return RunModelToFiles(g_pending_submit.options, g_pending_submit.jsonl_path,
                         g_pending_submit.stderr_path, error);
}

void FlushPendingSubmitAtExit() {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  std::string error;
  const int result = FlushPendingSubmitLocked(&error);
  if (result != 0) {
    std::cerr << "PvrGPU SystemC API deferred flush failed: " << error
              << '\n';
  }
}

}  // namespace

extern "C" int pvrgpu_systemc_submit_driver_command(
    const pvrgpu_systemc_submit_info *info, char *error,
    std::size_t error_size) {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  std::string message;
  if (!info) {
    CopyError(error, error_size, "missing SystemC API submit info");
    return 2;
  }
  if (info->version != PVRGPU_SYSTEMC_API_VERSION) {
    CopyError(error, error_size, "unsupported SystemC API submit version");
    return 2;
  }
  if (!info->command) {
    CopyError(error, error_size, "missing SystemC API command payload");
    return 2;
  }
  // API-v8 expands the nested sequence texture mip table. Reject an older
  // producer before reading `command`, sequence pointers, or any v8 tail byte.
  if (info->command->version != PVRGPU_SYSTEMC_API_VERSION) {
    CopyError(error, error_size, "unsupported SystemC API command version");
    return 2;
  }
  if (!info->jsonl_path || !info->jsonl_path[0]) {
    CopyError(error, error_size, "missing SystemC API jsonl_path");
    return 2;
  }
  if (!info->outdir || !info->outdir[0]) {
    CopyError(error, error_size, "missing SystemC API outdir");
    return 2;
  }

  pvrgpu::stub::Options options;
  options.output_dir = info->outdir;
  if (!SetMemoryMode(info->memory_mode, &options, &message)) {
    CopyError(error, error_size, message);
    return 2;
  }
  const bool pco_sequence =
      info->command->command &&
      std::string_view(info->command->command) == "draw_pco_sequence";
  if (pco_sequence
          ? !CopyPcoSequence(*info->command, &options, &message)
          : (!PcoSequenceTailIsEmpty(*info->command) ||
             !CopyCommand(*info->command, &options.driver_command,
                          &message))) {
    if (!pco_sequence && message.empty())
      message = "SystemC API sequence tail is invalid for a scalar command";
    CopyError(error, error_size, message);
    return 2;
  }

  if (g_pending_submit.valid) {
    const bool sequence_active =
        !g_pending_submit.options.driver_commands.empty();
    const bool sequence_root =
        IsIdeasPcoSequenceRoot(g_pending_submit.options.driver_command);
    if ((sequence_active || sequence_root) && g_pending_submit.executed) {
      CopyError(error, error_size,
                "SystemC API deferred submission has already executed");
      return 2;
    }
    const std::string stderr_path =
        info->stderr_path && info->stderr_path[0] ? info->stderr_path : "";
    const bool compatible_sequence_member =
        (sequence_active || sequence_root) &&
        g_pending_submit.jsonl_path == info->jsonl_path &&
        g_pending_submit.stderr_path == stderr_path &&
        g_pending_submit.options.output_dir == info->outdir &&
        g_pending_submit.options.memory_mode == options.memory_mode &&
        CommandsShareSequenceTarget(g_pending_submit.options.driver_command,
                                    options.driver_command);
    if ((sequence_active || sequence_root) &&
        !compatible_sequence_member) {
      CopyError(error, error_size,
                "SystemC API command is incompatible with the pending "
                "ordered sequence");
      return 2;
    }

    if (compatible_sequence_member) {
      std::vector<pvrgpu::stub::DriverCommand> &commands =
          g_pending_submit.options.driver_commands;
      const std::size_t existing_count =
          commands.empty() ? 1U : commands.size();
      if (existing_count >=
              pvrgpu::stub::kDriverPcoMaximumSequenceCommands ||
          existing_count >=
              pvrgpu::stub::kDriverPcoIdeasSequenceCommands) {
        CopyError(error, error_size,
                  "SystemC API ordered sequence command limit exceeded");
        return 2;
      }
      if (!IdeasDepthStateMatchesOrdinal(options.driver_command,
                                         existing_count)) {
        CopyError(error, error_size,
                  "SystemC API Ideas depth state is invalid for the draw "
                  "ordinal");
        return 2;
      }
      std::uint64_t payload_bytes = 0;
      try {
        if (commands.empty()) {
          payload_bytes = CommandOwnedPayloadBytes(
              g_pending_submit.options.driver_command);
        } else {
          for (const pvrgpu::stub::DriverCommand &command : commands) {
            const std::uint64_t bytes = CommandOwnedPayloadBytes(command);
            if (bytes > std::numeric_limits<std::uint64_t>::max() -
                            payload_bytes) {
              throw std::overflow_error(
                  "SystemC API sequence payload size overflow");
            }
            payload_bytes += bytes;
          }
        }
        const std::uint64_t next_bytes =
            CommandOwnedPayloadBytes(options.driver_command);
        if (next_bytes > std::numeric_limits<std::uint64_t>::max() -
                             payload_bytes ||
            payload_bytes + next_bytes >
                pvrgpu::stub::kDriverPcoMaximumSequencePayloadBytes) {
          CopyError(error, error_size,
                    "SystemC API ordered sequence payload limit exceeded");
          return 2;
        }
      } catch (const std::exception &exception) {
        CopyError(error, error_size, exception.what());
        return 2;
      }

      if (commands.empty())
        commands.push_back(g_pending_submit.options.driver_command);
      commands.push_back(std::move(options.driver_command));
      AdoptCapturedCounterMetadata(
          commands.back(), &g_pending_submit.options.driver_command);
      return 0;
    }
  }

  PendingSubmit pending;
  pending.options = std::move(options);
  pending.jsonl_path = info->jsonl_path;
  if (info->stderr_path && info->stderr_path[0])
    pending.stderr_path = info->stderr_path;
  pending.valid = true;

  if (!g_atexit_registered) {
    if (std::atexit(FlushPendingSubmitAtExit) != 0) {
      CopyError(error, error_size,
                "cannot register SystemC API deferred flush handler");
      return 1;
    }
    g_atexit_registered = true;
  }

  g_pending_submit = std::move(pending);
  return 0;
}
