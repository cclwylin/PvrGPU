#include "model_runner.h"
#include "compute_types.h"
#include "uniform_buffers.h"
#include "shader_images.h"
#include "texture/astc_decoder.h"
#include "texture/texture_unit.h"
#include "pco_sequence_profiles.h"
#include "pvrgpu_systemc_api.h"
#include "pvrgpu_tessellation.h"
#include "pvrgpu_systemc_compute_api.h"
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_bridge_mutex;

struct PendingSubmit {
  pvrgpu::stub::Options options;
  std::string jsonl_path;
  std::string stderr_path;
  bool valid = false;
  bool executed = false;
  std::uint64_t submission_generation = 0;
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
  // The root is identified by what it is -- the case, the command kind, the
  // member count it announces and its depth state -- not by counters it
  // states.  Those are the sequence's totals, summed once every member has
  // arrived.
  return command.enabled && command.command == "draw_pco_triangles" &&
         IsIdeasPcoSequenceCase(command.test_case.c_str()) &&
         command.draw_count ==
             pvrgpu::stub::kDriverPcoIdeasSequenceCommands &&
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

bool CopyPcoSequenceTexture(const pvrgpu_systemc_pco_sequence_texture &source,
    std::size_t consumer_command_index,
    pvrgpu::stub::DriverPcoSampledTexture *destination, std::string *error,
    pvrgpu::stub::TextureResource *compute_resource = nullptr);

std::uint32_t ComputeLowMask(std::uint32_t count) {
  return count == 32U ? UINT32_MAX : (UINT32_C(1) << count) - 1U;
}

/* Validate the complete envelope before cloning any borrowed resource. The
 * stage ABI is compiler metadata, never a shader identity or a canned answer.
 * Read/write aliases retain one backing allocation and independent views. */
bool PrepareComputeDispatch(
    const pvrgpu_systemc_compute_dispatch &source,
    pvrgpu::stub::ModelComputeDispatch *out, std::string *error) {
  const auto fail = [error](const char *message) {
    *error = message;
    return false;
  };
  const auto &abi = source.abi;
  const auto &stage = abi.stage;
  if (!source.binary || source.binary_size == 0 ||
      source.binary_size > PVRGPU_SYSTEMC_COMPUTE_MAX_BINARY_BYTES ||
      stage.entry_offset != 0 || source.memory_mode > 2U)
    return fail("invalid compute binary, entry point, or memory mode");
  if (stage.temps > pvrgpu::stub::kPcoTemporaryCount ||
      stage.vertex_inputs > pvrgpu::stub::kPcoVertexInputCount ||
      stage.vertex_outputs != 0 ||
      stage.coefficients > 256U ||
      stage.shareds > pvrgpu::stub::kPcoMaximumSharedCount ||
      stage.uniform_buffer_descriptor_count > 15U ||
      abi.storage_buffer_descriptor_count > 32U ||
      abi.sampled_texture_count > pvrgpu::stub::kPcoMaximumTextureDescriptorSets ||
      abi.image_descriptor_count > PVRGPU_SYSTEMC_COMPUTE_MAX_IMAGES ||
      abi.shared_memory_bytes > PVRGPU_SYSTEMC_COMPUTE_MAX_SHARED_BYTES ||
      (abi.shared_memory_bytes & 3U) || abi.scratch_bytes != 0)
    return fail("compute ABI exceeds supported native execution bounds");
  const std::uint32_t texture_prefix = 20U * abi.sampled_texture_count;
  const std::uint32_t user_prefix = texture_prefix + 4U *
      (stage.uniform_buffer_descriptor_count +
       abi.storage_buffer_descriptor_count);
  const std::uint32_t image_end = user_prefix + 8U * abi.image_descriptor_count;
  if (abi.image_descriptor_start != (abi.image_descriptor_count ? user_prefix : 0U) ||
      (abi.image_used_mask & ~ComputeLowMask(abi.image_descriptor_count)) != 0 ||
      ((abi.image_read_mask | abi.image_write_mask) & ~abi.image_used_mask) != 0)
    return fail("invalid compute image descriptor layout or use masks");
  const std::uint32_t private_count = abi.shared_memory_bytes ? 4U : 0U;
  if (abi.shared_memory_descriptor_count != private_count ||
      abi.shared_memory_descriptor_start != (private_count ? image_end : 0U))
    return fail("invalid compute private workgroup descriptor layout");
  const std::uint32_t prefix = image_end + private_count;
  if (stage.uniform_buffer_descriptor_start != texture_prefix ||
      abi.storage_buffer_descriptor_start !=
          texture_prefix + 4U * stage.uniform_buffer_descriptor_count ||
      stage.push_constant_start != prefix || prefix > stage.shareds ||
      stage.push_constant_count > stage.shareds - prefix ||
      source.push_word_count != stage.push_constant_count ||
      (source.push_word_count != 0 && !source.push_words))
    return fail("invalid compute descriptor/push-register layout");
  const auto fits = [](std::uint32_t start, std::uint32_t count,
                       std::uint32_t size, std::uint32_t max_count) {
    return count <= max_count && (count == 0 ||
           (start <= size && count <= size - start));
  };
  if (!fits(abi.local_invocation_index_start,
            abi.local_invocation_index_count, stage.vertex_inputs, 1U) ||
      !fits(abi.workgroup_id_start, abi.workgroup_id_count,
            stage.coefficients, 3U) ||
      !fits(abi.num_workgroups_start, abi.num_workgroups_count,
            stage.coefficients, 3U))
    return fail("invalid compute system-value register ranges");
  if (abi.workgroup_id_count != 0 && abi.num_workgroups_count != 0 &&
      abi.workgroup_id_start <
          abi.num_workgroups_start + abi.num_workgroups_count &&
      abi.num_workgroups_start <
          abi.workgroup_id_start + abi.workgroup_id_count)
    return fail("overlapping compute system-value coefficient ranges");
  if ((abi.uniform_buffer_used_mask &
       ~ComputeLowMask(stage.uniform_buffer_descriptor_count)) != 0 ||
      (abi.storage_buffer_used_mask &
       ~ComputeLowMask(abi.storage_buffer_descriptor_count)) != 0 ||
      ((abi.storage_buffer_read_mask | abi.storage_buffer_write_mask) &
       ~abi.storage_buffer_used_mask) != 0)
    return fail("invalid compute resource-use masks");
  std::uint64_t lanes = 1;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (source.grid[axis] > 65535U || source.block[axis] == 0 ||
        source.block[axis] > (axis == 2 ? 64U : 1024U) ||
        source.block[axis] != abi.local_size[axis])
      return fail("compute grid/block does not match the compiled local size");
    lanes *= source.block[axis];
  }
  if (lanes > 1024U)
    return fail("compute workgroup exceeds 1024 invocations");
  if (source.resource_count > PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCES ||
      source.binding_count > PVRGPU_SYSTEMC_COMPUTE_MAX_BINDINGS ||
      source.image_count > PVRGPU_SYSTEMC_COMPUTE_MAX_IMAGES ||
      (source.resource_count != 0 && !source.resources) ||
      (source.binding_count != 0 && !source.bindings) ||
      (source.image_count != 0 && !source.images))
    return fail("invalid compute resource or binding array");
  std::uint64_t total_bytes = 0;
  for (std::size_t index = 0; index < source.resource_count; ++index) {
    const auto &resource = source.resources[index];
    if (resource.bytes_size > PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES ||
        (resource.bytes_size != 0 && !resource.bytes))
      return fail("invalid compute backing resource");
    total_bytes += resource.bytes_size;
  }
  if (total_bytes > PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES)
    return fail("compute snapshot exceeds the 1 GiB global-memory bound");
  std::uint32_t present_ubos = 0, present_ssbos = 0;
  for (std::size_t index = 0; index < source.binding_count; ++index) {
    const auto &binding = source.bindings[index];
    if (binding.kind > PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER ||
        binding.resource_index >= source.resource_count ||
        (binding.access & ~3U) != 0)
      return fail("invalid compute buffer binding");
    const bool uniform = binding.kind == PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER;
    const auto count = uniform ? stage.uniform_buffer_descriptor_count
                              : abi.storage_buffer_descriptor_count;
    if (binding.slot >= count)
      return fail("compute buffer slot exceeds its descriptor range");
    const std::uint32_t bit = UINT32_C(1) << binding.slot;
    auto &present = uniform ? present_ubos : present_ssbos;
    if ((present & bit) != 0)
      return fail("duplicate compute binding slot");
    present |= bit;
    const auto &resource = source.resources[binding.resource_index];
    if (binding.offset > resource.bytes_size ||
        binding.bytes_size > resource.bytes_size - binding.offset ||
        binding.bytes_size > UINT32_MAX ||
        (uniform && binding.bytes_size >
             PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES))
      return fail("compute binding range exceeds its backing resource");
    const auto required = uniform
        ? ((abi.uniform_buffer_used_mask & bit) != 0 ? 1U : 0U)
        : (((abi.storage_buffer_read_mask & bit) != 0 ? 1U : 0U) |
           ((abi.storage_buffer_write_mask & bit) != 0 ? 2U : 0U));
    if ((binding.access & required) != required ||
        (uniform && (binding.access & 2U) != 0))
      return fail("compute binding permissions do not cover shader accesses");
  }
  if ((abi.uniform_buffer_used_mask & ~present_ubos) != 0 ||
      (abi.storage_buffer_used_mask & ~present_ssbos) != 0)
    return fail("compute shader uses an unbound buffer");
  std::uint32_t present_images = 0;
  for (std::size_t index = 0; index < source.image_count; ++index) {
    const auto &binding = source.images[index];
    const auto texel_bytes = binding.texel_bytes ? binding.texel_bytes : 4U;
    const auto depth = binding.depth ? binding.depth : 1U;
    const std::uint64_t layer_stride = binding.layer_stride_bytes
        ? binding.layer_stride_bytes : static_cast<std::uint64_t>(binding.height) * binding.row_stride_bytes;
    if (binding.slot >= abi.image_descriptor_count ||
        binding.resource_index >= source.resource_count ||
        (binding.access & ~3U) || binding.reserved ||
        (binding.format != PVRGPU_SYSTEMC_COMPUTE_IMAGE_R32UI &&
         binding.format != PVRGPU_SYSTEMC_COMPUTE_IMAGE_RAW) ||
        (texel_bytes != 4U && texel_bytes != 8U && texel_bytes != 16U) ||
        (binding.format == PVRGPU_SYSTEMC_COMPUTE_IMAGE_R32UI && texel_bytes != 4U) ||
        !binding.width || !binding.height || binding.width > UINT32_MAX / texel_bytes ||
        binding.row_stride_bytes < texel_bytes * binding.width ||
        layer_stride > UINT32_MAX || (layer_stride & 3U) ||
        (binding.row_stride_bytes & 3U) || (binding.offset & 3U))
      return fail("invalid compute image view");
    const std::uint32_t bit = UINT32_C(1) << binding.slot;
    if (present_images & bit) return fail("duplicate compute image slot");
    present_images |= bit;
    const auto &resource = source.resources[binding.resource_index];
    const std::uint64_t row_extent = static_cast<std::uint64_t>(binding.height - 1U) *
        binding.row_stride_bytes + texel_bytes * binding.width;
    const std::uint64_t footprint = static_cast<std::uint64_t>(depth - 1U) * layer_stride + row_extent;
    if (binding.offset > resource.bytes_size ||
        binding.bytes_size > resource.bytes_size - binding.offset ||
        binding.bytes_size > UINT32_MAX || row_extent > layer_stride || footprint > binding.bytes_size)
      return fail("compute image row/mip extent exceeds its backing resource");
    const auto required = ((abi.image_read_mask & bit) ? 1U : 0U) |
                          ((abi.image_write_mask & bit) ? 2U : 0U);
    if ((binding.access & required) != required)
      return fail("compute image permissions do not cover shader accesses");
  }
  if (abi.image_used_mask & ~present_images)
    return fail("compute shader uses an unbound image");

  pvrgpu::stub::ModelComputeDispatch prepared;
  auto &target = prepared.abi;
  target.stage = {stage.temps, stage.vertex_inputs, stage.vertex_outputs,
                  stage.coefficients, stage.shareds, stage.push_constant_start,
                  stage.push_constant_count, stage.entry_offset,
                  stage.uniform_buffer_descriptor_start,
                  stage.uniform_buffer_descriptor_count};
  for (unsigned axis = 0; axis < 3; ++axis) {
    target.local_size[axis] = abi.local_size[axis];
    prepared.grid[axis] = source.grid[axis];
    prepared.block[axis] = source.block[axis];
  }
  target.local_invocation_index_start = abi.local_invocation_index_start;
  target.local_invocation_index_count = abi.local_invocation_index_count;
  target.workgroup_id_start = abi.workgroup_id_start;
  target.workgroup_id_count = abi.workgroup_id_count;
  target.num_workgroups_start = abi.num_workgroups_start;
  target.num_workgroups_count = abi.num_workgroups_count;
  target.storage_buffer_descriptor_start = abi.storage_buffer_descriptor_start;
  target.storage_buffer_descriptor_count = abi.storage_buffer_descriptor_count;
  target.uniform_buffer_used_mask = abi.uniform_buffer_used_mask;
  target.storage_buffer_used_mask = abi.storage_buffer_used_mask;
  target.storage_buffer_read_mask = abi.storage_buffer_read_mask;
  target.storage_buffer_write_mask = abi.storage_buffer_write_mask;
  target.shared_memory_bytes = abi.shared_memory_bytes;
  target.scratch_bytes = abi.scratch_bytes;
  target.shared_memory_descriptor_start = abi.shared_memory_descriptor_start;
  target.shared_memory_descriptor_count = abi.shared_memory_descriptor_count;
  target.image_descriptor_start = abi.image_descriptor_start;
  target.image_descriptor_count = abi.image_descriptor_count;
  target.image_used_mask = abi.image_used_mask;
  target.image_read_mask = abi.image_read_mask;
  target.image_write_mask = abi.image_write_mask;
  target.sampled_texture_count = abi.sampled_texture_count;
  if (source.texture_count != abi.sampled_texture_count ||
      source.texture_word_count != texture_prefix ||
      (source.texture_count && (!source.textures || !source.texture_words)))
    return fail("invalid compute native texture descriptor/payload span");
  if (texture_prefix)
    prepared.texture_words.assign(source.texture_words, source.texture_words + texture_prefix);
  prepared.textures.resize(source.texture_count);
  prepared.texture_resources.resize(source.texture_count);
  for (std::size_t index = 0; index < source.texture_count; ++index) {
    const auto &texture = source.textures[index];
    if (texture.stage != PVRGPU_SYSTEMC_PCO_SHADER_STAGE_COMPUTE ||
        texture.source != PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD ||
        texture.descriptor_set != index ||
        !CopyPcoSequenceTexture(texture, 0, &prepared.textures[index], error,
                               &prepared.texture_resources[index]))
      return fail("invalid compute external texture payload or sampler layout");
    total_bytes += texture.declared_bytes_size;
    if (total_bytes > PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES)
      return fail("compute textures and buffers exceed the 1 GiB global-memory bound");
  }
  prepared.memory_mode = static_cast<pvrgpu::stub::MemoryMode>(source.memory_mode);
  prepared.binary.assign(source.binary, source.binary + source.binary_size);
  if (source.push_word_count != 0)
    prepared.push_words.assign(source.push_words,
                               source.push_words + source.push_word_count);
  prepared.resources.resize(source.resource_count);
  for (std::size_t index = 0; index < source.resource_count; ++index) {
    const auto &resource = source.resources[index];
    if (resource.bytes_size != 0)
      prepared.resources[index].bytes.assign(resource.bytes,
                                              resource.bytes + resource.bytes_size);
  }
  for (std::size_t index = 0; index < source.binding_count; ++index) {
    const auto &binding = source.bindings[index];
    prepared.bindings.push_back({binding.kind, binding.slot,
                                binding.resource_index, binding.access,
                                binding.offset, binding.bytes_size});
    if ((binding.access & 2U) != 0)
      prepared.resources[binding.resource_index].writable = true;
  }
  for (std::size_t index = 0; index < source.image_count; ++index) {
    const auto &binding = source.images[index];
    prepared.images.push_back({binding.slot, binding.resource_index,
      binding.access, binding.format, binding.offset, binding.bytes_size,
      binding.width, binding.height, binding.row_stride_bytes});
    auto &image = prepared.images.back();
    image.depth = binding.depth ? binding.depth : 1U;
    image.layer_stride_bytes = binding.layer_stride_bytes ? binding.layer_stride_bytes
        : binding.height * binding.row_stride_bytes;
    image.texel_bytes = binding.texel_bytes ? binding.texel_bytes : 4U;
    if (binding.access & 2U) prepared.resources[binding.resource_index].writable = true;
  }
  *out = std::move(prepared);
  return true;
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
         actual.entry_offset == expected.entry_offset &&
         actual.uniform_buffer_descriptor_start == expected.uniform_buffer_descriptor_start &&
         actual.uniform_buffer_descriptor_count == expected.uniform_buffer_descriptor_count;
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
         abi.uniform_buffer_descriptor_count <= pvrgpu::stub::kMaximumUniformBuffersPerStage &&
         abi.uniform_buffer_descriptor_start <= abi.shareds &&
         abi.uniform_buffer_descriptor_count * 4U <=
             abi.shareds - abi.uniform_buffer_descriptor_start &&
         abi.entry_offset == 0;
}

template <typename Abi>
std::string PcoStageAbiText(const Abi &abi) {
  std::ostringstream text;
  text << abi.temps << ',' << abi.vertex_inputs << ',' << abi.vertex_outputs
       << ',' << abi.coefficients << ',' << abi.shareds << ','
       << abi.push_constant_start << ',' << abi.push_constant_count << ','
       << abi.entry_offset;
  if (abi.uniform_buffer_descriptor_start || abi.uniform_buffer_descriptor_count)
    text << ',' << abi.uniform_buffer_descriptor_start
         << ',' << abi.uniform_buffer_descriptor_count;
  return text.str();
}

/*
 * Words of one packed vertex the float finiteness check below cannot judge.
 *
 * A command that states its own attribute layout carries each attribute as
 * the bytes the array holds, and the shader's own unpack decides what they
 * mean: an `int` attribute of -50, a pair of halves, a 2_10_10_10 -- read as
 * binary32 any of them can be a NaN, an infinity or a denormal.  So every
 * word such a command describes is opaque, and only the pinned capture
 * profiles, which really do carry floats, are checked.
 */
std::uint64_t OpaqueVertexWordMask(
    const pvrgpu_systemc_driver_command &source) {
  std::uint64_t mask = 0;
  std::uint32_t word = 0;
  const std::uint32_t attribute_count = std::min<std::uint32_t>(
      source.vertex_attribute_count,
      static_cast<std::uint32_t>(
          sizeof(source.vertex_attribute_components) /
          sizeof(source.vertex_attribute_components[0])));
  for (std::uint32_t attribute = 0; attribute < attribute_count; ++attribute) {
    const std::uint32_t components =
        source.vertex_attribute_components[attribute];
    for (std::uint32_t component = 0; component < components; ++component) {
      if (word >= 64U)
        return mask;
      mask |= UINT64_C(1) << word;
      ++word;
    }
  }
  return mask;
}

bool RawFloatVerticesAreFinite(const std::uint8_t *data,
                               std::uint64_t vertex_count,
                               std::uint32_t stride,
                               std::uint32_t component_count,
                               std::uint64_t opaque_word_mask = 0) {
  if (!data || component_count == 0 ||
      component_count * sizeof(float) > stride) {
    return false;
  }
  for (std::uint64_t vertex = 0; vertex < vertex_count; ++vertex) {
    const std::size_t offset = static_cast<std::size_t>(vertex * stride);
    for (std::uint32_t component = 0; component < component_count;
         ++component) {
      if (component < 64U &&
          (opaque_word_mask & (UINT64_C(1) << component)) != 0) {
        continue;
      }
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
// One to four colour attachments; zero is read as the single-target default.
bool DriverPcoRenderTargetCountIsValid(std::uint32_t render_target_count) {
  return render_target_count <= 4U;
}

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

// True when the stated viewport scale is the expected extent up to the sign of
// Y.  A window-system framebuffer is y-flipped, so GL states scale_y = -h/2
// there; the clip/cull stage applies ndc * scale + offset as written, so the
// reflection is honoured rather than special-cased.  Negating an IEEE float
// toggles its sign bit only.
bool PcoViewportScaleMatches(const std::array<std::uint32_t, 3> &expected,
                             const std::uint32_t actual[3]) {
  return actual[0] == expected[0] &&
         (actual[1] == expected[1] ||
          actual[1] == (expected[1] ^ UINT32_C(0x80000000))) &&
         actual[2] == expected[2];
}

bool ValidateStreamOutput(const pvrgpu_systemc_stream_output *so,
                         std::uint32_t output_dwords, std::string *error) {
  if (!so)
    return true;
  const auto refuse = [&](const char *reason) {
    *error = std::string("SystemC API stream output ") + reason;
    return false;
  };
  if (!so->bindings || !so->binding_count ||
      so->binding_count > PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_BINDINGS ||
      so->target_count > PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_BUFFERS ||
      ((so->target_count != 0) != (so->targets != nullptr)))
    return refuse("binding/target list is invalid");
  std::array<const pvrgpu_systemc_stream_output_target *,
             PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_BUFFERS> targets{};
  for (std::uint32_t i = 0; i < so->target_count; ++i) {
    const auto &t = so->targets[i];
    if (t.output_buffer >= targets.size() || targets[t.output_buffer] ||
        !t.resource_token || !t.target_token || !t.bytes || !t.bytes_size ||
        t.bytes_size > PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_RESOURCE_BYTES ||
        t.buffer_offset > t.bytes_size || t.buffer_size > t.bytes_size - t.buffer_offset ||
        t.internal_offset > t.buffer_size ||
        (t.buffer_offset % 4) || (t.internal_offset % 4) ||
        !t.stride_dwords || t.stride_dwords > 64)
      return refuse("target identity/range/stride is invalid");
    for (std::uint32_t j = 0; j < i; ++j) {
      const auto &prior = so->targets[j];
      if (t.target_token == prior.target_token)
        return refuse("target token is duplicated");
      if (t.resource_token == prior.resource_token &&
          (t.bytes_size != prior.bytes_size ||
           std::memcmp(t.bytes, prior.bytes, t.bytes_size) != 0))
        return refuse("aliased resource snapshots disagree");
    }
    targets[t.output_buffer] = &t;
  }
  for (std::uint32_t i = 0; i < so->binding_count; ++i) {
    const auto &b = so->bindings[i];
    if (b.stream || b.output_buffer >= targets.size() ||
        !b.num_components || b.num_components > 4 ||
        b.output_dword >= output_dwords || b.num_components > output_dwords - b.output_dword ||
        b.dst_offset_dwords > 64 || b.num_components > 64 - b.dst_offset_dwords)
      return refuse("binding output range/stream is invalid");
    const auto *target = targets[b.output_buffer];
    if (target && b.dst_offset_dwords + b.num_components > target->stride_dwords)
      return refuse("binding exceeds target stride");
    for (std::uint32_t j = 0; j < i; ++j) {
      const auto &prior = so->bindings[j];
      if (b.output_buffer == prior.output_buffer &&
          b.dst_offset_dwords < prior.dst_offset_dwords + prior.num_components &&
          prior.dst_offset_dwords < b.dst_offset_dwords + b.num_components)
        return refuse("binding destination ranges overlap");
    }
  }
  return true;
}

void CopyPcoPayloadFields(
    const pvrgpu_systemc_driver_command &source,
    pvrgpu::stub::DriverCommand *destination) {
  if (source.raw_vertex_data_size != 0) {
    destination->raw_vertex_data.assign(
        source.raw_vertex_data,
        source.raw_vertex_data + source.raw_vertex_data_size);
  } else {
    destination->raw_vertex_data.clear();
  }
  if (source.raw_index_data && source.raw_index_data_size != 0) {
    destination->raw_index_data.assign(
        source.raw_index_data,
        source.raw_index_data + source.raw_index_data_size);
  } else {
    destination->raw_index_data.clear();
  }
  destination->render_target_count =
      source.render_target_count == 0 ? 1U : source.render_target_count;
  destination->vertex_attribute_count = source.vertex_attribute_count;
  for (std::size_t attribute = 0;
       attribute < destination->vertex_attribute_components.size();
       ++attribute) {
    destination->vertex_attribute_components[attribute] =
        source.vertex_attribute_components[attribute];
    destination->vertex_attribute_integer[attribute] =
        source.vertex_attribute_integer[attribute];
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
  destination->geometry_pco.clear();
  destination->geometry_shared.clear();
  destination->stream_output = {};
  destination->fragment_images.clear();
  destination->fragment_image_descriptor_start = source.fragment_image_descriptor_start;
  destination->fragment_image_descriptor_count = source.fragment_image_descriptor_count;
  destination->fragment_image_read_mask = source.fragment_image_read_mask;
  destination->fragment_image_write_mask = source.fragment_image_write_mask;
  destination->fragment_early_tests = source.fragment_early_tests;
  if (source.fragment_image_count > PVRGPU_SYSTEMC_MAX_SHADER_IMAGES ||
      ((source.fragment_image_count != 0) != (source.fragment_images != nullptr)))
    throw std::runtime_error("SystemC API fragment image payload count/pointer mismatch");
  for (unsigned index = 0; index < source.fragment_image_count; ++index) {
    const auto &image = source.fragment_images[index];
    if (!image.bytes || !image.bytes_size || image.bytes_size > PVRGPU_SYSTEMC_MAX_SHADER_IMAGE_BYTES)
      throw std::runtime_error("SystemC API fragment image backing snapshot is invalid");
    pvrgpu::stub::DriverShaderImage owned;
    owned.image_slot = image.image_slot; owned.format = image.format; owned.access = image.access;
    owned.resource_token = image.resource_token; owned.offset = image.offset;
    owned.width = image.width; owned.height = image.height; owned.depth = image.depth;
    owned.row_stride = image.row_stride; owned.layer_stride = image.layer_stride;
    owned.texel_bytes = image.texel_bytes;
    owned.bytes.assign(image.bytes, image.bytes + image.bytes_size);
    destination->fragment_images.push_back(std::move(owned));
  }
  destination->explicit_varying_bindings = source.varying_bindings != nullptr;
  destination->varying_bindings.clear();
  for (std::uint32_t i = 0; i < source.varying_binding_count; ++i) {
    const auto &binding = source.varying_bindings[i];
    destination->varying_bindings.push_back({binding.output_dword, binding.num_components,
                                            binding.coefficient_dword, binding.flat});
  }
  if (source.stream_output) {
    const auto &so = *source.stream_output;
    for (std::uint32_t i = 0; i < so.binding_count; ++i) {
      const auto &b = so.bindings[i];
      destination->stream_output.bindings.push_back(
          {b.output_dword, b.num_components, b.output_buffer, b.dst_offset_dwords, b.stream});
    }
    for (std::uint32_t i = 0; i < so.target_count; ++i) {
      const auto &t = so.targets[i];
      pvrgpu::stub::DriverStreamOutputTarget owned;
      owned.output_buffer = t.output_buffer;
      owned.resource_token = t.resource_token;
      owned.target_token = t.target_token;
      owned.bytes.assign(t.bytes, t.bytes + t.bytes_size);
      owned.buffer_offset = t.buffer_offset;
      owned.buffer_size = t.buffer_size;
      owned.internal_offset = t.internal_offset;
      owned.stride_dwords = t.stride_dwords;
      destination->stream_output.targets.push_back(std::move(owned));
    }
  }
  destination->tessellation = {};
  if (source.tessellation) {
    const auto &t = *source.tessellation;
    auto &owned = destination->tessellation;
    owned.control_pco.assign(t.control_pco, t.control_pco + t.control_pco_size);
    owned.evaluation_pco.assign(t.evaluation_pco, t.evaluation_pco + t.evaluation_pco_size);
    owned.control_shared.assign(t.control_shared, t.control_shared + t.control_shared_count);
    owned.evaluation_shared.assign(t.evaluation_shared, t.evaluation_shared + t.evaluation_shared_count);
    const auto copy_abi = [](const pvrgpu_systemc_pco_stage_abi &a) {
      return pvrgpu::stub::DriverPcoStageAbi{a.temps, a.vertex_inputs, a.vertex_outputs,
          a.coefficients, a.shareds, a.push_constant_start, a.push_constant_count,
          a.entry_offset, a.uniform_buffer_descriptor_start, a.uniform_buffer_descriptor_count};
    };
    owned.control_abi = copy_abi(t.control_abi);
    owned.evaluation_abi = copy_abi(t.evaluation_abi);
#define COPY_TESS(field) owned.field = t.field
    COPY_TESS(input_vertices); COPY_TESS(output_vertices); COPY_TESS(vertices_per_instance);
    COPY_TESS(input_stride_dwords); COPY_TESS(output_vertex_stride_dwords);
    COPY_TESS(per_vertex_offset_dwords); COPY_TESS(patch_stride_dwords);
    COPY_TESS(control_barrier_count); COPY_TESS(domain); COPY_TESS(spacing);
    COPY_TESS(clockwise); COPY_TESS(point_mode);
#undef COPY_TESS
  }
  if (source.geometry_pco_size != 0) {
    destination->geometry_pco.assign(source.geometry_pco,
                                    source.geometry_pco + source.geometry_pco_size);
    destination->geometry_shared.assign(
        source.geometry_shared, source.geometry_shared + source.geometry_shared_count);
  }
  destination->geometry_pco_abi = {
      source.geometry_pco_abi.temps,
      source.geometry_pco_abi.vertex_inputs,
      source.geometry_pco_abi.vertex_outputs,
      source.geometry_pco_abi.coefficients,
      source.geometry_pco_abi.shareds,
      source.geometry_pco_abi.push_constant_start,
      source.geometry_pco_abi.push_constant_count,
      source.geometry_pco_abi.entry_offset,
      source.geometry_pco_abi.uniform_buffer_descriptor_start,
      source.geometry_pco_abi.uniform_buffer_descriptor_count,
  };
  destination->geometry_input_primitive_vertices = source.geometry_input_primitive_vertices;
  destination->geometry_output_primitive = source.geometry_output_primitive;
  destination->geometry_max_vertices = source.geometry_max_vertices;
  destination->geometry_invocations = source.geometry_invocations;
  destination->geometry_input_stride_dwords = source.geometry_input_stride_dwords;
  destination->geometry_vertices_per_instance = source.geometry_vertices_per_instance;
  destination->geometry_layer_output_start = source.geometry_layer_output_start;
  destination->geometry_layer_output_count = source.geometry_layer_output_count;
  destination->geometry_primitive_id_output_start = source.geometry_primitive_id_output_start;
  destination->geometry_primitive_id_output_count = source.geometry_primitive_id_output_count;
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
      source.vertex_pco_abi.uniform_buffer_descriptor_start,
      source.vertex_pco_abi.uniform_buffer_descriptor_count,
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
      source.fragment_pco_abi.uniform_buffer_descriptor_start,
      source.fragment_pco_abi.uniform_buffer_descriptor_count,
  };
  destination->position_output_start = source.position_output_start;
  destination->position_output_count = source.position_output_count;
  destination->fragment_position_start = source.fragment_position_start;
  destination->fragment_position_count = source.fragment_position_count;
  destination->varying_output_start = source.varying_output_start;
  destination->varying_output_count = source.varying_output_count;
  destination->fragment_varying_start = source.fragment_varying_start;
  destination->fragment_varying_count = source.fragment_varying_count;
  destination->varying_flat_mask = source.varying_flat_mask;
  for (std::size_t target = 0;
       target < destination->fragment_output_mask.size(); ++target) {
    destination->fragment_output_mask[target] =
        source.fragment_output_mask[target];
  }
  std::copy_n(source.viewport_scale_bits, 3,
              destination->viewport_scale_bits.begin());
  std::copy_n(source.viewport_translate_bits, 3,
              destination->viewport_translate_bits.begin());
  destination->front_ccw = source.front_ccw;
  destination->cull_face = source.cull_face;
  destination->fill_front = source.fill_front;
  destination->fill_back = source.fill_back;
  destination->scissor = source.scissor;
  destination->scissor_x = source.scissor_x;
  destination->scissor_y = source.scissor_y;
  destination->scissor_width = source.scissor_width;
  destination->scissor_height = source.scissor_height;
  destination->line_width_bits = source.line_width_bits;
  destination->point_size_bits = source.point_size_bits;
  destination->point_size_output_start = source.point_size_output_start;
  destination->point_size_output_count = source.point_size_output_count;
  destination->rasterizer_discard = source.rasterizer_discard;
  destination->multisample = source.multisample;
  destination->half_pixel_center = source.half_pixel_center;
  destination->bottom_edge_rule = source.bottom_edge_rule;
  destination->clip_halfz = source.clip_halfz;
  destination->depth_clip_near = source.depth_clip_near;
  destination->depth_clip_far = source.depth_clip_far;
  destination->depth_clamp = source.depth_clamp;
  destination->sample_mask = source.sample_mask;
  if (source.sample_frequency > 1)
    throw std::runtime_error("native command sample frequency is invalid");
  destination->sample_frequency = source.sample_frequency;
  destination->alpha_to_coverage = source.alpha_to_coverage;
  destination->alpha_to_coverage_dither = source.alpha_to_coverage_dither;
  destination->alpha_to_one = source.alpha_to_one;
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
  std::copy_n(source.blend_constant_color_bits, 4,
              destination->blend_constant_color_bits);
  destination->dither = source.dither;
  destination->depth_enable = source.depth_enable;
  destination->depth_write = source.depth_write;
  destination->depth_func = source.depth_func;
  destination->depth_clear_bits = source.depth_clear_bits;
  destination->depth_format = source.depth_format;
  destination->raster_samples = source.raster_samples ? source.raster_samples : 1;
  destination->framebuffer_layers = source.framebuffer_layers;
  destination->stencil_enable = source.stencil_enable;
  destination->stencil_clear = source.stencil_clear;
  for (std::size_t face = 0; face < 2; ++face) {
    destination->stencil_func[face] = source.stencil_func[face];
    destination->stencil_fail_op[face] = source.stencil_fail_op[face];
    destination->stencil_depth_fail_op[face] =
        source.stencil_depth_fail_op[face];
    destination->stencil_pass_op[face] = source.stencil_pass_op[face];
    destination->stencil_value_mask[face] = source.stencil_value_mask[face];
    destination->stencil_write_mask[face] = source.stencil_write_mask[face];
    destination->stencil_ref[face] = source.stencil_ref[face];
  }
  destination->attachment_clears.clear();
  if (source.attachment_clears && source.attachment_clear_count != 0) {
    destination->attachment_clears.reserve(source.attachment_clear_count);
    for (std::uint32_t clear = 0; clear < source.attachment_clear_count;
         ++clear) {
      pvrgpu::stub::DriverAttachmentClear record;
      record.x = source.attachment_clears[clear].x;
      record.y = source.attachment_clears[clear].y;
      record.width = source.attachment_clears[clear].width;
      record.height = source.attachment_clears[clear].height;
      record.aspects = source.attachment_clears[clear].aspects;
      record.depth_bits = source.attachment_clears[clear].depth_bits;
      record.stencil_value = source.attachment_clears[clear].stencil_value;
      destination->attachment_clears.push_back(record);
    }
  }
  destination->color_attachment_source_command_index =
      source.color_attachment_source_command_index;
  destination->depth_attachment_source_command_index =
      source.depth_attachment_source_command_index;
  destination->initial_color_attachment_bytes.clear();
  destination->initial_depth_attachment_bytes.clear();
  if (source.initial_depth_attachment_bytes_size != 0)
    destination->initial_depth_attachment_bytes.assign(
        source.initial_depth_attachment_bytes,
        source.initial_depth_attachment_bytes + source.initial_depth_attachment_bytes_size);
  if (source.initial_color_attachment_bytes_size != 0) {
    destination->initial_color_attachment_bytes.assign(
        source.initial_color_attachment_bytes,
        source.initial_color_attachment_bytes +
            source.initial_color_attachment_bytes_size);
  }
}

bool InitialColorAttachmentIsValid(
    const pvrgpu_systemc_driver_command &source, std::string *error) {
  const auto reject = [&](const char *reason) {
    *error = std::string("SystemC API initial color attachment is invalid: ") +
             reason;
    return false;
  };
  if (!source.initial_color_attachment_bytes &&
      source.initial_color_attachment_bytes_size == 0)
    return true;
  if (!source.initial_color_attachment_bytes ||
      source.initial_color_attachment_bytes_size == 0)
    return reject("pointer/size mismatch");
  if (source.color_attachment_source_command_index !=
      PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR)
    return reject("source must be NEW_CLEAR");
  const std::uint32_t targets = source.render_target_count ? source.render_target_count : 1U;
  if (targets > pvrgpu::stub::kMaxRenderTargets)
    return reject("color target count exceeds the native bound");
  const std::string_view format = source.format ? source.format : "";
  std::uint64_t bytes_per_pixel = 0;
  if (format == "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      format == "PIPE_FORMAT_R10G10B10A2_UNORM" ||
      format == "PIPE_FORMAT_B10G10R10A2_UNORM" ||
      format == "PIPE_FORMAT_R8G8B8A8_SRGB" ||
      format == "PIPE_FORMAT_B8G8R8A8_SRGB" ||
      format == "PIPE_FORMAT_R32_UINT" || format == "PIPE_FORMAT_R32_SINT")
    bytes_per_pixel = 4;
  else if (format == "PIPE_FORMAT_R32G32_UINT" ||
           format == "PIPE_FORMAT_R32G32_SINT")
    bytes_per_pixel = 8;
  else if (format == "PIPE_FORMAT_R32G32B32A32_UINT" ||
           format == "PIPE_FORMAT_R32G32B32A32_SINT" ||
           format == "PIPE_FORMAT_R32G32B32A32_FLOAT")
    bytes_per_pixel = 16;
  else
    return reject("unsupported transport format");
  if (source.framebuffer_width == 0 || source.framebuffer_height == 0 ||
      source.framebuffer_width > 4096 || source.framebuffer_height > 4096)
    return reject("unsupported framebuffer extent");
  const std::uint64_t expected =
      static_cast<std::uint64_t>(source.framebuffer_width) *
      source.framebuffer_height * bytes_per_pixel *
      (source.raster_samples ? source.raster_samples : 1) *
      (source.framebuffer_layers ? source.framebuffer_layers : 1);
  if (expected > pvrgpu::stub::kDriverPcoSequenceAttachmentStride)
    return reject("transport exceeds attachment address slot");
  if (expected * targets != source.initial_color_attachment_bytes_size)
    return reject("byte count does not match framebuffer transport");
  return true;
}

bool InitialDepthAttachmentIsValid(
    const pvrgpu_systemc_driver_command &source, std::string *error) {
  if (!source.initial_depth_attachment_bytes &&
      source.initial_depth_attachment_bytes_size == 0)
    return true;
  if (!source.initial_depth_attachment_bytes ||
      source.depth_attachment_source_command_index != PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR ||
      source.depth_format == 0) {
    *error = "SystemC API initial depth attachment has invalid pointer/source/format";
    return false;
  }
  const std::uint64_t expected = static_cast<std::uint64_t>(source.framebuffer_width) *
      source.framebuffer_height * (source.raster_samples ? source.raster_samples : 1) *
      (source.framebuffer_layers ? source.framebuffer_layers : 1) *
      pvrgpu::stub::DepthAttachmentBytesPerPixel(source.depth_format);
  if (expected == 0 || expected != source.initial_depth_attachment_bytes_size ||
      expected > pvrgpu::stub::kDriverPcoSequenceAttachmentStride) {
    *error = "SystemC API initial depth attachment byte count is invalid";
    return false;
  }
  return true;
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

bool SetPngOutputFromEnvironment(pvrgpu::stub::Options *options,
                                std::string *error) {
  const char *value = std::getenv("PVRGPU_SYSTEMC_DISABLE_PNG");
  if (!value || std::string_view(value) == "0") {
    options->emit_png = true;
    return true;
  }
  if (std::string_view(value) == "1") {
    options->emit_png = false;
    return true;
  }
  *error = "invalid PVRGPU_SYSTEMC_DISABLE_PNG (expected 0 or 1, or unset)";
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

// True when a viewport of the stated extent, centred on the stated offset,
// lies inside the render target.
bool ViewportOffsetIsInside(const std::uint32_t offset_bits[3],
                            std::uint32_t width, std::uint32_t height,
                            std::uint32_t framebuffer_width,
                            std::uint32_t framebuffer_height) {
  float offset[3];
  std::memcpy(offset, offset_bits, sizeof(offset));
  if (!std::isfinite(offset[0]) || !std::isfinite(offset[1]) ||
      !std::isfinite(offset[2]) || offset[2] != 0.5F) {
    return false;
  }
  const float half_width = static_cast<float>(width) * 0.5F;
  const float half_height = static_cast<float>(height) * 0.5F;
  return offset[0] - half_width >= -0.5F &&
         offset[1] - half_height >= -0.5F &&
         offset[0] + half_width <= static_cast<float>(framebuffer_width) + 0.5F &&
         offset[1] + half_height <= static_cast<float>(framebuffer_height) + 0.5F;
}

bool CopyPcoTrianglePayload(
    const pvrgpu_systemc_driver_command &source,
    pvrgpu::stub::DriverCommand *destination, std::string *error) {
  if (!destination || !error)
    return false;
  if (source.geometry_pco || source.geometry_pco_size || source.geometry_shared ||
      source.geometry_shared_count || source.geometry_invocations || source.tessellation) {
    *error = "independent Geometry stage requires the native sequence transport";
    return false;
  }
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
  // The six-float form shares a stride with the lit-mesh profile and is told
  // apart by owning no shared registers; the eight-float form is unambiguous
  // and may carry the draw's constant buffer.
  // Eight floats per vertex is also the texture profile's stride, so the
  // untextured form is told apart by its VTXIN count.
  const bool color_layout =
      (source.vertex_stride ==
           pvrgpu::stub::kDriverPcoPositionNormalVertexStride &&
       source.vertex_pco_abi.shareds == 0) ||
      (!ideas_sequence && source.vertex_stride == 8U * sizeof(float) &&
       source.vertex_pco_abi.vertex_inputs == 8) ||
      // A command that states its attribute widths describes itself.
      (source.vertex_attribute_count != 0 &&
       source.vertex_pco_abi.vertex_inputs ==
           source.vertex_attribute_count * 4U);
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
      !DriverPcoIndexPayloadIsValid(source) ||
      !DriverPcoRenderTargetCountIsValid(source.render_target_count) || expected_vertex_bytes == 0 ||
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
                                                  : 3U),
                                 OpaqueVertexWordMask(source))) {
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
  // Scale is half the viewport extent; the offset places it in the
  // attachment.  A draw rendering to part of its target states an offset that
  // is not the scale, which is only wrong if it leaves the render target.
  const bool viewport_scale_invalid =
      !PcoViewportScaleMatches(viewport_bits, source.viewport_scale_bits);
  const bool viewport_translate_invalid =
      !ViewportOffsetIsInside(source.viewport_translate_bits, source.width,
                              source.height, source.framebuffer_width,
                              source.framebuffer_height);
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
      source.fill_front != 0 || source.fill_back != 0 ||
      source.rasterizer_discard != 0 ||
      // GL_MULTISAMPLE on a single-sample attachment rasterizes as
      // single-sample; a multi-sampled attachment never reaches here.
      source.multisample > 1 || source.half_pixel_center != 1 ||
      source.bottom_edge_rule > 1 || source.clip_halfz != 0 ||
      source.depth_clip_near != 1 || source.depth_clip_far != 1 ||
      source.depth_clamp != 0 || source.sample_mask != UINT32_MAX ||
      source.alpha_to_coverage != 0 || source.alpha_to_one != 0 ||
      source.alpha_to_coverage_dither > 1 ||
      // The PBE honours a partial write mask, so any four-bit mask is valid.
      source.color_mask > 0x0f || source.blend_enable > 1 ||
      // Dither is cosmetic and never applied; either state is valid.
      source.dither > 1;
  const bool ideas_raster_invalid =
      ideas_sequence &&
      ((source.cull_face != 0 && source.cull_face != 2) ||
       !IdeasDepthStateIsSupported(source));
  const bool single_raster_invalid =
      !ideas_sequence &&
      (color_layout
           ? source.cull_face > 3
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
  if (!destination || !error)
    return false;
  /*
   * Name the field that failed.  Bundled into one boolean, every one of these
   * reported the same sentence, so each new colour format or command shape
   * cost a debugging cycle just to tell which half had refused.
   */
  const auto refuse = [&error](const std::string &what) {
    *error = "SystemC API nested PCO sequence draw header is invalid: " + what;
    return false;
  };
  // The outer envelope does not establish the version or byte size of a
  // nested draw. Read only its version before touching any versioned field.
  if (source.version != PVRGPU_SYSTEMC_API_VERSION) {
    return refuse("version=" + std::to_string(source.version) +
                  " expected=" + std::to_string(PVRGPU_SYSTEMC_API_VERSION));
  }
  if (source.uniform_buffer_count >
          5U * PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE ||
      ((source.uniform_buffer_count != 0) != (source.uniform_buffers != nullptr))) {
    *error = "SystemC API uniform buffer payload list is invalid";
    return false;
  }
  if (!source.command ||
      std::string_view(source.command) != "draw_pco_triangles") {
    return refuse(std::string("command=") +
                  (source.command ? source.command : "<none>"));
  }
  if (!source.case_name || !source.case_name[0])
    return refuse("case_name is empty");
  // The colour formats the PBE can write a draw into: four UNORM8 channels
  // (linear or sRGB-encoded), or one, two or four raw 32-bit integer channels.
  if (!source.format ||
      (std::string_view(source.format) != "PIPE_FORMAT_R8G8B8A8_UNORM" &&
       std::string_view(source.format) != "PIPE_FORMAT_R10G10B10A2_UNORM" &&
       std::string_view(source.format) != "PIPE_FORMAT_B10G10R10A2_UNORM" &&
       std::string_view(source.format) != "PIPE_FORMAT_R8G8B8A8_SRGB" &&
       std::string_view(source.format) != "PIPE_FORMAT_B8G8R8A8_SRGB" &&
       std::string_view(source.format) != "PIPE_FORMAT_R32_UINT" &&
       std::string_view(source.format) != "PIPE_FORMAT_R32G32_UINT" &&
       std::string_view(source.format) != "PIPE_FORMAT_R32G32B32A32_UINT" &&
       std::string_view(source.format) != "PIPE_FORMAT_R32_SINT" &&
       std::string_view(source.format) != "PIPE_FORMAT_R32G32_SINT" &&
       std::string_view(source.format) != "PIPE_FORMAT_R32G32B32A32_SINT" &&
       std::string_view(source.format) != "PIPE_FORMAT_R32G32B32A32_FLOAT")) {
    return refuse(std::string("format=") +
                  (source.format ? source.format : "<none>"));
  }
  if (!PcoSequenceTailIsEmpty(source))
    return refuse("the command carries a nested PCO sequence tail");
  const std::uint64_t end_vertex =
      static_cast<std::uint64_t>(source.first_vertex) + source.vertex_count;
  const bool triangles = source.primitive_mode == 4;
  const bool strip_or_fan =
      source.primitive_mode == 5 || source.primitive_mode == 6;
  const bool line_or_point = source.primitive_mode <= 3;
  const bool geometry = source.geometry_pco_size != 0;
  const bool tessellation = source.tessellation != nullptr;
  if (tessellation) {
    if (const char *reason = pvrgpu_tessellation_payload_error(source.tessellation))
      return refuse(reason);
    if (geometry || source.primitive_mode != 14 ||
        source.tessellation->input_stride_dwords != source.vertex_pco_abi.vertex_outputs ||
        (source.indexed ? source.index_count : source.vertex_count) % source.tessellation->vertices_per_instance)
      return refuse("tessellation stage/topology/input linkage");
    if (const char *reason = pvrgpu_tessellation_draw_extent_error(source.tessellation,
          source.indexed ? source.index_count : source.vertex_count))
      return refuse(reason);
  }
  const auto &raster_abi = geometry ? source.geometry_pco_abi
      : tessellation ? source.tessellation->evaluation_abi : source.vertex_pco_abi;
  if (source.stream_output && geometry)
    return refuse("stream output from geometry shaders is not implemented");
  if (!ValidateStreamOutput(source.stream_output, raster_abi.vertex_outputs, error))
    return false;
  if (source.varying_binding_count > PVRGPU_SYSTEMC_MAX_VARYING_BINDINGS ||
      (!source.varying_bindings && source.varying_binding_count))
    return refuse("explicit varying binding list is invalid");
  if (source.varying_bindings) {
    std::uint32_t next_coefficient = 4;
    for (std::uint32_t i = 0; i < source.varying_binding_count; ++i) {
      const auto &b = source.varying_bindings[i];
      if (!b.num_components || b.num_components > 4 || b.flat > 1 ||
          b.output_dword < source.varying_output_start ||
          b.output_dword > raster_abi.vertex_outputs ||
          b.num_components > raster_abi.vertex_outputs - b.output_dword ||
          b.coefficient_dword != next_coefficient)
        return refuse("explicit varying binding output/coefficient range is invalid");
      if (geometry && !b.flat) {
        for (const auto &range : {
                 std::pair{source.geometry_primitive_id_output_start,
                           source.geometry_primitive_id_output_count},
                 std::pair{source.geometry_layer_output_start,
                           source.geometry_layer_output_count}}) {
          if (range.second && std::uint64_t(b.output_dword) <
                                  std::uint64_t(range.first) + range.second &&
              std::uint64_t(range.first) <
                  std::uint64_t(b.output_dword) + b.num_components)
            return refuse("explicit geometry integer varying must be flat");
        }
      }
      next_coefficient += b.num_components * 4;
    }
    if (next_coefficient != source.fragment_pco_abi.coefficients)
      return refuse("explicit varying bindings do not cover fragment coefficients");
  }
  if (geometry) {
    const auto &gs = source.geometry_pco_abi;
    const uint32_t inputs = source.geometry_input_primitive_vertices;
    if (!source.geometry_pco || !source.geometry_shared ||
        source.geometry_pco_size > pvrgpu::stub::kDriverPcoMaximumBinaryBytes ||
        !PcoStageAbiIsBounded(gs, true, true) || gs.coefficients != 0 ||
        gs.vertex_inputs != 2 || gs.vertex_outputs < 4 ||
        gs.uniform_buffer_descriptor_start < 4 ||
        (gs.uniform_buffer_descriptor_start - 4U) % pvrgpu::stub::kPcoTextureDescriptorDwordCount != 0 ||
        (gs.uniform_buffer_descriptor_start - 4U) / pvrgpu::stub::kPcoTextureDescriptorDwordCount >
            pvrgpu::stub::kPcoMaximumTextureDescriptorSets ||
        gs.push_constant_start != gs.uniform_buffer_descriptor_start + 4U * gs.uniform_buffer_descriptor_count ||
        uint64_t(gs.push_constant_start) + gs.push_constant_count != gs.shareds ||
        source.geometry_shared_count != gs.shareds || gs.shareds < 4 ||
        (inputs != 1 && inputs != 2 && inputs != 3 && inputs != 4 && inputs != 6) ||
        source.geometry_input_stride_dwords != source.vertex_pco_abi.vertex_outputs ||
        source.geometry_input_stride_dwords < 4 ||
        source.geometry_input_stride_dwords > 64 ||
        source.geometry_max_vertices > 256 || source.geometry_invocations == 0 ||
        source.geometry_invocations > 32 ||
        (source.geometry_output_primitive != 0 && source.geometry_output_primitive != 3 &&
         source.geometry_output_primitive != 5) ||
        source.geometry_vertices_per_instance == 0 ||
        (source.indexed ? source.index_count : source.vertex_count) %
            source.geometry_vertices_per_instance != 0 ||
        source.geometry_layer_output_count > 1 || source.geometry_primitive_id_output_count > 1 ||
        uint64_t(source.geometry_layer_output_start) + source.geometry_layer_output_count > gs.vertex_outputs ||
        uint64_t(source.geometry_primitive_id_output_start) + source.geometry_primitive_id_output_count > gs.vertex_outputs ||
        source.geometry_shared[0] || source.geometry_shared[1] ||
        source.geometry_shared[2] || source.geometry_shared[3])
      return refuse("geometry ABI, topology or unrelocated input descriptor");
  } else if (source.geometry_pco || source.geometry_shared || source.geometry_shared_count ||
             source.geometry_invocations || source.geometry_input_primitive_vertices) {
    return refuse("geometry payload without executable");
  }
  /*
   * One named check per condition.  As a single bundled boolean this said only
   * that something about the vertex buffer or the topology was wrong, and each
   * refusal then cost a round of guessing; the field is what makes it one run.
   */
  const auto vbo_refusal = [&]() -> const char * {
    if (source.framebuffer_width == 0 || source.framebuffer_height == 0)
      return "framebuffer extent is zero";
    if (source.framebuffer_width > 4096 || source.framebuffer_height > 4096)
      return "framebuffer extent is beyond the model's limit";
    if (source.width == 0 || source.height == 0)
      return "viewport extent is zero";
    if (source.width > 4096 || source.height > 4096)
      return "viewport extent is beyond the model's limit";
    /*
     * One register word is the smallest a vertex can be: an attribute occupies
     * the words its source format does, and a packed one -- four bytes of
     * RGBA8, or a 2_10_10_10 -- is a single word that the shader's own unpack
     * expands.  Requiring two floats assumed every attribute arrived already
     * unpacked to one word per component.
     */
    const bool empty_geometry_attributes = (geometry || tessellation) &&
        source.vertex_attribute_count == 0 && source.vertex_pco_abi.vertex_inputs == 0 &&
        source.vertex_stride == 0 && !source.raw_vertex_data && source.raw_vertex_data_size == 0;
    if (!empty_geometry_attributes && source.vertex_stride < sizeof(std::uint32_t))
      return "vertex stride is below one register word";
    if (source.vertex_stride > 256)
      return "vertex stride is beyond the model's limit";
    if (source.vertex_stride % sizeof(std::uint32_t) != 0)
      return "vertex stride is not a whole number of register words";
    if (source.vertex_count == 0)
      return "vertex count is zero";
    if (source.first_vertex != 0)
      return "first vertex is not zero";
    if (source.instance_count != 1)
      return "instance count is not one";
    if (!triangles && !strip_or_fan && !line_or_point &&
        !(geometry && source.primitive_mode >= 10 && source.primitive_mode <= 13) &&
        !(tessellation && source.primitive_mode == 14))
      return "primitive mode is outside the supported topologies";
    // An indexed draw assembles primitives from its indices.
    if (!geometry && !tessellation && !DriverPcoArrayTopologyIsExpandable(source.primitive_mode,
                                            source.indexed != 0
                                                ? source.index_count
                                                : source.vertex_count)) {
      return "topology cannot be expanded from the element count";
    }
    if (!DriverPcoIndexPayloadIsValid(source))
      return "index payload is invalid";
    if (!DriverPcoRenderTargetCountIsValid(source.render_target_count))
      return "render target count is invalid";
    if ((geometry || tessellation) && source.render_target_count > 1)
      return "geometry MRT requires independent attachment LOAD";
    if (end_vertex == 0 ||
        end_vertex > std::numeric_limits<std::uint32_t>::max() ||
        (source.vertex_stride != 0 &&
         end_vertex > std::numeric_limits<std::uint64_t>::max() /
                         source.vertex_stride)) {
      return "vertex range overflows";
    }
    if (!source.raw_vertex_data && !empty_geometry_attributes)
      return "vertex data is absent";
    if (source.raw_vertex_data_size != end_vertex * source.vertex_stride)
      return "vertex data size does not match the range and stride";
    if (!empty_geometry_attributes && !RawFloatVerticesAreFinite(source.raw_vertex_data, end_vertex,
                                   source.vertex_stride,
                                   source.vertex_stride /
                                       sizeof(std::uint32_t),
                                   OpaqueVertexWordMask(source))) {
      return "a float vertex component is not finite";
    }
    return nullptr;
  }();
  if (vbo_refusal) {
    *error = std::string("SystemC API nested PCO sequence VBO/topology is "
                         "invalid: ") +
             vbo_refusal;
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
      // Position, then gl_PointSize when the shader writes it, then varyings.
      (!geometry && raster_abi.vertex_outputs !=
          source.position_output_count + source.point_size_output_count +
              source.varying_output_count) ||
      (geometry && raster_abi.vertex_outputs <
          source.varying_output_start + source.varying_output_count) ||
      source.varying_output_start !=
          source.position_output_count + source.point_size_output_count ||
      // A shape shaded from a uniform passes no varyings; position still
      // occupies the first four outputs and coefficients.
      source.varying_output_count >
          pvrgpu::stub::kDriverPcoMaximumVaryingComponents ||
      source.fragment_position_start != 0 ||
      source.fragment_position_count != 4 ||
      source.fragment_varying_start != source.fragment_position_count ||
      // The fragment stage interpolates the varyings it reads, which may be
      // fewer than the vertex stage writes.
      source.fragment_varying_count > source.varying_output_count * 4U ||
      (source.fragment_varying_count & 3U) != 0 ||
      source.fragment_pco_abi.coefficients !=
          source.fragment_position_count + source.fragment_varying_count) {
    std::ostringstream detail;
    detail << "SystemC API nested PCO sequence ABI/payload is invalid:"
           << " vs_out=" << source.vertex_pco_abi.vertex_outputs
           << " pos=" << source.position_output_count
           << " psize=" << source.point_size_output_count
           << " var=" << source.varying_output_count
           << " var_start=" << source.varying_output_start
           << " fs_pos=" << source.fragment_position_count
           << " fs_var=" << source.fragment_varying_count
           << " fs_var_start=" << source.fragment_varying_start
           << " fs_coeff=" << source.fragment_pco_abi.coefficients
           << " vs_sh=" << source.vertex_shared_count << "/"
           << source.vertex_pco_abi.shareds
           << " fs_sh=" << source.fragment_shared_count << "/"
           << source.fragment_pco_abi.shareds;
    *error = detail.str();
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
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA &&
      source.blend_destination_rgb_factor <=
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA &&
      source.blend_source_alpha_factor <=
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA &&
      source.blend_destination_alpha_factor <=
          PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
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
          pvrgpu::stub::kDriverPcoDepthFormatZ24X8Unorm ||
      source.depth_format ==
          pvrgpu::stub::kDriverPcoDepthFormatZ24UnormS8Uint ||
      source.depth_format == pvrgpu::stub::kDriverPcoDepthFormatZ32Float ||
      source.depth_format == pvrgpu::stub::kDriverPcoDepthFormatZ32FloatS8X24Uint;
  const bool color_attachment_source_valid =
      source.color_attachment_source_command_index ==
          PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR ||
      source.color_attachment_source_command_index < ordinal;
  const bool depth_attachment_source_valid =
      source.depth_attachment_source_command_index ==
          PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR ||
      source.depth_attachment_source_command_index < ordinal;
  // Name the field that is unsupported: "raster/resource state is invalid"
  // covers two dozen conditions and gives no way to tell which feature a
  // capture actually needs.
  const char *nested_reason = nullptr;
  if (source.sampled_texture_count >
      3U * pvrgpu::stub::kPcoMaximumTextureDescriptorSets)
    nested_reason = "sampled_texture_count";
  else if (source.sampled_texture_bytes ||
           source.sampled_texture_bytes_size != 0 ||
           source.sampled_texture_width != 0 ||
           source.sampled_texture_height != 0 ||
           source.sampled_texture_row_pitch != 0 ||
           source.sampled_texture_format ||
           source.sampled_texture_mip_count != 0)
    nested_reason = "legacy_texture_payload";
  else if (source.front_ccw > 1)
    nested_reason = "front_ccw";
  else if (source.cull_face > 3)
    nested_reason = "cull_face";
  else if (source.fill_front != 0 || source.fill_back != 0)
    nested_reason = "polygon_fill_mode";
  else if (source.rasterizer_discard != 0)
    nested_reason = "rasterizer_discard";
  else if (source.multisample > 1)
    nested_reason = "multisample";
  else if (source.alpha_to_coverage > 1)
    nested_reason = "alpha_to_coverage";
  else if (source.alpha_to_coverage_dither > 1)
    nested_reason = "alpha_to_coverage_dither";
  else if (source.alpha_to_one > 1)
    nested_reason = "alpha_to_one";
  else if (source.half_pixel_center != 1)
    nested_reason = "half_pixel_center";
  else if (source.bottom_edge_rule > 1)
    nested_reason = "bottom_edge_rule";
  else if (source.clip_halfz != 0)
    nested_reason = "clip_halfz";
  else if (source.depth_clip_near != 1 || source.depth_clip_far != 1)
    nested_reason = "depth_clip";
  else if (source.depth_clamp != 0)
    nested_reason = "depth_clamp";
  else if (source.raster_samples > 16 ||
           (source.raster_samples != 0 &&
            (source.raster_samples & (source.raster_samples - 1)) != 0))
    nested_reason = "raster_samples";
  else if (source.framebuffer_layers > 256 ||
           (source.framebuffer_layers && source.render_target_count > 1))
    nested_reason = "framebuffer_layers";
  else if (static_cast<std::uint64_t>(source.framebuffer_width) *
               source.framebuffer_height * (source.raster_samples ? source.raster_samples : 1U) *
               (source.framebuffer_layers ? source.framebuffer_layers : 1U) *
               std::max<std::uint32_t>(
                   (std::string_view(source.format) == "PIPE_FORMAT_R32G32B32A32_UINT" ||
                    std::string_view(source.format) == "PIPE_FORMAT_R32G32B32A32_SINT" ||
                    std::string_view(source.format) == "PIPE_FORMAT_R32G32B32A32_FLOAT") ? 16U :
                   (std::string_view(source.format) == "PIPE_FORMAT_R32G32_UINT" ||
                    std::string_view(source.format) == "PIPE_FORMAT_R32G32_SINT") ? 8U : 4U,
                   depth_format_supported && source.depth_format ?
                       pvrgpu::stub::DepthAttachmentBytesPerPixel(source.depth_format) : 0U) >
           pvrgpu::stub::kDriverPcoSequenceAttachmentStride)
    nested_reason = "framebuffer_attachment_extent";
  else if (source.color_mask > 0x0f)
    nested_reason = "color_mask";
  else if (source.blend_enable > 1 || !blend_enums_valid ||
           !disabled_blend_is_canonical)
    nested_reason = "blend";
  else if (source.dither > 1)
    // Dither is cosmetic and the model never dithers -- an RGBA8 store is
    // exact -- so either GL_DITHER state produces the same pixels.  Accept
    // both; only an out-of-range value is malformed.
    nested_reason = "dither";
  else if (source.depth_enable > 1 || source.depth_write > 1 ||
           (source.depth_write != 0 && source.depth_enable == 0) ||
           source.depth_func > 7)
    nested_reason = "depth_test_state";
  else if (!std::isfinite(depth_clear) || depth_clear < 0.0F ||
           depth_clear > 1.0F)
    nested_reason = "depth_clear_value";
  else if (!depth_format_supported)
    nested_reason = "depth_format";
  else if (!color_attachment_source_valid)
    nested_reason = "color_attachment_source";
  else if (!depth_attachment_source_valid)
    nested_reason = "depth_attachment_source";
  if (nested_reason) {
    std::ostringstream detail;
    detail << "SystemC API nested PCO sequence state is unsupported: "
           << nested_reason;
    if (std::string_view(nested_reason) == "depth_format")
      detail << " (" << source.depth_format << ")";
    *error = detail.str();
    return false;
  }
  if (!InitialColorAttachmentIsValid(source, error) ||
      !InitialDepthAttachmentIsValid(source, error))
    return false;

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
  for (std::uint32_t index = 0; index < source.uniform_buffer_count; ++index) {
    const auto &buffer = source.uniform_buffers[index];
    if (buffer.stage > PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION ||
        buffer.block_index >= PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE ||
        !buffer.bytes || buffer.bytes_size == 0 ||
        buffer.bytes_size > PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES) {
      *error = "SystemC API uniform buffer stage/index/bytes is invalid";
      return false;
    }
    pvrgpu::stub::DriverPcoUniformBuffer owned;
    owned.stage = static_cast<pvrgpu::stub::DriverPcoShaderStage>(buffer.stage);
    owned.block_index = buffer.block_index;
    owned.bytes.assign(buffer.bytes, buffer.bytes + buffer.bytes_size);
    command.uniform_buffers.push_back(std::move(owned));
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
  *destination = std::move(command);
  return true;
}

bool CopyCommand(const pvrgpu_systemc_driver_command &source,
                 pvrgpu::stub::DriverCommand *destination,
                 std::string *error) {
  if (!destination || !error)
    return false;
  // Keep the helper safe independently of the public entry-point check.
  if (source.version != PVRGPU_SYSTEMC_API_VERSION) {
    *error = "unsupported SystemC API command version";
    return false;
  }
  if (source.fragment_images || source.fragment_image_count ||
      source.fragment_image_descriptor_start || source.fragment_image_descriptor_count ||
      source.fragment_image_read_mask || source.fragment_image_write_mask || source.fragment_early_tests) {
    *error = "SystemC API fragment images/early tests require a nested PCO draw";
    return false;
  }
  if (source.uniform_buffer_count || source.uniform_buffers ||
      source.vertex_pco_abi.uniform_buffer_descriptor_start ||
      source.vertex_pco_abi.uniform_buffer_descriptor_count ||
      source.fragment_pco_abi.uniform_buffer_descriptor_start ||
      source.fragment_pco_abi.uniform_buffer_descriptor_count) {
    *error = "SystemC API uniform buffers require a nested PCO draw";
    return false;
  }
  if (source.stream_output || source.varying_bindings || source.varying_binding_count) {
    *error = "SystemC API stream output/explicit varying bindings require a nested PCO draw";
    return false;
  }
  if (source.framebuffer_layers) {
    *error = "SystemC API layered framebuffer requires a nested PCO draw";
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
  if (source.initial_color_attachment_bytes ||
      source.initial_color_attachment_bytes_size != 0 ||
      source.initial_depth_attachment_bytes ||
      source.initial_depth_attachment_bytes_size != 0) {
    *error = "SystemC API initial color attachment requires a nested PCO draw";
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
      command.vertex_pco.size() + command.fragment_pco.size() + command.geometry_pco.size() +
      command.tessellation.control_pco.size() + command.tessellation.evaluation_pco.size() +
      command.sampled_texture_bytes.size() + command.texture_rgba8_bytes.size() +
      command.initial_color_attachment_bytes.size() +
      command.initial_depth_attachment_bytes.size();
  for (const pvrgpu::stub::DriverPcoSampledTexture &texture :
       command.sampled_textures) {
    if (texture.bytes.size() >
        std::numeric_limits<std::uint64_t>::max() - byte_vectors) {
      throw std::overflow_error("SystemC API sequence payload size overflow");
    }
    byte_vectors += texture.bytes.size();
  }
  for (const auto &buffer : command.uniform_buffers) {
    if (buffer.bytes.size() >
        std::numeric_limits<std::uint64_t>::max() - byte_vectors)
      throw std::overflow_error("SystemC API uniform buffer payload size overflow");
    byte_vectors += buffer.bytes.size();
  }
  for (const auto &target : command.stream_output.targets) {
    if (target.bytes.size() > std::numeric_limits<std::uint64_t>::max() - byte_vectors)
      throw std::overflow_error("SystemC API stream output payload size overflow");
    byte_vectors += target.bytes.size();
  }
  for (const auto &image : command.fragment_images) {
    if (image.bytes.size() > std::numeric_limits<std::uint64_t>::max() - byte_vectors)
      throw std::overflow_error("SystemC API fragment image payload size overflow");
    byte_vectors += image.bytes.size();
  }
  const std::uint64_t dword_count =
      static_cast<std::uint64_t>(command.vertex_shared.size()) +
      command.fragment_shared.size() + command.geometry_shared.size() +
      command.tessellation.control_shared.size() + command.tessellation.evaluation_shared.size();
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
    std::string *error, pvrgpu::stub::TextureResource *compute_resource) {
  if (!destination || !error)
    return false;
  const std::string_view format = source.format ? source.format : "";
  // A combined depth/stencil image sampled through a 2D view stores the same
  // four bytes per texel as a colour image; the depth occupies the low 24
  // bits and the texture unit masks the stencil byte off.
  /*
   * A texture's storage unit, not its texel.
   *
   * An uncompressed image stores one four-byte texel per position, so its
   * block is 1x1.  ASTC stores one sixteen-byte block per footprint, so a row
   * is ceil(width / block width) blocks and a level is ceil(height / block
   * height) of those rows.  Expressing both as a block removes the "four
   * bytes per texel" the layout check below used to assume -- which is the
   * only reason a compressed image could not be described here.
   *
   * An sRGB image stores exactly what its non-sRGB form does; only the
   * texture unit's decode of R, G and B differs.
   */
  struct TextureStorageBlock {
    const char *format;
    std::uint32_t width;
    std::uint32_t height;
  };
  static constexpr TextureStorageBlock kAstcBlocks[] = {
      {"PIPE_FORMAT_ASTC_4x4", 4, 4},
      {"PIPE_FORMAT_ASTC_4x4_SRGB", 4, 4},
      {"PIPE_FORMAT_ASTC_5x4", 5, 4},
      {"PIPE_FORMAT_ASTC_5x4_SRGB", 5, 4},
      {"PIPE_FORMAT_ASTC_5x5", 5, 5},
      {"PIPE_FORMAT_ASTC_5x5_SRGB", 5, 5},
      {"PIPE_FORMAT_ASTC_6x5", 6, 5},
      {"PIPE_FORMAT_ASTC_6x5_SRGB", 6, 5},
      {"PIPE_FORMAT_ASTC_6x6", 6, 6},
      {"PIPE_FORMAT_ASTC_6x6_SRGB", 6, 6},
      {"PIPE_FORMAT_ASTC_8x5", 8, 5},
      {"PIPE_FORMAT_ASTC_8x5_SRGB", 8, 5},
      {"PIPE_FORMAT_ASTC_8x6", 8, 6},
      {"PIPE_FORMAT_ASTC_8x6_SRGB", 8, 6},
      {"PIPE_FORMAT_ASTC_8x8", 8, 8},
      {"PIPE_FORMAT_ASTC_8x8_SRGB", 8, 8},
      {"PIPE_FORMAT_ASTC_10x5", 10, 5},
      {"PIPE_FORMAT_ASTC_10x5_SRGB", 10, 5},
      {"PIPE_FORMAT_ASTC_10x6", 10, 6},
      {"PIPE_FORMAT_ASTC_10x6_SRGB", 10, 6},
      {"PIPE_FORMAT_ASTC_10x8", 10, 8},
      {"PIPE_FORMAT_ASTC_10x8_SRGB", 10, 8},
      {"PIPE_FORMAT_ASTC_10x10", 10, 10},
      {"PIPE_FORMAT_ASTC_10x10_SRGB", 10, 10},
      {"PIPE_FORMAT_ASTC_12x10", 12, 10},
      {"PIPE_FORMAT_ASTC_12x10_SRGB", 12, 10},
      {"PIPE_FORMAT_ASTC_12x12", 12, 12},
      {"PIPE_FORMAT_ASTC_12x12_SRGB", 12, 12},
  };
  std::uint32_t block_width = 1U;
  std::uint32_t block_height = 1U;
  std::uint32_t block_bytes = 0U;
  pvrgpu::stub::TextureFormat texture_format =
      pvrgpu::stub::TextureFormat::kRgba8Unorm;
  if (format == "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      format == "PIPE_FORMAT_B8G8R8A8_UNORM" ||
      format == "PIPE_FORMAT_R8G8B8X8_UNORM" ||
      format == "PIPE_FORMAT_R8G8B8A8_SRGB" ||
      format == "PIPE_FORMAT_Z32_UNORM" ||
      format == "PIPE_FORMAT_Z24_UNORM_S8_UINT") {
    block_bytes = 4U;
    texture_format =
        format == "PIPE_FORMAT_B8G8R8A8_UNORM"
            ? pvrgpu::stub::TextureFormat::kBgra8Unorm
        : format == "PIPE_FORMAT_R8G8B8X8_UNORM"
            ? pvrgpu::stub::TextureFormat::kRgbx8Unorm
        : format == "PIPE_FORMAT_R8G8B8A8_SRGB"
            ? pvrgpu::stub::TextureFormat::kRgba8Srgb
        : format == "PIPE_FORMAT_Z32_UNORM"
            ? pvrgpu::stub::TextureFormat::kZ32Unorm
        : format == "PIPE_FORMAT_Z24_UNORM_S8_UINT"
            ? pvrgpu::stub::TextureFormat::kZ24UnormS8Uint
            : pvrgpu::stub::TextureFormat::kRgba8Unorm;
  } else if (format == "PIPE_FORMAT_B5G6R5_UNORM" ||
             format == "PIPE_FORMAT_R5G6B5_UNORM") {
    // GL RGB565 storage is one little-endian uint16 (R in the high five
    // bits).  Mesa hands the driver B5G6R5 or R5G6B5 depending on the host;
    // the stored bytes are identical.
    block_bytes = 2U;
    texture_format = pvrgpu::stub::TextureFormat::kRgb565Unorm;
  } else if (format == "PIPE_FORMAT_R10G10B10A2_UNORM" ||
             format == "PIPE_FORMAT_B10G10R10A2_UNORM") {
    block_bytes = 4U;
    texture_format = format == "PIPE_FORMAT_B10G10R10A2_UNORM"
        ? pvrgpu::stub::TextureFormat::kBgr10A2Unorm
        : pvrgpu::stub::TextureFormat::kRgb10A2Unorm;
  } else if (format == "PIPE_FORMAT_R8G8B8A8_SNORM") {
    block_bytes = 4U;
    texture_format = pvrgpu::stub::TextureFormat::kRgba8Snorm;
  } else if (format == "PIPE_FORMAT_R16G16B16A16_FLOAT") {
    block_bytes = 8U;
    texture_format = pvrgpu::stub::TextureFormat::kRgba16Float;
  } else if (format == "PIPE_FORMAT_R32G32B32A32_UINT" ||
             format == "PIPE_FORMAT_R32G32B32A32_SINT") {
    block_bytes = 16U;
    texture_format = format == "PIPE_FORMAT_R32G32B32A32_UINT"
        ? pvrgpu::stub::TextureFormat::kRgba32Uint
        : pvrgpu::stub::TextureFormat::kRgba32Sint;
  } else if (format == "PIPE_FORMAT_R32G32B32A32_FLOAT") {
    block_bytes = 16U;
    texture_format = pvrgpu::stub::TextureFormat::kRgba32Float;
  } else if (format == "PIPE_FORMAT_R11G11B10_FLOAT") {
    block_bytes = 4U;
    texture_format = pvrgpu::stub::TextureFormat::kR11fG11fB10f;
  } else if (format == "PIPE_FORMAT_R9G9B9E5_FLOAT") {
    block_bytes = 4U;
    texture_format = pvrgpu::stub::TextureFormat::kRgb9e5Float;
  } else {
    for (const TextureStorageBlock &block : kAstcBlocks) {
      if (format == block.format) {
        block_width = block.width;
        block_height = block.height;
        block_bytes = 16U;  // every ASTC block is 128 bits
        texture_format =
            std::string_view(block.format).size() > 5 &&
                    std::string_view(block.format).substr(
                        std::string_view(block.format).size() - 5) == "_SRGB"
                ? pvrgpu::stub::TextureFormat::kAstcLdrSrgb
                : pvrgpu::stub::TextureFormat::kAstcLdr;
        break;
      }
    }
  }
  const auto blocks_for = [](std::uint32_t extent, std::uint32_t block) {
    return (static_cast<std::uint64_t>(extent) + block - 1U) / block;
  };
  // Each field names itself: a bundled predicate here only reports that some
  // unspecified part of the metadata was rejected.
  const auto reject = [&](const char *field) {
    *error = std::string("SystemC API PCO sequence texture ") + field +
             " is invalid";
    return false;
  };
  if (source.source > PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_DEPTH_ATTACHMENT)
    return reject("source");
  if (compute_resource ? source.stage != PVRGPU_SYSTEMC_PCO_SHADER_STAGE_COMPUTE
                       : source.stage > PVRGPU_SYSTEMC_PCO_SHADER_STAGE_GEOMETRY)
    return reject("stage");
  if (source.descriptor_set >=
      pvrgpu::stub::kPcoMaximumTextureDescriptorSets)
    return reject("descriptor set");
  if (source.binding != 0)
    return reject("binding");
  if (block_bytes == 0)
    return reject("format");
  if (source.declared_bytes_size == 0 ||
      source.declared_bytes_size >
          pvrgpu::stub::kDriverPcoMaximumSequencePayloadBytes)
    return reject("declared byte size");
  if (source.mip_count == 0 ||
      source.mip_count > PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS)
    return reject("mip count");
  const std::uint32_t samples = source.sample_count ? source.sample_count : 1U;
  if (samples != 1U && samples != 2U && samples != 4U && samples != 8U)
    return reject("sample count");
  if (source.texture_kind > 3U || source.layers > UINT16_MAX ||
      (source.texture_kind == 0U && source.layers > 1U))
    return reject("dimension/layer count");
  if (samples > 1U &&
      (source.source != PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD ||
       source.texture_kind > 1U || source.mip_count != 1U ||
       block_width != 1U || block_height != 1U))
    return reject("multisample external 2D single-mip layout");
  if (source.min_filter > PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR)
    return reject("min filter");
  if (source.mag_filter > PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR)
    return reject("mag filter");
  if (source.mip_filter > PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_LINEAR)
    return reject("mip filter");
  if (source.wrap_u > PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_MIRRORED_REPEAT)
    return reject("wrap u");
  if (source.wrap_v > PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_MIRRORED_REPEAT)
    return reject("wrap v");
  if (source.normalized_coordinates != 1)
    return reject("normalized coordinates");
  if (source.min_lod_u4_6 > source.max_lod_u4_6)
    return reject("lod range");

  std::uint64_t previous_end = 0;
  for (std::size_t level = 0; level < source.mip_count; ++level) {
    const pvrgpu_systemc_pco_texture_mip &mip = source.mip[level];
    const std::uint64_t tight_pitch =
        blocks_for(mip.width, block_width) * block_bytes * samples;
    // A 2D array stores `layers` images per level, layer-minor: the level
    // spans that many single-image byte sizes.  A 3D image (kind 2) stores
    // `depth` slices the same way, but its slice count halves with each level.
    const std::uint64_t base_slices =
        source.layers == 0U ? 1U : static_cast<std::uint64_t>(source.layers);
    const std::uint64_t level_slices =
        source.texture_kind == 2U
            ? ((base_slices >> level) == 0U ? 1U : (base_slices >> level))
            : base_slices;
    const std::uint64_t rows = blocks_for(mip.height, block_height);
    if (rows && level_slices &&
        mip.row_pitch > source.declared_bytes_size / rows / level_slices)
      return reject("mip byte extent");
    const std::uint64_t level_bytes =
        static_cast<std::uint64_t>(mip.row_pitch) * rows * level_slices;
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
    /*
     * Decline anything the texture unit cannot sample, here, before the draw
     * is claimed.
     *
     * The capability question has one answer, and it lives in
     * DriverPcoTextureDescriptorClassSupported.  Asking it from here as well
     * turns what would be a fatal exception in the middle of SampleRun -- a
     * killed process and a NoResult that tells nobody anything -- into a
     * named refusal the driver can act on.
     */
    {
      pvrgpu::stub::RogueTextureImageDescriptor probe_image;
      probe_image.width = source.mip[0].width;
      probe_image.height = source.mip[0].height;
      probe_image.row_pitch_bytes = source.mip[0].row_pitch;
      probe_image.mip_count = static_cast<std::uint8_t>(source.mip_count);
      probe_image.sample_count = static_cast<std::uint8_t>(samples);
      probe_image.format = texture_format;
      pvrgpu::stub::RogueTextureSamplerDescriptor probe_sampler;
      probe_sampler.min_filter = static_cast<pvrgpu::stub::TextureFilter>(
          source.min_filter);
      probe_sampler.mag_filter = static_cast<pvrgpu::stub::TextureFilter>(
          source.mag_filter);
      probe_sampler.mip_filter = static_cast<pvrgpu::stub::TextureFilter>(
          source.mip_filter);
      const auto probe_wrap = [](std::uint32_t capsule_wrap) {
        switch (capsule_wrap) {
        case PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_REPEAT:
          return pvrgpu::stub::TextureWrapMode::kRepeat;
        case PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_MIRRORED_REPEAT:
          return pvrgpu::stub::TextureWrapMode::kMirroredRepeat;
        default:
          return pvrgpu::stub::TextureWrapMode::kClampToEdge;
        }
      };
      probe_sampler.wrap_u = probe_wrap(source.wrap_u);
      probe_sampler.wrap_v = probe_wrap(source.wrap_v);
      probe_sampler.min_lod_u4_6 =
          static_cast<std::uint16_t>(source.min_lod_u4_6);
      probe_sampler.max_lod_u4_6 =
          static_cast<std::uint16_t>(source.max_lod_u4_6);
      probe_sampler.normalized_coordinates = 1;
      if (!pvrgpu::stub::DriverPcoTextureDescriptorClassSupported(
              probe_image, probe_sampler, 1U)) {
        *error =
            std::string("SystemC API PCO sequence texture cannot be sampled: "
                        "format=") + std::string(format) +
            " mips=" + std::to_string(source.mip_count) +
            " filters=" + std::to_string(source.min_filter) + "/" +
            std::to_string(source.mag_filter) + "/" +
            std::to_string(source.mip_filter) +
            " lod=" + std::to_string(source.min_lod_u4_6) + ".." +
            std::to_string(source.max_lod_u4_6);
        return false;
      }
    }
    /*
     * A compressed image's footprint has to be one the decoder covers, and
     * that is knowable here, before the draw is claimed.
     *
     * The block contents are not: the decoder answers every 128-bit LDR
     * block, invalid encodings included -- the spec gives those the error
     * colour, which is what the hardware returns too -- so there is nothing
     * about the bytes left to decline.  What would still be undecodable is a
     * footprint this decoder does not have, and refusing that in the middle
     * of a sample would abort the simulation and report NoResult, which
     * tells nobody anything.
     */
    if (block_width != 1U || block_height != 1U) {
      const pvrgpu::stub::AstcBlockFootprint footprint{block_width,
                                                       block_height};
      pvrgpu::stub::AstcDecodedBlock decoded;
      const std::uint8_t probe_block[16] = {};
      const char *refusal = nullptr;
      if (!pvrgpu::stub::DecodeAstcBlock(probe_block, footprint,
                                         /*srgb=*/false, &decoded, &refusal)) {
        *error = std::string("SystemC API PCO sequence compressed footprint ") +
                 std::to_string(block_width) + "x" +
                 std::to_string(block_height) + " cannot be decoded: " +
                 (refusal != nullptr ? refusal : "unstated");
        return false;
      }
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
  if (compute_resource) {
    auto &resource = *compute_resource;
    resource = {};
    resource.format = texture_format;
    resource.block_width = block_width;
    resource.block_height = block_height;
    resource.byte_size = source.declared_bytes_size;
    resource.mip_count = source.mip_count;
    resource.sample_count = samples;
    resource.layer_count = source.layers ? source.layers : 1U;
    resource.dimension_type = static_cast<pvrgpu::stub::TextureDimensionType>(source.texture_kind);
    resource.descriptor_set = source.descriptor_set;
    resource.binding = source.binding;
    for (std::size_t level = 0; level < source.mip_count; ++level)
      resource.mip[level] = {source.mip[level].width, source.mip[level].height,
                            source.mip[level].row_pitch, source.mip[level].offset};
  }
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
  texture.texture_kind = source.texture_kind;
  texture.layers = source.layers == 0U ? 1U : source.layers;
  texture.sample_count = samples;
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
  bool saw_stream_output = false;
  std::unordered_map<std::uint64_t, const pvrgpu::stub::DriverShaderImage *> image_snapshots;
  for (std::size_t ordinal = 0;
       ordinal < source.pco_sequence_command_count; ++ordinal) {
    pvrgpu::stub::DriverCommand command;
    if (!CopyPcoSequenceDraw(source.pco_sequence_commands[ordinal], ordinal,
                             &command, error)) {
      return false;
    }
    if (!command.stream_output.bindings.empty()) {
      if (saw_stream_output) {
        *error = "SystemC API sequence requires a completion between stream output draws";
        return false;
      }
      saw_stream_output = true;
    }
    const auto attachment_matches = [&](std::uint32_t source_ordinal,
                                        bool depth) {
      if (source_ordinal == pvrgpu::stub::kDriverPcoNewAttachment)
        return true;
      const pvrgpu::stub::DriverCommand &producer = commands[source_ordinal];
      return producer.framebuffer_width == command.framebuffer_width &&
             producer.framebuffer_height == command.framebuffer_height &&
             producer.raster_samples == command.raster_samples &&
             producer.framebuffer_layers == command.framebuffer_layers &&
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
    std::array<bool, pvrgpu::stub::kPcoMaximumTextureDescriptorSets> geometry_sets{};
    auto previous_stage = pvrgpu::stub::DriverPcoShaderStage::kVertex;
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
              : texture.stage == pvrgpu::stub::DriverPcoShaderStage::kGeometry ? geometry_sets : fragment_sets;
      if (texture.stage < previous_stage) {
        *error =
            "SystemC API PCO sequence texture stages are not VS-then-FS-then-GS";
        return false;
      }
      previous_stage = texture.stage;
      if (sets[texture.descriptor_set]) {
        *error = "SystemC API PCO sequence descriptor set is duplicated";
        return false;
      }
      sets[texture.descriptor_set] = true;
      if (texture.stage == pvrgpu::stub::DriverPcoShaderStage::kVertex)
        ++command.vertex_sampled_texture_count;
      else if (texture.stage == pvrgpu::stub::DriverPcoShaderStage::kGeometry)
        ++command.geometry_sampled_texture_count;
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
                        command.fragment_sampled_texture_count) ||
        !sets_are_dense(geometry_sets, command.geometry_sampled_texture_count)) {
      *error =
          "SystemC API PCO sequence descriptor sets are not stage-dense";
      return false;
    }
    const std::size_t descriptor_dwords =
        pvrgpu::stub::kPcoTextureDescriptorDwordCount;
    if (command.vertex_shared.size() <
            command.vertex_sampled_texture_count * descriptor_dwords ||
        command.fragment_shared.size() <
            command.fragment_sampled_texture_count * descriptor_dwords ||
        (command.geometry_sampled_texture_count &&
         (command.geometry_pco.empty() || command.geometry_shared.size() <
            4U + command.geometry_sampled_texture_count * descriptor_dwords)) ||
        (!command.geometry_pco.empty() && command.geometry_pco_abi.uniform_buffer_descriptor_start !=
            4U + command.geometry_sampled_texture_count * descriptor_dwords)) {
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
      const auto &shared = texture.stage == pvrgpu::stub::DriverPcoShaderStage::kVertex
                               ? command.vertex_shared
                               : texture.stage == pvrgpu::stub::DriverPcoShaderStage::kGeometry
                                   ? command.geometry_shared : command.fragment_shared;
      // Rogue TEXSTATE_IMAGE_WORD0[63:62] 保存 log2(samples)，與結構化
      // payload 必須一致，不能由 host metadata 蓋過 shader descriptor。
      const std::size_t image_word1 = texture.descriptor_set * descriptor_dwords + 1U +
          (texture.stage == pvrgpu::stub::DriverPcoShaderStage::kGeometry ? 4U : 0U);
      const std::uint32_t descriptor_samples = 1U << (shared.at(image_word1) >> 30U);
      if (descriptor_samples != texture.sample_count) {
        *error = "SystemC API PCO sequence texture descriptor/sample count mismatch";
        return false;
      }
      if (texture.source ==
          pvrgpu::stub::DriverPcoTextureSource::kExternalPayload) {
        continue;
      }
      const pvrgpu::stub::DriverCommand &producer =
          commands.at(texture.producer_command_index);
      /* Ordinary sampled images address one texel per pixel.  Multisample
       * attachments instead interleave every sample inside each pixel; an
       * alias is not a resolve and this capsule has no sample-index operand. */
      if (producer.raster_samples > 1U) {
        *error = "SystemC API PCO sequence sampled attachment requires a "
                 "single-sample producer";
        return false;
      }
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
    if (!pvrgpu::stub::ValidateDriverUniformBuffers(command, error) ||
        !pvrgpu::stub::ValidateDriverShaderImages(command, error))
      return false;
    for (const auto &image : command.fragment_images) {
      const auto [entry, inserted] = image_snapshots.emplace(image.resource_token, &image);
      if (!inserted && entry->second->bytes != image.bytes) {
        *error = "SystemC API fragment image snapshot changed within one sequence";
        return false;
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

/*
 * Paths this process has already opened for a model run.
 *
 * A case that reads back three times runs the model three times, and each run
 * writes its own hello/counter/done record set.  Truncating per run would
 * leave only the last of them, so the first run of a path truncates and every
 * run after it appends.  Consumers of `systemc.jsonl` therefore have to accept
 * more than one record set per case.
 */
std::set<std::string> g_opened_jsonl_paths;
std::set<std::string> g_opened_stderr_paths;

std::ios::openmode ModelOutputMode(std::set<std::string> *opened,
                                   const std::string &path) {
  if (opened->insert(path).second)
    return std::ios::out | std::ios::trunc;
  return std::ios::out | std::ios::app;
}

int RunModelToFiles(const pvrgpu::stub::Options &options,
                    const std::string &jsonl_path,
                    const std::string &stderr_path,
                    pvrgpu::stub::ModelFramebuffer *framebuffer,
                    std::string *error) {
  std::ofstream jsonl(jsonl_path,
                      ModelOutputMode(&g_opened_jsonl_paths, jsonl_path));
  if (!jsonl) {
    if (error)
      *error = "cannot open SystemC API jsonl_path: " + jsonl_path;
    return 1;
  }
  std::ofstream stderr_file;
  if (!stderr_path.empty()) {
    stderr_file.open(stderr_path,
                     ModelOutputMode(&g_opened_stderr_paths, stderr_path));
    if (!stderr_file) {
      if (error)
        *error = "cannot open SystemC API stderr_path: " + stderr_path;
      return 1;
    }
  }

  std::streambuf *old_stdout = std::cout.rdbuf(jsonl.rdbuf());
  std::streambuf *old_stderr =
      stderr_file ? std::cerr.rdbuf(stderr_file.rdbuf()) : nullptr;
  int result = 0;
  try {
    result = pvrgpu::stub::RunConfiguredModel(options, framebuffer);
  } catch (...) {
    // A rejected native instruction/state can throw out of SystemC. Restore
    // both buffers before local streams die so the caller retains diagnostics.
    std::cout.rdbuf(old_stdout);
    if (old_stderr) std::cerr.rdbuf(old_stderr);
    throw;
  }
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

/*
 * Recompute a sequence's input-assembly totals from the draws it accumulated.
 *
 * The driver states these on the first draw it submits, which for a sequence
 * built up over many submissions is before it can know the totals.  They used
 * to be constants recorded from a golden run; deriving them here means the
 * numbers follow the geometry the sequence actually carries, and no stage has
 * to be told them in advance.
 */
void DeriveSequenceInputAssembly(pvrgpu::stub::Options *options) {
  if (!options || options->driver_commands.empty())
    return;
  std::uint64_t vertices = 0;
  std::uint64_t primitives = 0;
  bool any_indexed = false;
  bool any_geometry = false;
  for (const pvrgpu::stub::DriverCommand &draw : options->driver_commands) {
    const bool geometry = !draw.geometry_pco.empty();
    const bool tessellation = !draw.tessellation.control_pco.empty();
    any_geometry = any_geometry || geometry || tessellation;
    const std::uint64_t total_occurrences = draw.indexed ? draw.index_count : draw.vertex_count;
    if (geometry && (!draw.geometry_vertices_per_instance ||
                     total_occurrences % draw.geometry_vertices_per_instance))
      throw std::runtime_error("invalid GS instance assembly extent");
    const std::uint64_t assembled =
        tessellation ? draw.tessellation.vertices_per_instance
        : geometry ? draw.geometry_vertices_per_instance : total_occurrences;
    const std::uint64_t instances =
        (geometry || tessellation) ? total_occurrences / assembled : draw.instance_count != 0 ? draw.instance_count : 1U;
    any_indexed = any_indexed || draw.indexed != 0;
    vertices += assembled * instances;
    switch (draw.primitive_mode) {
      case 0U: primitives += assembled * instances; break;
      case 1U: primitives += (assembled / 2U) * instances; break;
      case 2U: primitives += assembled * instances; break;
      case 3U:
        primitives += (assembled >= 2U ? assembled - 1U : 0U) * instances;
        break;
      case 4U: primitives += (assembled / 3U) * instances; break;
      case 5U:
      case 6U:
        primitives += (assembled >= 3U ? assembled - 2U : 0U) * instances;
        break;
      case 10U: primitives += (assembled / 4U) * instances; break;
      case 11U: primitives += (assembled >= 4U ? assembled - 3U : 0U) * instances; break;
      case 12U: primitives += (assembled / 6U) * instances; break;
      case 13U: primitives += (assembled >= 6U ? (assembled - 4U) / 2U : 0U) * instances; break;
      case 14U: primitives += (assembled / draw.tessellation.input_vertices) * instances; break;
      default:
        return;  // An unknown topology is rejected by the profile checks.
    }
  }
  pvrgpu::stub::DriverCommand &logical = options->driver_command;
  logical.draw_count =
      static_cast<std::uint32_t>(options->driver_commands.size());
  logical.ia_vertices = static_cast<std::uint32_t>(vertices);
  logical.ia_primitives = static_cast<std::uint32_t>(primitives);
  logical.clip_invocations = any_geometry ? 0U : static_cast<std::uint32_t>(primitives);
  // Vertex shading follows the vertex count exactly unless post-transform
  // reuse decides it, which only an indexed draw can do.
  logical.vs_invocations =
      any_indexed ? 0U : static_cast<std::uint32_t>(vertices);
}

/*
 * What the last flush left behind.  A pass writing several colour
 * attachments is read back one attachment at a time, and the flush that
 * produced them is consumed by the first of those reads, so the attachments
 * are held here until the next submission runs and replaces them.
 */
pvrgpu::stub::ModelFramebuffer g_last_framebuffer;
pvrgpu::stub::ModelGraphicsStats g_last_graphics_stats;
std::uint64_t g_last_graphics_generation = 0;

int FlushPendingSubmitLocked(pvrgpu::stub::ModelFramebuffer *framebuffer,
                             std::string *error) {
  if (!g_pending_submit.valid || g_pending_submit.executed) {
    if (framebuffer && g_last_framebuffer.valid())
      *framebuffer = g_last_framebuffer;
    return 0;
  }
  g_pending_submit.executed = true;
  if (IsIdeasPcoSequenceRoot(g_pending_submit.options.driver_command) &&
      g_pending_submit.options.driver_commands.size() !=
          pvrgpu::stub::kDriverPcoIdeasSequenceCommands) {
    if (error)
      *error = "Ideas PCO profile requires exactly 180 ordered draws";
    return 2;
  }
  DeriveSequenceInputAssembly(&g_pending_submit.options);
  pvrgpu::stub::ModelFramebuffer produced;
  const int status =
      RunModelToFiles(g_pending_submit.options, g_pending_submit.jsonl_path,
                      g_pending_submit.stderr_path, &produced, error);
  if (status == 0) {
    g_last_graphics_stats = produced.graphics_stats;
    g_last_graphics_generation = g_pending_submit.submission_generation;
    g_last_framebuffer = std::move(produced);
    if (framebuffer)
      *framebuffer = g_last_framebuffer;
  }
  return status;
}

/*
 * Last resort for work nobody read back.
 *
 * The explicit flush is the normal path now, but a submission that is never
 * mapped for read -- a frame the application only presents, say -- would
 * otherwise never run at all, and its counters and PNG would disappear.  So
 * this stays, and runs only what the readback path left behind.
 */
void FlushPendingSubmitAtExit() {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  std::string error;
  const int result = FlushPendingSubmitLocked(nullptr, &error);
  if (result != 0) {
    std::cerr << "PvrGPU SystemC API deferred flush failed: " << error
              << '\n';
  }
  pvrgpu::stub::ShutdownConfiguredModel();
}

}  // namespace

extern "C" int pvrgpu_systemc_submit_compute(
    const pvrgpu_systemc_compute_dispatch *dispatch,
    pvrgpu_systemc_compute_stats *stats, char *error, std::size_t error_size) {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  CopyError(error, error_size, "");
  // An old or truncated envelope is permitted to expose only its version.
  if (!dispatch || dispatch->version != PVRGPU_SYSTEMC_COMPUTE_API_VERSION) {
    CopyError(error, error_size, "unsupported SystemC compute API version");
    return 2;
  }
  // Stats layout is versioned with the envelope; do not overwrite an older
  // caller's smaller output structure on a version mismatch.
  if (stats)
    *stats = {};
  try {
    pvrgpu::stub::ModelComputeDispatch prepared;
    std::string diagnostic;
    if (!PrepareComputeDispatch(*dispatch, &prepared, &diagnostic)) {
      CopyError(error, error_size, diagnostic);
      return 2;
    }
    // A pending draw must finish before compute begins on the shared memory
    // service. The driver materializes attachment resources before snapshotting.
    const int pending = FlushPendingSubmitLocked(nullptr, &diagnostic);
    if (pending != 0) {
      CopyError(error, error_size, diagnostic);
      return pending;
    }
    if (!g_atexit_registered) {
      if (std::atexit(FlushPendingSubmitAtExit) != 0) {
        CopyError(error, error_size, "cannot register compute session teardown");
        return 2;
      }
      g_atexit_registered = true;
    }
    pvrgpu::stub::ModelComputeStats result;
    const int status = pvrgpu::stub::RunConfiguredCompute(
        &prepared, &result, &diagnostic);
    if (status != 0) {
      CopyError(error, error_size, diagnostic);
      return status;
    }
    for (std::size_t index = 0; index < prepared.resources.size(); ++index) {
      const auto &resource = prepared.resources[index];
      if (resource.writable && !resource.bytes.empty())
        std::memcpy(dispatch->resources[index].bytes,
                     resource.bytes.data(), resource.bytes.size());
    }
    if (stats) {
      *stats = {result.workgroups, result.invocations, result.alu_instructions,
                result.memory_instructions, result.atomic_instructions,
                result.load_instructions,
                result.store_instructions, result.dram_read_bytes,
                result.dram_write_bytes, result.direct_read_bytes,
                result.direct_write_bytes, result.readback_bytes,
                result.pool_allocations, result.pool_releases,
                result.texture_requests, result.texel_fetches};
    }
    return 0;
  } catch (const std::exception &failure) {
    CopyError(error, error_size, failure.what());
  } catch (...) {
    CopyError(error, error_size, "unhandled SystemC compute failure");
  }
  return 2;
}

extern "C" int pvrgpu_systemc_can_execute_pco_binary(std::uint32_t stage,
                                                    const std::uint8_t *binary,
                                                    std::size_t binary_size,
                                                    char *error,
                                                    std::size_t error_size) {
  if (!binary || binary_size == 0) {
    CopyError(error, error_size, "empty PCO binary");
    return 2;
  }
  if (stage > PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION) {
    CopyError(error, error_size, "invalid graphics shader stage");
    return 2;
  }
  try {
    const std::vector<std::uint8_t> bytes(binary, binary + binary_size);
    if (stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_CONTROL ||
        stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION) {
      (void)pvrgpu::stub::DecodeTessellationPcoProgram(
          stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_CONTROL
              ? pvrgpu::stub::ShaderStage::kTessellationControl
              : pvrgpu::stub::ShaderStage::kTessellationEvaluation, bytes);
    } else if (stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_GEOMETRY) {
      (void)pvrgpu::stub::DecodeGeometryPcoProgram(bytes);
    } else {
      (void)pvrgpu::stub::DecodePcoProgram(
          stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_VERTEX
              ? pvrgpu::stub::ShaderStage::kVertex : pvrgpu::stub::ShaderStage::kFragment,
          bytes);
    }
  } catch (const std::exception &failure) {
    CopyError(error, error_size, failure.what());
    return 2;
  } catch (...) {
    CopyError(error, error_size, "PCO binary could not be decoded");
    return 2;
  }
  return 0;
}

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
  // Capture the output policy with the owned command; changing the process
  // environment after submission must not change a deferred job's output.
  if (!SetPngOutputFromEnvironment(&options, &message)) {
    CopyError(error, error_size, message);
    return 2;
  }
  if (!SetMemoryMode(info->memory_mode, &options, &message)) {
    CopyError(error, error_size, message);
    return 2;
  }
  const bool pco_sequence =
      info->command->command &&
      std::string_view(info->command->command) == "draw_pco_sequence";
  try {
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
  } catch (const std::exception &exception) {
    // Payload copies own their storage before replacing the pending submit.
    // Invalid extents and allocation failures must not cross the C ABI.
    CopyError(error, error_size, exception.what());
    return 2;
  }

  /*
   * A submission that has already run is finished business, not a conflict:
   * the readback that ran it consumed it, and what arrives now is the next
   * frame's work.  Only a submission still waiting to run can absorb another
   * command into its ordered sequence.
   */
  if (g_pending_submit.valid && !g_pending_submit.executed) {
    const bool sequence_active =
        !g_pending_submit.options.driver_commands.empty();
    const bool sequence_root =
        IsIdeasPcoSequenceRoot(g_pending_submit.options.driver_command);
    const std::string stderr_path =
        info->stderr_path && info->stderr_path[0] ? info->stderr_path : "";
    const bool compatible_sequence_member =
        (sequence_active || sequence_root) &&
        g_pending_submit.jsonl_path == info->jsonl_path &&
        g_pending_submit.stderr_path == stderr_path &&
        g_pending_submit.options.output_dir == info->outdir &&
        g_pending_submit.options.emit_png == options.emit_png &&
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
      g_pending_submit.submission_generation = info->submission_generation;
      return 0;
    }
  }

  PendingSubmit pending;
  pending.submission_generation = info->submission_generation;
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

  /* The attachments held from the previous flush describe work that this
   * submission replaces, so they stop being readable now rather than when
   * the new flush happens to run. */
  g_last_framebuffer = {};
  g_pending_submit = std::move(pending);
  return 0;
}

extern "C" int pvrgpu_systemc_flush_graphics_stats(
    pvrgpu_systemc_graphics_stats *stats, char *error,
    std::size_t error_size) {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  if (!stats || stats->version != PVRGPU_SYSTEMC_API_VERSION ||
      stats->submission_generation == 0) {
    CopyError(error, error_size, "invalid SystemC graphics statistics request");
    return 2;
  }
  const std::uint64_t requested = stats->submission_generation;
  if (requested != g_last_graphics_generation) {
    if (!g_pending_submit.valid || g_pending_submit.executed ||
        g_pending_submit.submission_generation != requested) {
      CopyError(error, error_size,
                "SystemC graphics statistics submission ownership mismatch");
      return 2;
    }
    std::string message;
    const int status = FlushPendingSubmitLocked(nullptr, &message);
    if (status != 0) {
      CopyError(error, error_size,
                message.empty() ? "SystemC graphics statistics flush failed" : message);
      return status;
    }
  }
  if (requested != g_last_graphics_generation) {
    CopyError(error, error_size, "SystemC graphics statistics were not published");
    return 2;
  }
  stats->physical_submissions = g_last_graphics_stats.physical_submissions;
  stats->primitives_generated = g_last_graphics_stats.primitives_generated;
  stats->ia_primitives = g_last_graphics_stats.ia_primitives;
  stats->gs_primitives = g_last_graphics_stats.gs_primitives;
  stats->gs_invocations = g_last_graphics_stats.gs_invocations;
  stats->stream_output_primitives_written = g_last_graphics_stats.stream_output_primitives_written;
  stats->stream_output_primitives_storage_needed =
      g_last_graphics_stats.stream_output_primitives_storage_needed;
  return 0;
}

extern "C" int pvrgpu_systemc_flush_stream_output(
    pvrgpu_systemc_stream_output_readback *readback, char *error,
    std::size_t error_size) {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  if (!readback || readback->version != PVRGPU_SYSTEMC_API_VERSION) {
    CopyError(error, error_size, "unsupported SystemC stream output readback version");
    return 2;
  }
  readback->data_written = 0;
  if (!readback->submission_generation || !readback->resource_token ||
      !readback->target_token || !readback->bytes || !readback->bytes_size) {
    CopyError(error, error_size, "invalid SystemC stream output readback request");
    return 2;
  }
  if (readback->submission_generation != g_last_graphics_generation) {
    if (!g_pending_submit.valid || g_pending_submit.executed ||
        readback->submission_generation != g_pending_submit.submission_generation) {
      CopyError(error, error_size, "SystemC stream output submission ownership mismatch");
      return 2;
    }
    std::string message;
    const int status = FlushPendingSubmitLocked(nullptr, &message);
    if (status) {
      CopyError(error, error_size, message);
      return status;
    }
  }
  for (const auto &source : g_last_framebuffer.stream_outputs) {
    if (source.resource_token != readback->resource_token ||
        source.target_token != readback->target_token)
      continue;
    if (source.bytes.size() != readback->bytes_size) {
      CopyError(error, error_size, "SystemC stream output readback extent mismatch");
      return 2;
    }
    std::memcpy(readback->bytes, source.bytes.data(), source.bytes.size());
    readback->internal_offset = source.internal_offset;
    readback->data_written = 1;
    CopyError(error, error_size, "");
    return 0;
  }
  CopyError(error, error_size, "SystemC stream output target ownership mismatch");
  return 2;
}

extern "C" int pvrgpu_systemc_flush_shader_image(
    pvrgpu_systemc_shader_image_readback *readback, char *error,
    std::size_t error_size) {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  if (!readback || readback->version != PVRGPU_SYSTEMC_API_VERSION) {
    CopyError(error, error_size, "unsupported SystemC shader image readback version");
    return 2;
  }
  readback->data_written = 0;
  if (!readback->submission_generation || !readback->resource_token ||
      !readback->bytes || !readback->bytes_size) {
    CopyError(error, error_size, "invalid SystemC shader image readback request");
    return 2;
  }
  if (readback->submission_generation != g_last_graphics_generation) {
    if (!g_pending_submit.valid || g_pending_submit.executed ||
        readback->submission_generation != g_pending_submit.submission_generation) {
      CopyError(error, error_size, "SystemC shader image submission ownership mismatch");
      return 2;
    }
    std::string message;
    const int status = FlushPendingSubmitLocked(nullptr, &message);
    if (status) {
      CopyError(error, error_size, message);
      return status;
    }
  }
  for (const auto &source : g_last_framebuffer.shader_images) {
    if (source.resource_token != readback->resource_token) continue;
    if (source.bytes.size() != readback->bytes_size) {
      CopyError(error, error_size, "SystemC shader image readback extent mismatch");
      return 2;
    }
    std::memcpy(readback->bytes, source.bytes.data(), source.bytes.size());
    readback->data_written = 1;
    CopyError(error, error_size, "");
    return 0;
  }
  CopyError(error, error_size, "SystemC shader image resource ownership mismatch");
  return 2;
}

extern "C" int pvrgpu_systemc_flush_readback(
    pvrgpu_systemc_readback_info *readback, char *error,
    std::size_t error_size) {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  if (!readback) {
    CopyError(error, error_size, "missing SystemC API readback info");
    return 2;
  }
  if (readback->version != PVRGPU_SYSTEMC_API_VERSION) {
    CopyError(error, error_size, "unsupported SystemC API readback version");
    return 2;
  }
  readback->pixels_written = 0;
  const std::uint32_t samples = readback->sample_count ? readback->sample_count : 1;
  const std::uint32_t layers = readback->layer_count ? readback->layer_count : 1;
  if (layers > 256) {
    CopyError(error, error_size, "unsupported SystemC API readback layer count");
    return 2;
  }
  if (samples > 16 || (samples & (samples - 1)) != 0) {
    CopyError(error, error_size, "unsupported SystemC API readback sample count");
    return 2;
  }
  if (!readback->pixels || readback->width == 0 || readback->height == 0 ||
      readback->width > 4096 || readback->height > 4096 ||
      readback->bytes_per_pixel == 0 || readback->bytes_per_pixel > 16) {
    CopyError(error, error_size, "missing SystemC API readback destination");
    return 2;
  }
  const std::uint64_t required = static_cast<std::uint64_t>(readback->width) *
                                 readback->height *
                                 readback->bytes_per_pixel * samples * layers;
  if (static_cast<std::uint64_t>(readback->pixels_size) < required) {
    CopyError(error, error_size,
              "SystemC API readback destination is too small");
    return 2;
  }

  /*
   * A submission that has already run still has attachments to hand out: a
   * pass writing several colour targets is read back one target at a time,
   * and the first of those reads is what runs it.  The flush below returns
   * what that run produced, so this asks it rather than returning early --
   * doing that here published attachment zero and nothing else, because the
   * second read never reached the flush at all.  With nothing submitted and
   * nothing cached it still succeeds and claims no pixels, leaving the caller
   * whatever it already has.
   */
  pvrgpu::stub::ModelFramebuffer framebuffer;
  std::string message;
  const int result = FlushPendingSubmitLocked(&framebuffer, &message);
  if (result != 0) {
    CopyError(error, error_size,
              message.empty() ? "SystemC API flush failed" : message);
    return result;
  }
  if (!framebuffer.valid())
    return 0;
  /*
   * A readback of a different surface than the one the model rendered is not
   * something to paper over with a rescale: report no pixels and let the
   * caller keep its own contents.
   */
  if (framebuffer.width != readback->width ||
      framebuffer.height != readback->height ||
      framebuffer.sample_count != samples || framebuffer.layer_count != layers)
    return 0;
  const bool depth_readback = readback->attachment == UINT32_MAX;
  if (depth_readback) {
    if (framebuffer.depth_format == 0 ||
        framebuffer.depth_format != readback->depth_format ||
        pvrgpu::stub::DepthAttachmentBytesPerPixel(framebuffer.depth_format) !=
            readback->bytes_per_pixel)
      return 0;
  } else if (framebuffer.bytes_per_pixel != readback->bytes_per_pixel) {
    return 0;
  }

  /*
   * Attachment zero is the frame's own surface; the rest are the additional
   * colour targets the same pass wrote, in target order.  An attachment the
   * pass did not write publishes nothing rather than the first one's pixels.
   */
  const std::vector<std::uint8_t> *source = nullptr;
  if (depth_readback) {
    source = &framebuffer.depth_pixels;
  } else if (readback->attachment == 0) {
    source = &framebuffer.pixels;
  } else if (readback->attachment - 1 < framebuffer.extra.size()) {
    source = &framebuffer.extra[readback->attachment - 1];
  }
  if (!source || source->size() != static_cast<std::size_t>(required))
    return 0;

  std::memcpy(readback->pixels, source->data(),
              static_cast<std::size_t>(required));
  readback->pixels_written = 1;
  return 0;
}
