// Submitter（工作提交器）module 的實作。
// 產生目前通過 gate 的內建 GLBench fixtures：Fill.Solid 的四頂點
// triangle strip、TriangleSetup family 的 128×128 indexed lattice，以及
// AttributeFetchShader case 1/2/4/8 的精確 64×64 indexed lattice，以及
// VaryingsShader case 1/2/4/8 的 spacing=1/4、4×4 fullscreen indexed lattice。所有
// GLBench attributes 共用同一個 tightly-packed float2 VBO；VertexFetch 依
// binding 將真實 x/y 放入 VTXIN，GLES z/w default 由公開 PCO shader
// lowering 實作。需要 face
// culling 的 case 明確啟用 BACK/CCW；HalfCulled 使用 pinned GLBench srand(0)
// mixed-winding index stream。每個 DrawList 都使用 Mesa 產生的真實公開 PCO
// shader binary；
// 大型 vertex/index/pipeline payload 留在 MemoryPool，output FIFO 只傳
// PipelineTxn handle 與 frame/sequence metadata。
#include "submitter.h"

#include "common/functional_types.h"
#include "common/glbench_triangle_fixture.h"
#include "common/glbench_texture_fixture.h"
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"
#include "pco_sequence_profiles.h"
#include "shader/pco_iss.h"
#include "texture/texture_unit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kBuiltinVertexBufferGpuAddress =
    UINT64_C(0x20000000);
inline constexpr std::uint64_t kBuiltinTexcoordBufferGpuAddress =
    UINT64_C(0x21000000);
inline constexpr std::uint64_t kBuiltinIndexBufferGpuAddress =
    UINT64_C(0x22000000);
inline constexpr std::uint64_t kDriverSequenceVertexAddressStride =
    UINT64_C(0x00100000);

std::uint64_t SequenceExternalTextureAddress(
    std::size_t submission, DriverPcoShaderStage stage,
    std::uint32_t descriptor_set) {
  if (submission >= kDriverPcoMaximumNestedSequenceCommands ||
      descriptor_set >= kPcoMaximumTextureDescriptorSets) {
    throw std::runtime_error(
        "Submitter sequence external texture slot is out of bounds");
  }
  const std::uint64_t stage_index =
      stage == DriverPcoShaderStage::kVertex ? 0U : 1U;
  const std::uint64_t slot =
      (static_cast<std::uint64_t>(submission) * 2U + stage_index) *
          kPcoMaximumTextureDescriptorSets +
      descriptor_set;
  if (slot > (std::numeric_limits<std::uint64_t>::max() -
              kDriverPcoSequenceExternalAddressBase) /
                 kDriverPcoSequenceAttachmentStride) {
    throw std::overflow_error(
        "Submitter sequence external texture address overflow");
  }
  return kDriverPcoSequenceExternalAddressBase +
         slot * kDriverPcoSequenceAttachmentStride;
}

float FloatFromBits(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool DriverTriangleFragmentColorSupported(const DriverCommand &command) {
  static constexpr std::array<std::uint32_t, 4> kOpaqueRed = {
      UINT32_C(0x3f800000), 0U, 0U, UINT32_C(0x3f800000)};
  return command.fragment_color_bits == kOpaqueRed;
}

bool DriverIndexedQuadCommandSupported(const DriverCommand &command) {
  return command.draw_count != 0 && command.index_count == 6 &&
         command.unique_vertices == 4 && command.primitive_count == 2;
}

// Gallium's pipe_prim_type value for PIPE_PRIM_TRIANGLES.  The API command
// carries the producer enum as an integer so unsupported topology cannot be
// silently reinterpreted by the model.
inline constexpr std::uint32_t kPipePrimTriangles = 4;
inline constexpr std::uint32_t kPipePrimTriangleStrip = 5;
inline constexpr std::uint32_t kPipePrimTriangleFan = 6;

bool IsIdeasPcoSequenceCommand(const DriverCommand &command) {
  const std::string_view name(command.test_case);
  return name == "ideas" || name.rfind("ideas.", 0) == 0 ||
         name.find(".ideas.") != std::string_view::npos;
}

bool IdeasDepthStateMatchesOrdinal(const DriverCommand &command,
                                   std::size_t ordinal) {
  const bool depth_enabled =
      ordinal >= kDriverPcoIdeasDepthEnabledFirstCommand &&
      ordinal < kDriverPcoIdeasDepthEnabledEndCommand;
  return command.depth_clear_bits == UINT32_C(0x3f800000) &&
         command.depth_format != 0 &&
         command.depth_enable == (depth_enabled ? 1U : 0U) &&
         command.depth_write == (depth_enabled ? 1U : 0U) &&
         command.depth_func == (depth_enabled ? 3U : 0U);
}

bool IdeasDepthStateIsSupported(const DriverCommand &command) {
  return IdeasDepthStateMatchesOrdinal(command, 0U) ||
         IdeasDepthStateMatchesOrdinal(
             command, kDriverPcoIdeasDepthEnabledFirstCommand);
}

bool DriverPcoStageAbiIsBounded(const DriverPcoStageAbi &abi,
                                bool allow_zero_temps = false) {
  return (allow_zero_temps || abi.temps != 0) &&
         abi.temps <= kPcoTemporaryCount &&
         abi.vertex_inputs <= kPcoVertexInputCount &&
         abi.vertex_outputs <= kPcoVertexOutputCount &&
         abi.coefficients <= kPcoMaximumVaryingCoefficientCount &&
         abi.shareds <= kPcoMaximumSharedCount &&
         abi.push_constant_start <= abi.shareds &&
         abi.push_constant_count <= abi.shareds - abi.push_constant_start &&
         abi.entry_offset == 0;
}

bool DriverIdeasPcoSequenceCommandSupported(const DriverCommand &command) {
  static constexpr std::array<std::uint32_t, 4> kOpaqueBlack = {
      0, 0, 0, UINT32_C(0x3f800000)};
  static constexpr std::array<std::uint32_t, 3> kViewportBits = {
      UINT32_C(0x42200000), UINT32_C(0x41f00000), UINT32_C(0x3f000000)};
  const bool position_layout =
      command.vertex_stride == 4U * sizeof(float);
  const bool two_attribute_layout =
      command.vertex_stride == 8U * sizeof(float);
  const bool layout = position_layout || two_attribute_layout;
  const bool topology =
      (command.primitive_mode == kPipePrimTriangleStrip &&
       (command.vertex_count == 18U || command.vertex_count == 26U)) ||
      (command.primitive_mode == kPipePrimTriangleFan &&
       command.vertex_count == 12U);
  if (!layout || !topology ||
      command.framebuffer_width != 80 || command.framebuffer_height != 60 ||
      command.width != 80 || command.height != 60 ||
      command.format != "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      command.clear_color_bits != kOpaqueBlack || command.first_vertex != 0 ||
      command.instance_count != 1 || command.indexed != 0 ||
      command.vertex_pco.empty() || command.fragment_pco.empty() ||
      command.vertex_pco.size() > kDriverPcoMaximumBinaryBytes ||
      command.fragment_pco.size() > kDriverPcoMaximumBinaryBytes ||
      command.vertex_shared.size() != command.vertex_pco_abi.shareds ||
      command.fragment_shared.size() != command.fragment_pco_abi.shareds ||
      !DriverPcoStageAbiIsBounded(command.vertex_pco_abi) ||
      !DriverPcoStageAbiIsBounded(command.fragment_pco_abi,
                                  position_layout) ||
      command.vertex_pco_abi.vertex_inputs !=
          command.vertex_stride / sizeof(float) ||
      command.vertex_pco_abi.coefficients != 0 ||
      command.vertex_pco_abi.push_constant_start != 0 ||
      command.vertex_pco_abi.push_constant_count !=
          command.vertex_pco_abi.shareds ||
      command.fragment_pco_abi.vertex_inputs != 0 ||
      command.fragment_pco_abi.vertex_outputs != 0 ||
      command.fragment_pco_abi.push_constant_start != 0 ||
      command.fragment_pco_abi.push_constant_count !=
          command.fragment_pco_abi.shareds ||
      command.position_output_start != 0 ||
      command.position_output_count != 4 ||
      command.vertex_pco_abi.vertex_outputs !=
          command.position_output_count + command.varying_output_count ||
      command.varying_output_count >
          kDriverPcoMaximumVaryingComponents ||
      command.fragment_position_start != 0 ||
      command.fragment_pco_abi.coefficients !=
          command.fragment_position_count + command.fragment_varying_count ||
      (position_layout
           ? command.vertex_pco_abi.vertex_outputs != 4 ||
                 command.vertex_pco_abi.shareds != 32 ||
                 command.varying_output_start != 0 ||
                 command.varying_output_count != 0 ||
                 command.fragment_position_count != 0 ||
                 command.fragment_varying_start != 0 ||
                 command.fragment_varying_count != 0 ||
                 command.fragment_pco_abi.coefficients != 0 ||
                 (command.fragment_pco_abi.shareds != 0 &&
                  command.fragment_pco_abi.shareds != 4)
           : command.vertex_pco_abi.vertex_outputs != 14 ||
                 command.vertex_pco_abi.shareds != 44 ||
                 command.fragment_pco_abi.shareds != 12 ||
                 command.varying_output_start != 4 ||
                 command.varying_output_count != 10 ||
                 command.fragment_position_count != 4 ||
                 command.fragment_varying_start != 4 ||
                 command.fragment_varying_count != 40 ||
                 command.fragment_pco_abi.coefficients != 44) ||
      command.viewport_scale_bits != kViewportBits ||
      command.viewport_translate_bits != kViewportBits ||
      command.front_ccw != 0 ||
      (command.cull_face != 0 && command.cull_face != 2) ||
      command.fill_front != 0 || command.fill_back != 0 ||
      command.scissor != 0 || command.rasterizer_discard != 0 ||
      command.multisample != 0 || command.half_pixel_center != 1 ||
      command.bottom_edge_rule != 0 || command.clip_halfz != 0 ||
      command.depth_clip_near != 1 || command.depth_clip_far != 1 ||
      command.depth_clamp != 0 || command.sample_mask != UINT32_MAX ||
      command.color_mask != 0x0f || command.blend_enable != 0 ||
      command.dither != 1 || !IdeasDepthStateIsSupported(command) ||
      command.sampled_texture_count != 0 ||
      !command.sampled_texture_bytes.empty() ||
      command.declared_sampled_texture_bytes_size != 0 ||
      command.sampled_texture_width != 0 ||
      command.sampled_texture_height != 0 ||
      command.sampled_texture_row_pitch != 0 ||
      !command.sampled_texture_format.empty() ||
      command.sampled_texture_mip_count != 0) {
    return false;
  }
  const std::uint64_t expected_bytes =
      static_cast<std::uint64_t>(command.vertex_count) *
      command.vertex_stride;
  if (expected_bytes == 0 ||
      command.raw_vertex_data.size() != expected_bytes ||
      command.declared_raw_vertex_data_size != expected_bytes ||
      command.declared_vertex_pco_size != command.vertex_pco.size() ||
      command.declared_fragment_pco_size != command.fragment_pco.size()) {
    return false;
  }
  for (std::uint64_t vertex = 0; vertex < command.vertex_count; ++vertex) {
    const std::size_t offset =
        static_cast<std::size_t>(vertex * command.vertex_stride);
    for (std::size_t component = 0;
         component < command.vertex_stride / sizeof(float); ++component) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, command.raw_vertex_data.data() + offset +
                             component * sizeof(bits),
                  sizeof(bits));
      if (!std::isfinite(FloatFromBits(bits)))
        return false;
    }
  }
  return true;
}

bool DriverPcoTrianglesCommandSupported(const DriverCommand &command) {
  if (IsIdeasPcoSequenceCommand(command))
    return DriverIdeasPcoSequenceCommandSupported(command);
  static constexpr std::array<std::uint32_t, 4> kOpaqueBlack = {
      0, 0, 0, UINT32_C(0x3f800000)};
  static constexpr std::array<std::uint32_t, 3> kViewportBits = {
      UINT32_C(0x42200000), UINT32_C(0x41f00000), UINT32_C(0x3f000000)};
  const bool conditionals_layout =
      command.vertex_stride == kDriverPcoPositionVertexStride &&
      command.vertex_pco_abi.vertex_inputs == 4;
  const bool lit_mesh_layout =
      command.vertex_stride == kDriverPcoPositionNormalVertexStride &&
      command.vertex_pco_abi.vertex_inputs == 8;
  const bool texture_layout =
      command.vertex_stride ==
          kDriverPcoPositionNormalTexcoordVertexStride &&
      command.vertex_pco_abi.vertex_inputs == 12;
  if (command.framebuffer_width != 80 || command.framebuffer_height != 60 ||
      command.width != 80 || command.height != 60 ||
      command.format != "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      command.clear_color_bits != kOpaqueBlack ||
      (!conditionals_layout && !lit_mesh_layout && !texture_layout) ||
      command.vertex_count == 0 || command.vertex_count % 3U != 0 ||
      command.first_vertex != 0 || command.instance_count != 1 ||
      command.primitive_mode != kPipePrimTriangles || command.indexed != 0 ||
      command.vertex_pco.empty() || command.fragment_pco.empty() ||
      command.vertex_pco.size() > kDriverPcoMaximumBinaryBytes ||
      command.fragment_pco.size() > kDriverPcoMaximumBinaryBytes ||
      command.vertex_shared.size() != command.vertex_pco_abi.shareds ||
      command.fragment_shared.size() != command.fragment_pco_abi.shareds ||
      !DriverPcoStageAbiIsBounded(command.vertex_pco_abi) ||
      !DriverPcoStageAbiIsBounded(command.fragment_pco_abi)) {
    return false;
  }
  const std::uint64_t end_vertex =
      static_cast<std::uint64_t>(command.first_vertex) +
      command.vertex_count;
  const std::uint64_t expected_bytes =
      end_vertex * static_cast<std::uint64_t>(command.vertex_stride);
  if (end_vertex > std::numeric_limits<std::uint32_t>::max() ||
      expected_bytes == 0 ||
      expected_bytes > std::numeric_limits<std::uint32_t>::max() ||
      command.raw_vertex_data.size() != expected_bytes) {
    return false;
  }
  if (command.vertex_pco_abi.coefficients != 0 ||
      command.fragment_pco_abi.vertex_inputs != 0 ||
      command.fragment_pco_abi.vertex_outputs != 0 ||
      command.position_output_start != 0 ||
      command.position_output_count != 4 ||
      command.vertex_pco_abi.vertex_outputs !=
          command.position_output_count + command.varying_output_count ||
      (command.varying_output_count != 0 &&
       command.varying_output_start != command.position_output_count) ||
      command.fragment_position_start != 0 ||
      (command.fragment_varying_count != 0 &&
       command.fragment_varying_start != command.fragment_position_count) ||
      command.fragment_pco_abi.coefficients !=
          command.fragment_position_count + command.fragment_varying_count ||
      (conditionals_layout &&
       (!DriverPcoStageAbiMatches(command.vertex_pco_abi,
                                  kConditionalsVertexPcoAbi) ||
        !DriverPcoStageAbiMatches(command.fragment_pco_abi,
                                  kConditionalsFragmentPcoAbi) ||
        command.varying_output_start != 0 ||
        command.varying_output_count != 0 ||
        command.fragment_position_count != 0 ||
        command.fragment_varying_start != 0 ||
        command.fragment_varying_count != 0 ||
        command.vertex_pco != ConditionalsVertexPcoBinary() ||
        command.fragment_pco != ConditionalsFragmentPcoBinary())) ||
      (lit_mesh_layout &&
       (command.varying_output_start != 4 ||
        command.varying_output_count == 0 ||
        command.varying_output_count > kDriverPcoMaximumVaryingComponents ||
        command.fragment_position_count != 4 ||
        command.fragment_varying_start != 4 ||
        command.fragment_varying_count !=
            command.varying_output_count * 4U)) ||
      (texture_layout &&
       (command.vertex_count != 36 ||
        command.vertex_pco_abi.vertex_outputs != 7 ||
        command.vertex_pco_abi.shareds != 32 ||
        command.vertex_pco_abi.push_constant_start != 0 ||
        command.vertex_pco_abi.push_constant_count != 32 ||
        command.fragment_pco_abi.coefficients != 16 ||
        command.fragment_pco_abi.shareds != 20 ||
        command.fragment_pco_abi.push_constant_start != 0 ||
        command.fragment_pco_abi.push_constant_count != 0 ||
        command.varying_output_start != 4 ||
        command.varying_output_count != 3 ||
        command.fragment_position_count != 4 ||
        command.fragment_varying_start != 4 ||
        command.fragment_varying_count != 12)) ||
      command.viewport_scale_bits != kViewportBits ||
      command.viewport_translate_bits != kViewportBits ||
      command.front_ccw != 0 || command.cull_face != 2 ||
      command.fill_front != 0 || command.fill_back != 0 ||
      command.scissor != 0 || command.rasterizer_discard != 0 ||
      command.multisample != 0 || command.half_pixel_center != 1 ||
      command.bottom_edge_rule != 0 || command.clip_halfz != 0 ||
      command.depth_clip_near != 1 || command.depth_clip_far != 1 ||
      command.depth_clamp != 0 || command.sample_mask != UINT32_MAX ||
      command.color_mask != 0x0f || command.blend_enable != 0 ||
      command.dither != 1 || command.depth_enable != 1 ||
      command.depth_write != 1 || command.depth_func != 3 ||
      command.depth_clear_bits != UINT32_C(0x3f800000) ||
      command.depth_format == 0) {
    return false;
  }
  if (texture_layout) {
    if (command.sampled_texture_count != 1 ||
        command.sampled_texture_bytes.size() != kDriverPcoTextureBytes ||
        command.declared_sampled_texture_bytes_size !=
            command.sampled_texture_bytes.size() ||
        command.sampled_texture_width != kDriverPcoTextureWidth ||
        command.sampled_texture_height != kDriverPcoTextureHeight ||
        command.sampled_texture_row_pitch != kDriverPcoTextureRowPitch ||
        command.sampled_texture_format !=
            "PIPE_FORMAT_R8G8B8X8_UNORM" ||
        command.sampled_texture_mip_count != 1) {
      return false;
    }
  } else if (command.sampled_texture_count != 0 ||
             !command.sampled_texture_bytes.empty() ||
             command.declared_sampled_texture_bytes_size != 0 ||
             command.sampled_texture_width != 0 ||
             command.sampled_texture_height != 0 ||
             command.sampled_texture_row_pitch != 0 ||
             !command.sampled_texture_format.empty() ||
             command.sampled_texture_mip_count != 0) {
    return false;
  }
  for (std::uint64_t vertex = 0; vertex < end_vertex; ++vertex) {
    const std::size_t offset =
        static_cast<std::size_t>(vertex * command.vertex_stride);
    const std::size_t component_count = texture_layout
                                            ? 8U
                                            : lit_mesh_layout ? 6U : 3U;
    for (std::size_t component = 0; component < component_count; ++component) {
      std::uint32_t bits = 0;
      std::memcpy(&bits,
                  command.raw_vertex_data.data() + offset +
                      component * sizeof(bits),
                  sizeof(bits));
      if (!std::isfinite(FloatFromBits(bits)))
        return false;
    }
  }
  return true;
}

std::uint64_t Fnv1a64(const void *data, std::size_t size) {
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (std::size_t offset = 0; offset < size; ++offset) {
    hash ^= bytes[offset];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

template <typename Container>
std::uint64_t Fnv1a64(const Container &values) {
  return Fnv1a64(values.data(), values.size() * sizeof(values.front()));
}

void DebugSequenceResourceHashes(
    const GpuMemorySystem &memory, std::size_t consumer_ordinal,
    const DriverPcoSampledTexture &texture, std::uint64_t gpu_address) {
  const char *enabled = std::getenv("PVRGPU_SEQUENCE_DEBUG_HASHES");
  if (!enabled || std::string_view(enabled) != "1")
    return;
  for (std::size_t level = 0; level < texture.mip_count; ++level) {
    const DriverPcoTextureMipLayout &mip = texture.mip[level];
    const std::size_t bytes =
        static_cast<std::size_t>(mip.row_pitch_bytes) * mip.height;
    const std::vector<std::uint8_t> payload = memory.backing().Read(
        gpu_address + mip.offset_bytes, bytes);
    std::cerr << "sequence-resource-hash consumer=" << consumer_ordinal
              << " producer=" << texture.producer_command_index
              << " source=" << static_cast<unsigned>(texture.source)
              << " set=" << texture.descriptor_set << " mip=" << level
              << " bytes=" << bytes << " fnv1a64=" << std::hex
              << std::setw(16) << std::setfill('0') << Fnv1a64(payload)
              << std::dec << std::setfill(' ') << '\n';
    const char *dump_dir =
        std::getenv("PVRGPU_SEQUENCE_DEBUG_DUMP_DIR");
    if (dump_dir && dump_dir[0]) {
      const std::string path =
          std::string(dump_dir) + "/consumer" +
          std::to_string(consumer_ordinal) + "-producer" +
          std::to_string(texture.producer_command_index) + "-source" +
          std::to_string(static_cast<unsigned>(texture.source)) + "-set" +
          std::to_string(texture.descriptor_set) + "-mip" +
          std::to_string(level) + ".bin";
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      if (!output)
        throw std::runtime_error(
            "Submitter cannot open sequence debug dump");
      output.write(reinterpret_cast<const char *>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
      if (!output)
        throw std::runtime_error(
            "Submitter cannot write sequence debug dump");
    }
  }
}


constexpr std::uint64_t Bits(std::uint64_t value, unsigned first,
                             unsigned last) {
  const unsigned width = last - first + 1U;
  const std::uint64_t mask =
      width == 64U ? ~UINT64_C(0) : ((UINT64_C(1) << width) - 1U);
  return (value & mask) << first;
}

void StoreU64(
    std::array<std::uint32_t, kFillTexNearestSharedDwordCount> &destination,
    std::size_t first_dword, std::uint64_t value) {
  destination.at(first_dword) = static_cast<std::uint32_t>(value);
  destination.at(first_dword + 1U) =
      static_cast<std::uint32_t>(value >> 32U);
}

void PatchPcoDescriptorAddress(std::vector<std::uint32_t> *shared,
                               std::uint32_t descriptor_set,
                               std::uint64_t gpu_address) {
  if (!shared || descriptor_set >= kPcoMaximumTextureDescriptorSets ||
      gpu_address == 0 || (gpu_address & 3U) != 0) {
    throw std::runtime_error(
        "Submitter PCO descriptor relocation arguments are invalid");
  }
  const std::size_t base =
      static_cast<std::size_t>(descriptor_set) *
      kPcoTextureDescriptorDwordCount;
  if (base + 3U >= shared->size())
    throw std::runtime_error(
        "Submitter PCO descriptor relocation is out of bounds");
  std::uint64_t word1 = (*shared)[base + 2U] |
                        (static_cast<std::uint64_t>((*shared)[base + 3U])
                         << 32U);
  constexpr std::uint64_t kAddressMask =
      ((UINT64_C(1) << 38U) - 1U) << 16U;
  const std::uint64_t encoded_address = gpu_address >> 2U;
  if ((word1 & kAddressMask) != 0 ||
      encoded_address >= (UINT64_C(1) << 38U)) {
    throw std::runtime_error(
        "Submitter PCO descriptor address is not canonical/encodable");
  }
  word1 |= encoded_address << 16U;
  (*shared)[base + 2U] = static_cast<std::uint32_t>(word1);
  (*shared)[base + 3U] = static_cast<std::uint32_t>(word1 >> 32U);
}

bool DriverTexturedTrianglesCommandSupported(const DriverCommand &command) {
  if (command.framebuffer_width == 0 || command.framebuffer_height == 0 ||
      command.width == 0 || command.height == 0 ||
      command.width > command.framebuffer_width ||
      command.height > command.framebuffer_height ||
      command.texture_width == 0 || command.texture_width > 16384U ||
      command.texture_height == 0 || command.texture_height > 16384U ||
      command.texture_rgba8_path.empty()) {
    return false;
  }
  for (const auto &vertex : command.vertex_bits) {
    if (!std::isfinite(FloatFromBits(vertex[0])) ||
        !std::isfinite(FloatFromBits(vertex[1]))) {
      return false;
    }
  }
  for (const auto &texcoord : command.texcoord_bits) {
    if (!std::isfinite(FloatFromBits(texcoord[0])) ||
        !std::isfinite(FloatFromBits(texcoord[1]))) {
      return false;
    }
  }
  return true;
}

std::vector<float> DriverTriangleFloat2Vertices(
    const DriverCommand &command) {
  std::vector<float> vertices;
  vertices.reserve(6);
  for (std::size_t index = 0; index < 3; ++index) {
    const auto &vertex = command.vertex_bits[index];
    vertices.push_back(FloatFromBits(vertex[0]));
    vertices.push_back(FloatFromBits(vertex[1]));
  }
  return vertices;
}

template <std::size_t N>
std::vector<float> Float2ValuesFromBits(
    const std::array<std::array<std::uint32_t, 2>, N> &bits) {
  std::vector<float> values;
  values.reserve(N * 2U);
  for (const auto &value : bits) {
    values.push_back(FloatFromBits(value[0]));
    values.push_back(FloatFromBits(value[1]));
  }
  return values;
}

std::vector<std::uint8_t> LoadExactRgba8Sidecar(
    const DriverCommand &command) {
  const std::uint64_t expected_bytes =
      static_cast<std::uint64_t>(command.texture_width) *
      command.texture_height * 4U;
  if (expected_bytes == 0 ||
      expected_bytes > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "Submitter driver texture byte size is unsupported");
  }
  if (!command.texture_rgba8_bytes.empty()) {
    if (command.texture_rgba8_bytes.size() != expected_bytes) {
      throw std::runtime_error(
          "Submitter copied driver RGBA8 texture size is invalid");
    }
    return command.texture_rgba8_bytes;
  }
  std::ifstream input(command.texture_rgba8_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Submitter cannot open driver RGBA8 texture: " +
                             command.texture_rgba8_path);
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(expected_bytes));
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error(
        "Submitter driver RGBA8 texture sidecar is truncated");
  }
  char extra = 0;
  if (input.read(&extra, 1) || input.gcount() != 0) {
    throw std::runtime_error(
        "Submitter driver RGBA8 texture sidecar has extra bytes");
  }
  return bytes;
}

GlbenchFillTextureFixture MakeDriverTexturedTrianglesFixture(
    const DriverCommand &command) {
  if (!DriverTexturedTrianglesCommandSupported(command)) {
    throw std::runtime_error(
        "Submitter driver textured-triangle fields are unsupported");
  }
  GlbenchFillTextureFixture fixture;
  fixture.positions = Float2ValuesFromBits(command.vertex_bits);
  fixture.texture_coordinates = Float2ValuesFromBits(command.texcoord_bits);
  fixture.texture_bytes = LoadExactRgba8Sidecar(command);
  fixture.vertex_scale_bits = UINT32_C(0x3f800000);

  fixture.resource.gpu_address = kGlbenchTextureGpuAddress;
  fixture.resource.byte_size =
      static_cast<std::uint32_t>(fixture.texture_bytes.size());
  fixture.resource.mip_count = 1;
  fixture.resource.format = TextureFormat::kRgba8Unorm;
  fixture.resource.layout = TextureLayout::kLinear;
  fixture.resource.mip[0].width = command.texture_width;
  fixture.resource.mip[0].height = command.texture_height;
  fixture.resource.mip[0].row_pitch_bytes = command.texture_width * 4U;
  fixture.resource.mip[0].offset_bytes = 0;

  const std::uint64_t image_word0 =
      Bits(4U, 0, 2) | Bits(3U, 5, 7) | Bits(2U, 8, 10) |
      Bits(1U, 11, 13) | Bits(0U, 14, 16) | Bits(12U, 27, 33) |
      Bits(command.texture_width - 1U, 34, 47) |
      Bits(command.texture_height - 1U, 48, 61);
  StoreU64(fixture.fragment_shared, 0, image_word0);
  const std::uint64_t image_word1 =
      Bits(command.texture_width - 1U, 0, 14) |
      Bits(kGlbenchTextureGpuAddress >> 2U, 16, 53) | Bits(1U, 60, 63);
  StoreU64(fixture.fragment_shared, 2, image_word1);
  fixture.fragment_shared[4] = fixture.resource.byte_size;

  const std::uint64_t nearest_clamp_to_edge_sampler =
      Bits(4095U, 0, 12) | Bits(2U, 33, 35) | Bits(2U, 41, 43);
  StoreU64(fixture.fragment_shared, 8, nearest_clamp_to_edge_sampler);
  StoreU64(fixture.fragment_shared, 10, 0U);
  StoreU64(fixture.fragment_shared, 16, nearest_clamp_to_edge_sampler);
  StoreU64(fixture.fragment_shared, 18, 0U);
  fixture.sampler.min_filter = TextureFilter::kNearest;
  fixture.sampler.mag_filter = TextureFilter::kNearest;
  fixture.sampler.mip_filter = TextureFilter::kNearest;
  fixture.sampler.wrap_u = TextureWrapMode::kClampToEdge;
  fixture.sampler.wrap_v = TextureWrapMode::kClampToEdge;
  fixture.sampler.normalized_coordinates = 1;
  fixture.sampler.base_mip_level = 0;
  return fixture;
}

VertexBufferResource StoreFloat2VertexBuffer(MemoryPool &pool,
                                             const std::vector<float> &values,
                                             std::uint64_t gpu_address,
                                             GpuMemorySystem *memory) {
  if (values.empty() || values.size() % 2 != 0 ||
      values.size() > std::numeric_limits<std::uint32_t>::max() /
                          sizeof(float)) {
    throw std::runtime_error("Submitter float2 VBO size is invalid");
  }
  VertexBufferResource resource;
  if (memory)
    HostWriteArray(*memory, gpu_address, values);
  else
    resource.data = StoreNewArray(pool, values);
  resource.gpu_address = gpu_address;
  resource.byte_size =
      static_cast<std::uint32_t>(values.size() * sizeof(float));
  return resource;
}

VertexBufferResource StoreRawVertexBuffer(
    MemoryPool &pool, const std::vector<std::uint8_t> &bytes,
    std::uint64_t gpu_address, GpuMemorySystem *memory) {
  if (bytes.empty() ||
      bytes.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("Submitter raw VBO size is invalid");
  }
  VertexBufferResource resource;
  if (memory)
    HostWriteArray(*memory, gpu_address, bytes);
  else
    resource.data = StoreNewArray(pool, bytes);
  resource.gpu_address = gpu_address;
  resource.byte_size = static_cast<std::uint32_t>(bytes.size());
  return resource;
}

DriverPcoTopologyExpansion ExpandDriverPcoTopologyImpl(
    const DriverCommand &command) {
  DriverPcoTopologyExpansion result;
  if (command.primitive_mode == kPipePrimTriangles) {
    result.vertices = command.raw_vertex_data;
    result.input_primitives = command.vertex_count / 3U;
    result.emitted_primitives = result.input_primitives;
    return result;
  }
  if (command.primitive_mode != kPipePrimTriangleStrip &&
      command.primitive_mode != kPipePrimTriangleFan) {
    throw std::runtime_error(
        "Submitter cannot expand the driver PCO primitive topology");
  }
  if (command.vertex_count < 3 || command.vertex_stride == 0)
    throw std::runtime_error("Submitter driver PCO topology is empty");
  const std::uint64_t triangle_count = command.vertex_count - 2U;
  const std::uint64_t expanded_bytes =
      triangle_count * 3U * command.vertex_stride;
  if (expanded_bytes > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("Submitter expanded PCO VBO size overflow");

  result.input_primitives = static_cast<std::uint32_t>(triangle_count);
  result.vertices.reserve(static_cast<std::size_t>(expanded_bytes));
  const auto append_vertex = [&](std::uint32_t vertex) {
    const std::size_t begin =
        static_cast<std::size_t>(vertex) * command.vertex_stride;
    const std::size_t end = begin + command.vertex_stride;
    if (end > command.raw_vertex_data.size())
      throw std::runtime_error("Submitter PCO topology index is out of range");
    result.vertices.insert(result.vertices.end(),
                           command.raw_vertex_data.begin() + begin,
                           command.raw_vertex_data.begin() + end);
  };
  const auto same_position = [&](std::uint32_t lhs, std::uint32_t rhs) {
    // Duplicate-position accounting is part of the locked Ideas counter
    // contract only. Other generic strips (notably Shadow's float2 mask quad)
    // are expanded without inferring an attribute layout from stride.
    if (!IsIdeasPcoSequenceCommand(command))
      return false;
    const std::size_t lhs_begin =
        static_cast<std::size_t>(lhs) * command.vertex_stride;
    const std::size_t rhs_begin =
        static_cast<std::size_t>(rhs) * command.vertex_stride;
    constexpr std::size_t position_components = 3U;
    if (lhs_begin + position_components * sizeof(float) >
            command.raw_vertex_data.size() ||
        rhs_begin + position_components * sizeof(float) >
            command.raw_vertex_data.size()) {
      throw std::runtime_error(
          "Submitter PCO topology position is out of range");
    }
    for (std::size_t component = 0; component < position_components;
         ++component) {
      std::uint32_t lhs_bits = 0;
      std::uint32_t rhs_bits = 0;
      std::memcpy(&lhs_bits,
                  command.raw_vertex_data.data() + lhs_begin +
                      component * sizeof(lhs_bits),
                  sizeof(lhs_bits));
      std::memcpy(&rhs_bits,
                  command.raw_vertex_data.data() + rhs_begin +
                      component * sizeof(rhs_bits),
                  sizeof(rhs_bits));
      if (FloatFromBits(lhs_bits) != FloatFromBits(rhs_bits))
        return false;
    }
    return true;
  };
  const auto append_triangle = [&](std::uint32_t first,
                                   std::uint32_t second,
                                   std::uint32_t third) {
    if (same_position(first, second) || same_position(first, third) ||
        same_position(second, third)) {
      ++result.duplicate_position_primitives;
    }
    append_vertex(first);
    append_vertex(second);
    append_vertex(third);
    ++result.emitted_primitives;
  };
  for (std::uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
    if (command.primitive_mode == kPipePrimTriangleFan) {
      append_triangle(0, triangle + 1U, triangle + 2U);
    } else if ((triangle & 1U) == 0) {
      append_triangle(triangle, triangle + 1U, triangle + 2U);
    } else {
      append_triangle(triangle + 1U, triangle, triangle + 2U);
    }
  }
  if (result.emitted_primitives != result.input_primitives ||
      result.vertices.size() !=
          static_cast<std::uint64_t>(result.emitted_primitives) * 3U *
              command.vertex_stride) {
    throw std::runtime_error(
        "Submitter PCO topology expansion accounting is inconsistent");
  }
  return result;
}

template <typename T>
void StoreIndexBuffer(MemoryPool &pool, GpuMemorySystem *memory,
                      const std::vector<T> &indices, PipelineState *state) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (!state || indices.empty() ||
      indices.size() > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::runtime_error("Submitter index buffer size is invalid");
  }
  if (memory) {
    HostWriteArray(*memory, kBuiltinIndexBufferGpuAddress, indices);
    state->index_buffer_gpu_address = kBuiltinIndexBufferGpuAddress;
    state->index_buffer_bytes = indices.size() * sizeof(T);
  } else {
    state->vertex_indices = StoreNewArray(pool, indices);
  }
}

VertexAttributeBinding MakeFloat2Binding(std::uint32_t buffer_index,
                                         std::uint16_t destination_register,
                                         std::uint8_t destination_components) {
  VertexAttributeBinding binding;
  binding.buffer_index = buffer_index;
  binding.offset_bytes = 0;
  binding.stride_bytes = 2U * sizeof(float);
  binding.destination_register = destination_register;
  binding.component_type = VertexComponentType::kFloat32;
  binding.source_components = 2;
  binding.destination_components = destination_components;
  binding.normalized = 0;
  binding.integer = 0;
  binding.instance_divisor = 0;
  return binding;
}

VertexAttributeBinding MakeDriverPcoFloat3Binding(
    std::uint32_t offset_bytes, std::uint32_t stride_bytes,
    std::uint16_t destination_register) {
  VertexAttributeBinding binding;
  binding.buffer_index = 0;
  binding.offset_bytes = offset_bytes;
  binding.stride_bytes = stride_bytes;
  binding.destination_register = destination_register;
  binding.component_type = VertexComponentType::kFloat32;
  binding.source_components = 3;
  binding.destination_components = 4;
  binding.normalized = 0;
  binding.integer = 0;
  binding.instance_divisor = 0;
  return binding;
}

VertexAttributeBinding MakeDriverPcoFloat2Binding(
    std::uint32_t offset_bytes, std::uint32_t stride_bytes,
    std::uint16_t destination_register) {
  VertexAttributeBinding binding;
  binding.buffer_index = 0;
  binding.offset_bytes = offset_bytes;
  binding.stride_bytes = stride_bytes;
  binding.destination_register = destination_register;
  binding.component_type = VertexComponentType::kFloat32;
  binding.source_components = 2;
  binding.destination_components = 4;
  binding.normalized = 0;
  binding.integer = 0;
  binding.instance_divisor = 0;
  return binding;
}

} // namespace

DriverPcoTopologyExpansion
ExpandDriverPcoTopology(const DriverCommand &command) {
  return ExpandDriverPcoTopologyImpl(command);
}

Submitter::Submitter(sc_core::sc_module_name name, MemoryPool &pool,
                     const Options &options, GpuMemorySystem *memory,
                     sc_core::sc_event *sequence_completion)
    : sc_module(name), pool_(pool), options_(options), memory_(memory),
      sequence_completion_(sequence_completion) {
  SC_THREAD(Run);
}

void Submitter::Run() {
  const bool driver_command = options_.driver_command.enabled;
  const bool driver_clear_command =
      driver_command && options_.driver_command.command == "clear_color";
  const bool driver_triangle_command =
      driver_command && options_.driver_command.command == "draw_triangle";
  const bool driver_indexed_quad_command =
      driver_command && options_.driver_command.command == "draw_indexed_quad";
  const bool driver_textured_triangles_command =
      driver_command &&
      options_.driver_command.command == "draw_textured_triangles";
  const bool driver_pco_triangles_command =
      driver_command &&
      (options_.driver_command.command == "draw_pco_triangles" ||
       options_.driver_command.command == "draw_pco_sequence");
  const bool driver_pco_sequence_command =
      driver_command &&
      options_.driver_command.command == "draw_pco_sequence";
  const bool driver_primitive_sequence_command =
      driver_command &&
      options_.driver_command.command == "draw_primitive_sequence";
  const FunctionalCase functional_case =
      FunctionalCaseFromName(options_.test_case);
  if (!IsRasterFunctionalCase(functional_case))
    throw std::runtime_error("Submitter received an unsupported GLBench case");
  const std::uint32_t command_framebuffer_width =
      (driver_indexed_quad_command || driver_textured_triangles_command ||
       driver_pco_triangles_command)
          ? options_.driver_command.framebuffer_width
          : options_.driver_command.width;
  const std::uint32_t command_framebuffer_height =
      (driver_indexed_quad_command || driver_textured_triangles_command ||
       driver_pco_triangles_command)
          ? options_.driver_command.framebuffer_height
          : options_.driver_command.height;
  if (driver_command &&
      (options_.frames != 1 ||
       options_.width != command_framebuffer_width ||
       options_.height != command_framebuffer_height ||
       (driver_clear_command &&
        functional_case != FunctionalCase::kDriverClearColor) ||
       (driver_triangle_command &&
        functional_case != FunctionalCase::kDriverTriangleSolid) ||
       (driver_indexed_quad_command &&
        functional_case != FunctionalCase::kDriverIndexedQuad) ||
       (driver_textured_triangles_command &&
        functional_case != FunctionalCase::kDriverTexturedTriangles) ||
       (driver_pco_triangles_command &&
        functional_case != FunctionalCase::kDriverPcoTriangles) ||
       (!driver_clear_command && !driver_triangle_command &&
        !driver_indexed_quad_command &&
        !driver_textured_triangles_command &&
        !driver_pco_triangles_command &&
        !driver_primitive_sequence_command))) {
    throw std::runtime_error(
        "Submitter driver command options do not match the one-frame command");
  }
  std::string sequence_profile_error;
  const bool native_sequence_profile_supported =
      !driver_pco_sequence_command ||
      DriverPcoSequenceSupported(options_, &sequence_profile_error);
  if (!options_.driver_commands.empty() &&
      (!driver_pco_triangles_command ||
       options_.driver_commands.size() >
           kDriverPcoMaximumSequenceCommands ||
       sequence_completion_ == nullptr ||
       (driver_pco_sequence_command
            ? !native_sequence_profile_supported
            : (!IsIdeasPcoSequenceCommand(options_.driver_command) ||
               options_.driver_commands.size() !=
                   kDriverPcoIdeasSequenceCommands ||
               options_.driver_command.draw_count !=
                   options_.driver_commands.size())))) {
    throw std::runtime_error(
        "Submitter ordered PCO sequence options are invalid" +
        (sequence_profile_error.empty()
             ? std::string{}
             : std::string(": ") + sequence_profile_error));
  }
  if (driver_triangle_command &&
      !DriverTriangleFragmentColorSupported(options_.driver_command)) {
    throw std::runtime_error(
        "Submitter driver triangle currently supports only opaque red fragments");
  }
  if (driver_indexed_quad_command &&
      !DriverIndexedQuadCommandSupported(options_.driver_command)) {
    throw std::runtime_error(
        "Submitter driver indexed quad command fields are unsupported");
  }
  if (driver_textured_triangles_command &&
      !DriverTexturedTrianglesCommandSupported(options_.driver_command)) {
    throw std::runtime_error(
        "Submitter driver textured-triangle command fields are unsupported");
  }
  if (driver_pco_triangles_command) {
    if (options_.driver_commands.empty()) {
      if (!DriverPcoTrianglesCommandSupported(options_.driver_command)) {
        throw std::runtime_error(
            "Submitter driver PCO-triangle command fields are unsupported");
      }
    } else if (!driver_pco_sequence_command) {
      for (std::size_t ordinal = 0;
           ordinal < options_.driver_commands.size(); ++ordinal) {
        const DriverCommand &command = options_.driver_commands[ordinal];
        if (!DriverPcoTrianglesCommandSupported(command) ||
            !IdeasDepthStateMatchesOrdinal(command, ordinal)) {
          throw std::runtime_error(
              "Submitter ordered PCO sequence member is unsupported");
        }
      }
    }
  }
  if (!options_.driver_commands.empty() &&
      !driver_pco_sequence_command) {
    std::uint64_t source_vertices = 0;
    std::uint64_t input_primitives = 0;
    std::uint64_t emitted_primitives = 0;
    std::uint64_t duplicate_position_primitives = 0;
    for (const DriverCommand &command : options_.driver_commands) {
      const DriverPcoTopologyExpansion expansion =
          ExpandDriverPcoTopology(command);
      source_vertices += command.vertex_count;
      input_primitives += expansion.input_primitives;
      emitted_primitives += expansion.emitted_primitives;
      duplicate_position_primitives +=
          expansion.duplicate_position_primitives;
    }
    const DriverCommand &logical = options_.driver_command;
    if (source_vertices != logical.ia_vertices ||
        source_vertices != logical.vs_invocations ||
        input_primitives != logical.ia_primitives ||
        input_primitives != logical.clip_invocations ||
        emitted_primitives != input_primitives ||
        logical.clip_primitives > input_primitives ||
        logical.setup_triangles != logical.clip_primitives) {
      throw std::runtime_error(
          "Submitter Ideas topology expansion disagrees with captured "
          "sequence counters: vertices=" + std::to_string(source_vertices) +
          " input=" + std::to_string(input_primitives) +
          " emitted=" + std::to_string(emitted_primitives) +
          " duplicate_position=" +
          std::to_string(duplicate_position_primitives) +
          " expected=" + std::to_string(logical.ia_vertices) + ',' +
          std::to_string(logical.ia_primitives) + ',' +
          std::to_string(logical.clip_primitives));
    }
  }

  const std::size_t submission_count = options_.driver_commands.empty()
                                           ? options_.frames
                                           : options_.driver_commands.size();
  std::vector<std::uint64_t> sequence_color_addresses(submission_count, 0);
  std::vector<std::uint64_t> sequence_depth_addresses(submission_count, 0);
  if (driver_pco_sequence_command) {
    for (std::size_t ordinal = 0; ordinal < submission_count; ++ordinal) {
      const DriverCommand &nested = options_.driver_commands[ordinal];
      const auto color_source = nested.color_attachment_source_command_index;
      const auto depth_source = nested.depth_attachment_source_command_index;
      if (color_source == kDriverPcoNewAttachment) {
        sequence_color_addresses[ordinal] =
            kDriverPcoSequenceColorAddressBase +
            ordinal * kDriverPcoSequenceAttachmentStride;
      } else if (color_source < ordinal) {
        sequence_color_addresses[ordinal] =
            sequence_color_addresses[color_source];
      } else {
        throw std::runtime_error(
            "Submitter sequence color attachment dependency is invalid");
      }
      if (nested.depth_format == 0) {
        if (depth_source != kDriverPcoNewAttachment ||
            nested.depth_enable != 0 || nested.depth_write != 0) {
          throw std::runtime_error(
              "Submitter absent depth attachment state is invalid");
        }
      } else if (depth_source == kDriverPcoNewAttachment) {
        sequence_depth_addresses[ordinal] =
            kDriverPcoSequenceDepthAddressBase +
            ordinal * kDriverPcoSequenceAttachmentStride;
      } else if (depth_source < ordinal &&
                 sequence_depth_addresses[depth_source] != 0) {
        sequence_depth_addresses[ordinal] =
            sequence_depth_addresses[depth_source];
      } else {
        throw std::runtime_error(
            "Submitter sequence depth attachment dependency is invalid");
      }
    }
  }
  for (std::size_t submission = 0; submission < submission_count;
       ++submission) {
    const unsigned frame = static_cast<unsigned>(submission + 1U);
    const DriverCommand &command = options_.driver_commands.empty()
                                       ? options_.driver_command
                                       : options_.driver_commands[submission];
    MemoryAccessStats sequence_dependency_stats;
    if (driver_pco_sequence_command) {
      if (!memory_)
        throw std::runtime_error(
            "Submitter PCO sequence requires unified GPU memory");
      for (const DriverPcoSampledTexture &texture :
           command.sampled_textures) {
        if (texture.source ==
            DriverPcoTextureSource::kPreviousColorAttachment) {
          const std::uint64_t address = sequence_color_addresses.at(
              texture.producer_command_index);
          sequence_dependency_stats +=
              MaterializeSequenceColorMipChain(*memory_, texture, address);
          DebugSequenceResourceHashes(*memory_, submission, texture, address);
        } else if (texture.source ==
                   DriverPcoTextureSource::kPreviousDepthAttachment) {
          const std::uint64_t address = sequence_depth_addresses.at(
              texture.producer_command_index);
          if (!memory_->backing().Contains(
                  address,
                  static_cast<std::size_t>(texture.declared_bytes_size))) {
            throw std::runtime_error(
                "Submitter previous depth attachment is absent from DRAM");
          }
          DebugSequenceResourceHashes(*memory_, submission, texture, address);
        }
      }
    }
    PipelineState state;
    state.width = driver_pco_sequence_command
                      ? command.framebuffer_width
                      : options_.width;
    state.height = driver_pco_sequence_command
                       ? command.framebuffer_height
                       : options_.height;
    state.workload_class = WorkloadClass(options_.test_case);
    state.sequence = frame;
    state.functional_case = functional_case;
    state.stage = PipelineStage::kSubmitted;
    state.memory_mode = options_.memory_mode;
    state.cache_bypass = options_.cache_bypass ? 1U : 0U;
    if (driver_pco_sequence_command) {
      state.framebuffer_gpu_address = sequence_color_addresses[submission];
      const std::uint64_t color_bytes =
          static_cast<std::uint64_t>(state.width) * state.height * 4U;
      if (command.color_attachment_source_command_index !=
          kDriverPcoNewAttachment) {
        if (color_bytes == 0 ||
            color_bytes > std::numeric_limits<std::size_t>::max() ||
            !memory_->backing().Contains(
                state.framebuffer_gpu_address,
                static_cast<std::size_t>(color_bytes))) {
          throw std::runtime_error(
              "Submitter aliased color attachment is absent from DRAM");
        }
        MemoryReadResult color_load = memory_->Readback(
            state.framebuffer_gpu_address,
            static_cast<std::size_t>(color_bytes),
            MemoryClient::kFramebufferReadback);
        if (color_load.data.size() != color_bytes)
          throw std::runtime_error(
              "Submitter aliased color attachment readback is truncated");
        sequence_dependency_stats += color_load.stats;
        state.color_attachment_load =
            StoreNewArray(pool_, color_load.data);
        state.color_attachment_load_enable = 1;
        state.color_attachment_load_bytes = color_bytes;
      }
      bool depth_is_consumed = false;
      for (std::size_t future = submission + 1U;
           future < options_.driver_commands.size(); ++future) {
        const DriverCommand &future_command =
            options_.driver_commands[future];
        depth_is_consumed =
            depth_is_consumed ||
            (future_command.depth_format != 0 &&
             future_command.depth_attachment_source_command_index !=
                 kDriverPcoNewAttachment &&
             sequence_depth_addresses[future] ==
                 sequence_depth_addresses[submission]);
        for (const DriverPcoSampledTexture &texture :
             future_command.sampled_textures) {
          depth_is_consumed =
              depth_is_consumed ||
              (texture.source ==
                   DriverPcoTextureSource::kPreviousDepthAttachment &&
               sequence_depth_addresses.at(texture.producer_command_index) ==
                   sequence_depth_addresses[submission]);
        }
      }
      if (command.depth_format != 0) {
        state.depth_attachment_format = command.depth_format;
        state.depth_attachment_gpu_address =
            sequence_depth_addresses[submission];
        const std::size_t depth_bytes_per_pixel =
            DepthAttachmentBytesPerPixel(command.depth_format);
        const std::uint64_t depth_bytes =
            static_cast<std::uint64_t>(state.width) * state.height *
            depth_bytes_per_pixel;
        if (depth_bytes == 0 ||
            depth_bytes > kDriverPcoSequenceAttachmentStride ||
            depth_bytes > std::numeric_limits<std::size_t>::max()) {
          throw std::runtime_error(
              "Submitter depth attachment byte size is invalid");
        }
        if (command.depth_attachment_source_command_index !=
            kDriverPcoNewAttachment) {
          if (!memory_->backing().Contains(
                  state.depth_attachment_gpu_address,
                  static_cast<std::size_t>(depth_bytes))) {
            throw std::runtime_error(
                "Submitter aliased depth attachment is absent from DRAM");
          }
          MemoryReadResult depth_load = memory_->Readback(
              state.depth_attachment_gpu_address,
              static_cast<std::size_t>(depth_bytes),
              MemoryClient::kFramebufferReadback);
          if (depth_load.data.size() != depth_bytes)
            throw std::runtime_error(
                "Submitter aliased depth attachment readback is truncated");
          sequence_dependency_stats += depth_load.stats;
          state.depth_attachment_load =
              StoreNewArray(pool_, depth_load.data);
          state.depth_attachment_load_enable = 1;
          state.depth_attachment_load_bytes = depth_bytes;
        }
      }
      if (depth_is_consumed && command.depth_format != 0) {
        state.capture_depth_attachment = 1;
      }
      ApplyMemoryAccessStats(state.counters, sequence_dependency_stats);
      const std::uint64_t dependency_cycles =
          MemoryAccessDelayCycles(sequence_dependency_stats);
      state.counters.renderer_cycles += dependency_cycles;
      WaitForCycles(dependency_cycles);
    }
    if (driver_pco_triangles_command) {
      state.vertex_pco_abi = command.vertex_pco_abi;
      state.fragment_pco_abi = command.fragment_pco_abi;
      state.position_output_start =
          command.position_output_start;
      state.position_output_count =
          command.position_output_count;
      state.fragment_position_start =
          command.fragment_position_start;
      state.fragment_position_count =
          command.fragment_position_count;
      state.varying_output_start =
          command.varying_output_start;
      state.varying_output_count =
          command.varying_output_count;
      state.fragment_varying_start =
          command.fragment_varying_start;
      state.fragment_varying_count =
          command.fragment_varying_count;
      state.vertex_sampled_texture_count =
          command.vertex_sampled_texture_count;
      state.sampled_texture_count =
          command.fragment_sampled_texture_count != 0 ||
                  !command.sampled_textures.empty()
              ? command.fragment_sampled_texture_count
              : command.sampled_texture_count;
    }
    state.raster_state.sample_count = 1;
    const bool triangle_setup = IsTriangleSetupFamily(functional_case);
    const bool attribute_fetch = IsAttributeFetchFamily(functional_case);
    const bool varyings = IsVaryingsFamily(functional_case);
    const bool texture_case = IsTextureFamily(functional_case);
    const bool shader_varyings = UsesShaderVaryings(state);
    const std::uint32_t varying_count = VaryingVectorCount(state);
    const std::uint32_t attribute_count =
        functional_case == FunctionalCase::kAttributeFetchShaderEightAttribute
            ? 8U
            : functional_case ==
                  FunctionalCase::kAttributeFetchShaderFourAttribute
                  ? 4U
                  : functional_case ==
                        FunctionalCase::kAttributeFetchShaderTwoAttribute
                        ? 2U
                        : 1U;
    const bool indexed_triangle =
        IsIndexedTriangleRasterCase(functional_case);
    const bool driver_clear = driver_clear_command;
    const bool driver_triangle = driver_triangle_command;
    const bool driver_indexed_quad = driver_indexed_quad_command;
    const bool driver_textured_triangles =
        driver_textured_triangles_command;
    const bool driver_pco_triangles = driver_pco_triangles_command;
    const bool driver_pco_texture =
        driver_pco_triangles && UsesTextureSampling(state);
    const bool driver_primitive_sequence = driver_primitive_sequence_command;
    const bool driver_counter_only_primitive_sequence =
        driver_primitive_sequence &&
        functional_case == FunctionalCase::kDriverClearColor;
    const bool driver_clear_like =
        driver_clear || driver_counter_only_primitive_sequence;
    const bool depth_case =
        driver_clear_like || driver_textured_triangles ||
        functional_case == FunctionalCase::kFillSolidDepthNotEqual ||
        functional_case == FunctionalCase::kFillSolidDepthNever;
    state.raster_state.depth.test_enable = depth_case ? 1U : 0U;
    state.raster_state.depth.write_enable = depth_case ? 1U : 0U;
    state.raster_state.depth.compare_op =
        (driver_clear_like ||
         functional_case == FunctionalCase::kFillSolidDepthNever)
            ? DepthCompareOp::kNever
            : driver_textured_triangles ? DepthCompareOp::kLessOrEqual
                                        : DepthCompareOp::kNotEqual;
    if (functional_case == FunctionalCase::kFillSolidBlended) {
      state.raster_state.blend.enable = 1;
      state.raster_state.blend.rgb_equation = BlendEquation::kAdd;
      state.raster_state.blend.alpha_equation = BlendEquation::kAdd;
      state.raster_state.blend.source_rgb_factor = BlendFactor::kSourceAlpha;
      state.raster_state.blend.destination_rgb_factor =
          BlendFactor::kOneMinusSourceAlpha;
      state.raster_state.blend.source_alpha_factor =
          BlendFactor::kSourceAlpha;
      state.raster_state.blend.destination_alpha_factor =
          BlendFactor::kOneMinusSourceAlpha;
    }
    if (indexed_triangle) {
      state.raster_state.clear_color[0] = 0.0F;
      state.raster_state.clear_color[1] = 1.0F;
      state.raster_state.clear_color[2] = 0.0F;
      state.raster_state.clear_color[3] = 1.0F;
    }
    if (RequiresBackCcwFaceCull(functional_case)) {
      state.raster_state.face_cull.enable = 1;
      state.raster_state.face_cull.mode = CullFaceMode::kBack;
      state.raster_state.face_cull.front_face =
          FrontFaceWinding::kCounterClockwise;
    }
    if (driver_textured_triangles) {
      state.raster_state.face_cull.enable = 1;
      state.raster_state.face_cull.mode = CullFaceMode::kBack;
      state.raster_state.face_cull.front_face = FrontFaceWinding::kClockwise;
    }
    if (driver_pco_triangles) {
      state.raster_state.depth.test_enable =
          static_cast<std::uint8_t>(command.depth_enable);
      state.raster_state.depth.write_enable =
          static_cast<std::uint8_t>(command.depth_write);
      state.raster_state.depth.compare_op =
          static_cast<DepthCompareOp>(command.depth_func);
      state.raster_state.depth.clear_depth =
          FloatFromBits(command.depth_clear_bits);
      state.raster_state.blend.enable =
          static_cast<std::uint8_t>(command.blend_enable);
      state.raster_state.blend.rgb_equation =
          static_cast<BlendEquation>(command.blend_rgb_equation);
      state.raster_state.blend.alpha_equation =
          static_cast<BlendEquation>(command.blend_alpha_equation);
      state.raster_state.blend.source_rgb_factor =
          static_cast<BlendFactor>(command.blend_source_rgb_factor);
      state.raster_state.blend.destination_rgb_factor =
          static_cast<BlendFactor>(command.blend_destination_rgb_factor);
      state.raster_state.blend.source_alpha_factor =
          static_cast<BlendFactor>(command.blend_source_alpha_factor);
      state.raster_state.blend.destination_alpha_factor =
          static_cast<BlendFactor>(command.blend_destination_alpha_factor);
      state.raster_state.face_cull.enable = command.cull_face == 0 ? 0U : 1U;
      state.raster_state.face_cull.mode =
          command.cull_face == 1
              ? CullFaceMode::kFront
              : command.cull_face == 3 ? CullFaceMode::kFrontAndBack
                                       : CullFaceMode::kBack;
      state.raster_state.face_cull.front_face =
          command.front_ccw == 0 ? FrontFaceWinding::kClockwise
                                 : FrontFaceWinding::kCounterClockwise;
      state.raster_state.color_mask =
          static_cast<std::uint8_t>(command.color_mask);
    }
    if (driver_clear_like || driver_triangle || driver_indexed_quad ||
        driver_textured_triangles || driver_pco_triangles) {
      for (std::size_t component = 0; component < 4; ++component) {
        state.raster_state.clear_color[component] =
            FloatFromBits(command.clear_color_bits[component]);
      }
    }
    state.counters.frame = frame;
    state.counters.functional_frame = 1;

    std::vector<float> vertex_buffer;
    GlbenchFillTextureFixture texture_fixture;
    if (texture_case) {
      texture_fixture = driver_textured_triangles
                            ? MakeDriverTexturedTrianglesFixture(command)
                            : MakeGlbenchFillTextureFixture(functional_case);
    }
    const DriverPcoTopologyExpansion expanded_pco =
        driver_pco_triangles
            ? ExpandDriverPcoTopology(command)
            : DriverPcoTopologyExpansion{};
    const std::vector<std::uint8_t> &expanded_pco_vertices =
        expanded_pco.vertices;
    if (driver_triangle) {
      vertex_buffer = DriverTriangleFloat2Vertices(command);
      const std::vector<std::uint16_t> indices = {0, 1, 2};
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_index = 0;
      state.draw.index_count = static_cast<std::uint32_t>(indices.size());
      state.draw.base_vertex = 0;
      state.draw.index_format = IndexFormat::kUint16;
      StoreIndexBuffer(pool_, memory_, indices, &state);
    } else if (driver_indexed_quad) {
      vertex_buffer = {
          -1.0F, -1.0F,
          -1.0F, 1.0F,
          1.0F,  -1.0F,
          1.0F,  1.0F,
      };
      const std::vector<std::uint16_t> indices = {0, 2, 1, 1, 2, 3};
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_index = 0;
      state.draw.index_count = static_cast<std::uint32_t>(indices.size());
      state.draw.base_vertex = 0;
      state.draw.index_format = IndexFormat::kUint16;
      StoreIndexBuffer(pool_, memory_, indices, &state);
    } else if (indexed_triangle) {
      const GlbenchTriangleMeshShape &mesh =
          varyings ? kGlbenchVaryingsMesh
                   : attribute_fetch ? kGlbenchAttributeFetchMesh
                                     : kGlbenchTriangleSetupMesh;
      // Official varying cases call CreateLattice(size=1/4, shift=1)
      // independently of viewport size. Passing the 4×4 mesh dimensions as
      // the helper's coordinate denominator preserves exactly {-1,-.5,0,.5,1}.
      const std::uint32_t lattice_width = varyings ? mesh.width : state.width;
      const std::uint32_t lattice_height =
          varyings ? mesh.height : state.height;
      vertex_buffer = MakeGlbenchTriangleFloat2Vertices(
          lattice_width, lattice_height, mesh);
      const GlbenchTriangleWindingPattern winding =
          functional_case == FunctionalCase::kTriangleSetupHalfCulled
              ? GlbenchTriangleWindingPattern::kSrandZeroHalfCulled
              : GlbenchTriangleWindingPattern::kAllClockwise;
      const std::vector<std::uint16_t> indices =
          MakeGlbenchTriangleIndices(winding, mesh);
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_index = 0;
      state.draw.index_count = static_cast<std::uint32_t>(indices.size());
      state.draw.base_vertex = 0;
      state.draw.index_format = IndexFormat::kUint16;
      StoreIndexBuffer(pool_, memory_, indices, &state);
    } else if (driver_pco_triangles) {
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_vertex = 0;
      state.draw.vertex_count = static_cast<std::uint32_t>(
          expanded_pco_vertices.size() / command.vertex_stride);
      state.draw.index_format = IndexFormat::kNone;
    } else if (driver_textured_triangles) {
      vertex_buffer = texture_fixture.positions;
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_vertex = 0;
      state.draw.vertex_count = 6;
    } else if (texture_case) {
      vertex_buffer = texture_fixture.positions;
      state.draw.topology = PrimitiveTopology::kTriangleStrip;
      state.draw.first_vertex = 0;
      state.draw.vertex_count = 4;
    } else {
      vertex_buffer = {
          -1.0F, -1.0F,
          1.0F,  -1.0F,
          -1.0F, 1.0F,
          1.0F,  1.0F,
      };
      state.draw.topology = PrimitiveTopology::kTriangleStrip;
      state.draw.first_vertex = 0;
      state.draw.vertex_count = 4;
    }
    const VertexBufferResource vertex_resource =
        driver_pco_triangles
            ? StoreRawVertexBuffer(pool_, expanded_pco_vertices,
                                   kBuiltinVertexBufferGpuAddress +
                                       submission *
                                           kDriverSequenceVertexAddressStride,
                                   memory_)
            : StoreFloat2VertexBuffer(pool_, vertex_buffer,
                                      kBuiltinVertexBufferGpuAddress, memory_);
    std::vector<VertexBufferResource> vertex_resources{vertex_resource};
    if (texture_case) {
      vertex_resources.push_back(StoreFloat2VertexBuffer(
          pool_, texture_fixture.texture_coordinates,
          kBuiltinTexcoordBufferGpuAddress, memory_));
    }
    state.vertex_buffer_resources = StoreNewArray(pool_, vertex_resources);
    std::vector<VertexAttributeBinding> bindings;
    if (driver_pco_triangles) {
      if (IsIdeasPcoSequenceCommand(command)) {
        // Gallium expands indexed Ideas occurrences into aligned float4
        // records, but the original position/normal declarations are float3.
        // Fetch xyz and synthesize the GLES default w=1 so the binding's
        // readable mask exactly matches the real PCO VTXIN0..2 / VTXIN4..6
        // reads while the ABI allocation remains VTXIN4 / VTXIN8.
        bindings = {MakeDriverPcoFloat3Binding(0, command.vertex_stride, 0)};
        if (command.vertex_stride == 8U * sizeof(float)) {
          bindings.push_back(MakeDriverPcoFloat3Binding(
              4U * sizeof(float), command.vertex_stride, 4));
        }
      } else {
        const bool float2_position =
            command.vertex_stride == 2U * sizeof(float) &&
            command.vertex_pco_abi.vertex_inputs == 4;
        const bool terrain_main_layout =
            command.vertex_stride == 11U * sizeof(float) &&
            command.vertex_pco_abi.vertex_inputs == 16;
        if (terrain_main_layout) {
          // Terrain D3 is captured from four separate Gallium VBOs and
          // deep-copied into one occurrence stream as float3 position,
          // float3 normal, float3 tangent and float2 texcoord.  Preserve the
          // four vec4-aligned VTXIN destinations and synthesize the ordinary
          // GLES defaults only for each attribute's missing components.
          bindings = {
              MakeDriverPcoFloat3Binding(0, command.vertex_stride, 0),
              MakeDriverPcoFloat3Binding(3U * sizeof(float),
                                         command.vertex_stride, 4),
              MakeDriverPcoFloat3Binding(6U * sizeof(float),
                                         command.vertex_stride, 8),
              MakeDriverPcoFloat2Binding(9U * sizeof(float),
                                         command.vertex_stride, 12),
          };
        } else {
          bindings = {float2_position
                          ? MakeDriverPcoFloat2Binding(
                                0, command.vertex_stride, 0)
                          : MakeDriverPcoFloat3Binding(
                                0, command.vertex_stride, 0)};
          if (command.vertex_stride ==
                  kDriverPcoPositionNormalVertexStride ||
              command.vertex_stride ==
                  kDriverPcoPositionNormalTexcoordVertexStride) {
            bindings.push_back(MakeDriverPcoFloat3Binding(
                3U * sizeof(float), command.vertex_stride, 4));
          }
          if (command.vertex_stride ==
              kDriverPcoPositionNormalTexcoordVertexStride) {
            bindings.push_back(MakeDriverPcoFloat2Binding(
                6U * sizeof(float), command.vertex_stride, 8));
          }
        }
      }
    } else if (attribute_fetch && attribute_count > 1) {
      // GLBench binds every cN to the same VBO object. PVI supplies two
      // float32 components per input in consecutive VTXIN registers.
      bindings.reserve(attribute_count);
      for (std::uint32_t attribute = 0; attribute < attribute_count;
           ++attribute) {
        bindings.push_back(MakeFloat2Binding(0,
            static_cast<std::uint16_t>(attribute * 2U), 2));
      }
    } else if (texture_case) {
      bindings = {
          MakeFloat2Binding(0, 0, 2),
          MakeFloat2Binding(1, 2, 2),
      };
    } else {
      bindings = {MakeFloat2Binding(0, 0, 4)};
    }
    state.vertex_attribute_bindings = StoreNewArray(pool_, bindings);
    if (shader_varyings) {
      if (varying_count == 0)
        throw std::runtime_error("Submitter varying count is invalid");
      std::vector<ShaderVaryingBinding> linkages;
      linkages.reserve(varying_count);
      for (std::uint32_t varying = 0; varying < varying_count; ++varying) {
        ShaderVaryingBinding linkage;
        if (driver_pco_triangles) {
          const std::uint32_t component_offset =
              varying * kVaryingVectorComponentCount;
          const std::uint32_t component_count = std::min(
              kVaryingVectorComponentCount,
              command.varying_output_count - component_offset);
          linkage.vertex_output_base = static_cast<std::uint16_t>(
              command.varying_output_start + component_offset);
          linkage.coefficient_set_base = static_cast<std::uint16_t>(
              command.fragment_varying_start / kCoefficientSetDwordCount +
              component_offset);
          linkage.w_coefficient_set = static_cast<std::uint16_t>(
              command.fragment_position_start /
              kCoefficientSetDwordCount);
          linkage.component_count = static_cast<std::uint8_t>(
              component_count);
        } else {
          linkage.vertex_output_base = static_cast<std::uint16_t>(
              4U + varying * kVaryingVectorComponentCount);
          linkage.coefficient_set_base = static_cast<std::uint16_t>(
              1U + varying * kVaryingVectorComponentCount);
          linkage.w_coefficient_set = 0;
          linkage.component_count = texture_case
                                        ? 2U
                                        : kVaryingVectorComponentCount;
        }
        linkage.interpolation = InterpolationMode::kSmooth;
        if (!IsExactVaryingBinding(state, linkage, varying))
          throw std::runtime_error("Submitter varying linkage is invalid");
        linkages.push_back(linkage);
      }
      state.shader_varying_bindings =
          StoreNewArray(pool_, linkages);
    }
    if (texture_case) {
      if (memory_) {
        HostWriteArray(*memory_, texture_fixture.resource.gpu_address,
                       texture_fixture.texture_bytes);
      } else {
        texture_fixture.resource.data =
            StoreNewArray(pool_, texture_fixture.texture_bytes);
      }
      state.texture_resources = StoreNewArray(
          pool_, std::vector<TextureResource>{texture_fixture.resource});
      state.sampler_states = StoreNewArray(
          pool_, std::vector<SamplerState>{texture_fixture.sampler});
      state.vertex_shared_registers = StoreNewArray(
          pool_, std::vector<ShaderSharedRegister>{
                     {texture_fixture.vertex_scale_bits}});
      state.fragment_shared_registers = StoreNewArray(
          pool_, std::vector<std::uint32_t>(
                     texture_fixture.fragment_shared.begin(),
                     texture_fixture.fragment_shared.end()));
    } else if (driver_pco_triangles) {
      std::vector<std::uint32_t> vertex_shared_words = command.vertex_shared;
      std::vector<std::uint32_t> fragment_shared = command.fragment_shared;
      if (!command.sampled_textures.empty()) {
        if (!driver_pco_sequence_command || !memory_ ||
            command.sampled_textures.size() !=
                command.sampled_texture_count ||
            command.vertex_sampled_texture_count +
                    command.fragment_sampled_texture_count !=
                command.sampled_texture_count) {
          throw std::runtime_error(
              "Submitter PCO sequence texture state is inconsistent");
        }
        std::vector<TextureResource> vertex_resources(
            command.vertex_sampled_texture_count);
        std::vector<SamplerState> vertex_samplers(
            command.vertex_sampled_texture_count);
        std::vector<TextureResource> fragment_resources(
            command.fragment_sampled_texture_count);
        std::vector<SamplerState> fragment_samplers(
            command.fragment_sampled_texture_count);
        std::vector<bool> vertex_present(command.vertex_sampled_texture_count,
                                         false);
        std::vector<bool> fragment_present(
            command.fragment_sampled_texture_count, false);
        for (const DriverPcoSampledTexture &texture :
             command.sampled_textures) {
          const bool vertex_stage =
              texture.stage == DriverPcoShaderStage::kVertex;
          auto &resources =
              vertex_stage ? vertex_resources : fragment_resources;
          auto &samplers =
              vertex_stage ? vertex_samplers : fragment_samplers;
          auto &present =
              vertex_stage ? vertex_present : fragment_present;
          if (texture.declared_bytes_size == 0 ||
              texture.declared_bytes_size >
                  std::numeric_limits<std::uint32_t>::max() ||
              texture.declared_bytes_size >
                  kDriverPcoSequenceAttachmentStride ||
              texture.mip_count == 0 ||
              texture.mip_count > kMaximumTextureMipLevels ||
              texture.descriptor_set >= resources.size() ||
              present[texture.descriptor_set] || texture.binding != 0) {
            throw std::runtime_error(
              "Submitter PCO sequence resource metadata is invalid");
          }
          std::uint64_t gpu_address = 0;
          switch (texture.source) {
          case DriverPcoTextureSource::kExternalPayload:
            gpu_address = SequenceExternalTextureAddress(
                submission, texture.stage, texture.descriptor_set);
            if (texture.bytes.size() != texture.declared_bytes_size) {
              throw std::runtime_error(
                  "Submitter external PCO texture bytes are invalid");
            }
            HostWriteArray(*memory_, gpu_address, texture.bytes);
            DebugSequenceResourceHashes(*memory_, submission, texture,
                                        gpu_address);
            break;
          case DriverPcoTextureSource::kPreviousColorAttachment:
            if (texture.producer_command_index >= submission) {
              throw std::runtime_error(
                  "Submitter PCO color dependency is not earlier");
            }
            gpu_address = sequence_color_addresses.at(
                texture.producer_command_index);
            break;
          case DriverPcoTextureSource::kPreviousDepthAttachment:
            if (texture.producer_command_index >= submission) {
              throw std::runtime_error(
                  "Submitter PCO depth dependency is not earlier");
            }
            gpu_address = sequence_depth_addresses.at(
                texture.producer_command_index);
            break;
          }
          if (!memory_->backing().Contains(
                  gpu_address,
                  static_cast<std::size_t>(texture.declared_bytes_size))) {
            throw std::runtime_error(
                "Submitter PCO sequence resource is absent from DRAM");
          }
          PatchPcoDescriptorAddress(
              vertex_stage ? &vertex_shared_words : &fragment_shared,
              texture.descriptor_set, gpu_address);

          TextureResource resource;
          resource.gpu_address = gpu_address;
          resource.byte_size =
              static_cast<std::uint32_t>(texture.declared_bytes_size);
          resource.mip_count = static_cast<std::uint8_t>(texture.mip_count);
          resource.format =
              texture.format == "PIPE_FORMAT_Z32_UNORM"
                  ? TextureFormat::kZ32Unorm
                  : texture.format == "PIPE_FORMAT_R8G8B8A8_UNORM"
                        ? TextureFormat::kRgba8Unorm
                        : texture.format == "PIPE_FORMAT_R8G8B8X8_UNORM"
                              ? TextureFormat::kRgbx8Unorm
                        : throw std::runtime_error(
                              "Submitter PCO sequence texture format is "
                              "unsupported");
          resource.layout = TextureLayout::kLinear;
          resource.descriptor_set =
              static_cast<std::uint8_t>(texture.descriptor_set);
          resource.binding = static_cast<std::uint8_t>(texture.binding);
          for (std::size_t level = 0; level < texture.mip_count; ++level) {
            resource.mip[level] = {
                texture.mip[level].width,
                texture.mip[level].height,
                texture.mip[level].row_pitch_bytes,
                texture.mip[level].offset_bytes,
            };
          }
          resources[texture.descriptor_set] = resource;

          SamplerState sampler;
          sampler.min_filter = texture.min_filter == 0
                                   ? TextureFilter::kNearest
                                   : TextureFilter::kLinear;
          sampler.mag_filter = texture.mag_filter == 0
                                   ? TextureFilter::kNearest
                                   : TextureFilter::kLinear;
          sampler.mip_filter = texture.mip_filter == 0
                                   ? TextureFilter::kNearest
                                   : TextureFilter::kLinear;
          sampler.wrap_u = texture.wrap_u == 0
                               ? TextureWrapMode::kClampToEdge
                               : TextureWrapMode::kRepeat;
          sampler.wrap_v = texture.wrap_v == 0
                               ? TextureWrapMode::kClampToEdge
                               : TextureWrapMode::kRepeat;
          sampler.min_lod_u4_6 =
              static_cast<std::uint16_t>(texture.min_lod_u4_6);
          sampler.max_lod_u4_6 =
              static_cast<std::uint16_t>(texture.max_lod_u4_6);
          sampler.normalized_coordinates =
              static_cast<std::uint8_t>(texture.normalized_coordinates);
          sampler.base_mip_level = 0;
          sampler.descriptor_set =
              static_cast<std::uint8_t>(texture.descriptor_set);
          sampler.binding = static_cast<std::uint8_t>(texture.binding);
          samplers[texture.descriptor_set] = sampler;
          present[texture.descriptor_set] = true;
        }
        if (std::find(vertex_present.begin(), vertex_present.end(), false) !=
                vertex_present.end() ||
            std::find(fragment_present.begin(), fragment_present.end(),
                      false) != fragment_present.end()) {
          throw std::runtime_error(
              "Submitter PCO sequence descriptor sets are not stage-dense");
        }
        if (!vertex_resources.empty()) {
          state.vertex_texture_resources =
              StoreNewArray(pool_, vertex_resources);
          state.vertex_sampler_states =
              StoreNewArray(pool_, vertex_samplers);
        }
        if (!fragment_resources.empty()) {
          state.texture_resources = StoreNewArray(pool_, fragment_resources);
          state.sampler_states = StoreNewArray(pool_, fragment_samplers);
        }
      } else if (driver_pco_texture) {
        TextureResource resource;
        resource.gpu_address = kGlbenchTextureGpuAddress;
        resource.byte_size = static_cast<std::uint32_t>(
            command.sampled_texture_bytes.size());
        resource.mip_count = 1;
        resource.format = TextureFormat::kRgbx8Unorm;
        resource.layout = TextureLayout::kLinear;
        resource.mip[0].width = command.sampled_texture_width;
        resource.mip[0].height = command.sampled_texture_height;
        resource.mip[0].row_pitch_bytes = command.sampled_texture_row_pitch;
        resource.mip[0].offset_bytes = 0;
        if (memory_) {
          HostWriteArray(*memory_, resource.gpu_address,
                         command.sampled_texture_bytes);
        } else {
          resource.data = StoreNewArray(
              pool_, command.sampled_texture_bytes);
        }
        state.texture_resources = StoreNewArray(
            pool_, std::vector<TextureResource>{resource});

        SamplerState sampler;
        sampler.min_filter = TextureFilter::kNearest;
        sampler.mag_filter = TextureFilter::kNearest;
        sampler.mip_filter = TextureFilter::kNearest;
        sampler.wrap_u = TextureWrapMode::kClampToEdge;
        sampler.wrap_v = TextureWrapMode::kClampToEdge;
        sampler.normalized_coordinates = 1;
        sampler.base_mip_level = 0;
        state.sampler_states = StoreNewArray(
            pool_, std::vector<SamplerState>{sampler});
      }
      std::vector<ShaderSharedRegister> vertex_shared;
      vertex_shared.reserve(vertex_shared_words.size());
      for (const std::uint32_t value : vertex_shared_words)
        vertex_shared.push_back({value});
      if (!vertex_shared.empty())
        state.vertex_shared_registers = StoreNewArray(pool_, vertex_shared);
      if (!fragment_shared.empty()) {
        state.fragment_shared_registers =
            StoreNewArray(pool_, fragment_shared);
      }
    }
    state.vertex_code = StoreNewArray(
        pool_, driver_pco_triangles ? command.vertex_pco
                    : texture_case ? FillTexNearestVertexPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderEight
                    ? VaryingsEightVertexPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderFour
                          ? VaryingsFourVertexPcoBinary()
                          : functional_case == FunctionalCase::kVaryingsShaderTwo
                          ? VaryingsTwoVertexPcoBinary()
                          : varyings ? VaryingsOneVertexPcoBinary()
                        : attribute_count == 8
                   ? AttributeFetchEightAttributeVertexPcoBinary()
                   : attribute_count == 4
                         ? AttributeFetchFourAttributeVertexPcoBinary()
                         : attribute_count == 2
                         ? AttributeFetchTwoAttributeVertexPcoBinary()
                         : attribute_fetch ? AttributeFetchVertexPcoBinary()
                                           : FillSolidVertexPcoBinary());
    state.fragment_code = StoreNewArray(
        pool_, driver_pco_triangles ? command.fragment_pco
                    : texture_case ? FillTexNearestFragmentPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderEight
                    ? VaryingsEightFragmentPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderFour
                          ? VaryingsFourFragmentPcoBinary()
                          : functional_case == FunctionalCase::kVaryingsShaderTwo
                          ? VaryingsTwoFragmentPcoBinary()
                          : varyings ? VaryingsOneFragmentPcoBinary()
                        : attribute_fetch
                   ? AttributeFetchGrayFragmentPcoBinary()
                   : driver_indexed_quad ? FillSolidBlackFragmentPcoBinary()
                   : driver_triangle
                         ? FillSolidFragmentPcoBinary()
                   : functional_case == FunctionalCase::kTriangleSetupHalfCulled
                   ? TriangleSetupCyanFragmentPcoBinary()
                   : triangle_setup ? TriangleSetupOrangeFragmentPcoBinary()
                                    : FillSolidFragmentPcoBinary());
    state.drawlist_stats = StoreNewArray(pool_, std::vector<DrawListStats>{{}});

    const PoolHandle handle = pool_.Allocate(sizeof(PipelineState));
    if (output.num_free() == 0)
      fifo_stalls_++;
    state.counters.fifo_stall_events = fifo_stalls_;
    StorePipelineState(pool_, handle, state);
    output.write({handle, frame, frame});
    if (!options_.driver_commands.empty() && sequence_completion_ &&
        submission + 1U < submission_count)
      wait(*sequence_completion_);
  }
}

} // namespace pvrgpu::stub
