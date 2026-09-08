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
#include "uniform_buffers.h"
#include "shader_images.h"
#include "common/geometry_emission.h"
#include "common/tessellation_state.h"
#include "common/stream_output_types.h"

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

/*
 * Where a sequence's own buffers live in DRAM.
 *
 * Every submission of a sequence coexists in memory, so each needs its own
 * vertex and index addresses; a shared one is not a tight packing but a
 * corruption.  The index buffer used to be a single fixed address for the whole
 * sequence, and the vertex region was sixteen strides long before it ran into
 * the texture-coordinate and index regions above it.  A 33-draw indexed
 * sequence therefore read submission 32's vertex bytes as its indices -- dEQP's
 * fragment_ops.stencil.* aborted the model with a resolved index of 49024,
 * which is the top half of the float -1.0f sitting at that address.
 *
 * The regions are now equal, disjoint and bounded, and running past one throws
 * instead of aliasing into the next.
 */
// One slot per submission, with room for the largest single draw the corpus
// carries several times over: glmark2's terrain expands one draw to 16.5 MiB.
// A slot a draw can overrun is a slot that silently lands on its neighbour's
// vertices, so the margin is deliberate and the overrun is refused by name.
inline constexpr std::uint64_t kDriverSequenceVertexAddressStride =
    UINT64_C(0x20000000);
inline constexpr std::size_t kDriverSequenceAddressSlots =
    kDriverPcoMaximumSequenceCommands;
// DRAM is a sparse page map, so the regions cost nothing until written and are
// sized for the longest sequence the API accepts rather than for the corpus in
// hand.  Each is derived from the slot geometry so it cannot drift from it, and
// they sit above the attachment regions of model_types.h.
inline constexpr std::uint64_t kDriverSequenceAddressRegionBytes =
    kDriverSequenceVertexAddressStride *
    static_cast<std::uint64_t>(kDriverSequenceAddressSlots);
inline constexpr std::uint64_t kBuiltinVertexBufferGpuAddress =
    UINT64_C(0x10000000000);
inline constexpr std::uint64_t kBuiltinIndexBufferGpuAddress =
    kBuiltinVertexBufferGpuAddress + kDriverSequenceAddressRegionBytes;
inline constexpr std::uint64_t kBuiltinTexcoordBufferGpuAddress =
    kBuiltinIndexBufferGpuAddress + kDriverSequenceAddressRegionBytes;
// UBO addresses are full 64-bit descriptor addresses. Reserve the region after
// all three large vertex/index/UV regions, not inside texture/attachment space.
inline constexpr std::uint64_t kUniformBufferGpuAddressBase =
    kBuiltinTexcoordBufferGpuAddress + kDriverSequenceAddressRegionBytes;
inline constexpr std::uint64_t kGeometryPrimitiveGpuAddressBase =
    kUniformBufferGpuAddressBase +
    kDriverSequenceAddressSlots * 5U * kMaximumUniformBuffersPerStage *
        kMaximumUniformBufferBytes;
inline constexpr std::uint64_t kGeometryPrimitiveGpuAddressStride = 4096;
inline constexpr std::uint64_t kTessellationGpuAddressBase =
    kGeometryPrimitiveGpuAddressBase +
    kDriverSequenceAddressSlots * kGeometryPrimitiveGpuAddressStride;
inline constexpr std::uint64_t kStreamOutputResourceAddressStride = UINT64_C(0x10000000);
inline constexpr std::uint64_t kStreamOutputGpuAddressBase =
    kParameterCoefficientsGpuAddress + kParameterRegionBytes;
static_assert(kStreamOutputGpuAddressBase <= UINT64_MAX -
                  kDriverSequenceAddressSlots * 4U * kStreamOutputResourceAddressStride,
              "stream output address region wraps");
inline constexpr std::uint64_t kShaderImageGpuAddressBase = kStreamOutputGpuAddressBase +
    kDriverSequenceAddressSlots * 4U * kStreamOutputResourceAddressStride;
static_assert(kShaderImageGpuAddressBase <= UINT64_MAX -
    kDriverSequenceAddressSlots * kMaximumFragmentImages * kMaximumFragmentImageBytes,
    "fragment image address region wraps");
// 以全部合法 slots 驗證區域，而非只以本輪實際 draw 數推測不會相撞。
static_assert(kDriverPcoMrtColorAddressBase +
                  kDriverSequenceAddressSlots * kMaxRenderTargets *
                      kDriverPcoSequenceAttachmentStride <=
                  kDriverPcoSequenceExternalAddressBase,
              "MRT attachments overlap external textures");
static_assert(kDriverPcoSequenceExternalAddressEnd <= kBuiltinVertexBufferGpuAddress,
              "external textures overlap vertex buffers");
static_assert(kTessellationGpuAddressBase +
                  kDriverSequenceAddressSlots * kTessellationDrawAddressStride <= kParameterTrianglesGpuAddress,
              "driver buffers overlap parameter memory");

std::uint64_t SequenceUniformBufferAddress(
    std::size_t submission, DriverPcoShaderStage stage, std::uint32_t block) {
  const auto stage_index = static_cast<unsigned>(stage);
  if (submission >= kDriverSequenceAddressSlots || stage_index > 4 ||
      block >= kMaximumUniformBuffersPerStage)
    throw std::runtime_error("Submitter uniform buffer address slot is invalid");
  const std::uint64_t slot =
      (static_cast<std::uint64_t>(submission) * 5U + stage_index) *
          kMaximumUniformBuffersPerStage + block;
  if (slot > (std::numeric_limits<std::uint64_t>::max() -
              kUniformBufferGpuAddressBase) / kMaximumUniformBufferBytes)
    throw std::overflow_error("Submitter uniform buffer address wraps");
  return kUniformBufferGpuAddressBase + slot * kMaximumUniformBufferBytes;
}

// The address a submission's vertex or index buffer owns.  Sequences longer
// than the region holds are refused by name rather than wrapped onto a
// neighbour.
std::uint64_t SequenceBufferAddress(std::uint64_t base,
                                    std::size_t submission,
                                    const char *what) {
  if (submission >= kDriverSequenceAddressSlots) {
    throw std::runtime_error(
        std::string("Submitter sequence has more submissions than the ") +
        what + " address region holds: submission=" +
        std::to_string(submission) +
        " slots=" + std::to_string(kDriverSequenceAddressSlots));
  }
  return base + static_cast<std::uint64_t>(submission) *
                    kDriverSequenceVertexAddressStride;
}

// Take the next attachment slot in a region.
//
// Each region holds kDriverPcoSequenceAttachmentSlots strides before it runs
// into the next one, so running past that would silently place a colour
// attachment on top of a depth attachment.  Slots are handed out as surfaces
// are created rather than indexed by ordinal: a sequence of any length is fine
// as long as the surfaces it creates fit, and a draw that continues from an
// earlier surface creates none.
std::size_t TakeSequenceAttachmentSlot(std::size_t *next, const char *region) {
  if (!next || *next >= kDriverPcoSequenceAttachmentSlots) {
    throw std::runtime_error(
        std::string("Submitter sequence creates more ") + region +
        " attachments than its address region holds");
  }
  return (*next)++;
}

std::uint64_t SequenceExternalTextureAddress(
    std::size_t submission, DriverPcoShaderStage stage,
    std::uint32_t descriptor_set, std::uint64_t byte_size,
    DriverPcoExternalTextureAllocation *allocation) {
  if (submission >= kDriverPcoMaximumNestedSequenceCommands ||
      (stage != DriverPcoShaderStage::kVertex &&
       stage != DriverPcoShaderStage::kFragment &&
       stage != DriverPcoShaderStage::kGeometry) ||
      descriptor_set >= kPcoMaximumTextureDescriptorSets) {
    throw std::runtime_error(
        "Submitter sequence external texture slot is out of bounds");
  }
  std::uint64_t address = 0;
  if (!AllocateSequenceExternalTextureAddress(byte_size, allocation, &address))
    throw std::runtime_error(
        "Submitter sequence external texture allocation exceeds its region/payload limit");
  return address;
}

float FloatFromBits(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// The expected viewport scale up to the sign of Y: a window-system framebuffer
// is y-flipped, so GL states scale_y = -h/2 there.  Negating an IEEE float
// toggles only its sign bit.
bool ViewportScaleMatches(const std::array<std::uint32_t, 3> &actual,
                          const std::array<std::uint32_t, 3> &expected) {
  return actual[0] == expected[0] &&
         (actual[1] == expected[1] ||
          actual[1] == (expected[1] ^ UINT32_C(0x80000000))) &&
         actual[2] == expected[2];
}

// Mesa flips front_ccw together with the viewport Y sign
// (st_atom_rasterizer.c: front_ccw ^= 1 for Y_0_BOTTOM targets, whose
// viewport is not inverted), so front_ccw XOR (scale_y > 0) is invariant for
// one GL front-face setting.  The validated slice is (front_ccw=0, scale_y>0)
// -> clockwise; extend it by that invariant instead of re-deriving GL's
// convention from scratch.
FrontFaceWinding FrontFaceFromDriverCommand(std::uint32_t front_ccw,
                                            std::uint32_t scale_y_bits) {
  const bool scale_y_positive = FloatFromBits(scale_y_bits) > 0.0F;
  const bool clockwise = (front_ccw == 0) == scale_y_positive;
  return clockwise ? FrontFaceWinding::kClockwise
                   : FrontFaceWinding::kCounterClockwise;
}

/*
 * How many raw 32-bit channels a colour attachment stores, or zero when it is
 * not one of the integer formats.  dEQP's shader executor renders a scalar
 * result into a one-channel target, a vec2 into two channels and a vec3 or
 * vec4 into four -- GLES has no three-channel integer format -- and the
 * result's signedness picks UINT or SINT.  The stored pixel is the same raw
 * dword either way; only the host's reading of it differs.  This is also what
 * widens the framebuffer past four bytes a pixel.
 */
std::uint32_t ColorAttachmentRawDwords(const std::string &format) {
  if (format == "PIPE_FORMAT_R32_UINT" || format == "PIPE_FORMAT_R32_SINT")
    return 1U;
  if (format == "PIPE_FORMAT_R32G32_UINT" ||
      format == "PIPE_FORMAT_R32G32_SINT") {
    return 2U;
  }
  if (format == "PIPE_FORMAT_R32G32B32A32_UINT" ||
      format == "PIPE_FORMAT_R32G32B32A32_SINT") {
    return 4U;
  }
  return 0U;
}

/* The colour formats a generic PCO draw may target: eight-bit UNORM, or one,
 * two or four raw 32-bit integer channels of either signedness. */
bool DriverPcoColorAttachmentFormatSupported(const std::string &format) {
  return format == "PIPE_FORMAT_R8G8B8A8_UNORM" ||
         PackedUnormFormatFromName(format) != PackedUnormFormat::kNone ||
         format == "PIPE_FORMAT_R32G32B32A32_FLOAT" ||
         ColorAttachmentRawDwords(format) != 0U;
}

bool PcoSingleDrawResolutionSupported(const DriverCommand &command) {
  // The rasterizer is resolution independent; the single-draw path only needs
  // a full-surface render target within the model's addressable extent.
  return command.width == command.framebuffer_width &&
         command.height == command.framebuffer_height &&
         command.framebuffer_width != 0 && command.framebuffer_height != 0 &&
         command.framebuffer_width <= 4096 &&
         command.framebuffer_height <= 4096;
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
inline constexpr std::uint32_t kPipePrimPoints = 0;
inline constexpr std::uint32_t kPipePrimLines = 1;
inline constexpr std::uint32_t kPipePrimLineLoop = 2;
inline constexpr std::uint32_t kPipePrimLineStrip = 3;
inline constexpr std::uint32_t kPipePrimTriangles = 4;
inline constexpr std::uint32_t kPipePrimTriangleStrip = 5;
inline constexpr std::uint32_t kPipePrimTriangleFan = 6;

// Non-indexed triangle topologies ExpandDriverPcoTopologyImpl can turn into a
// triangle list: a whole-triangle list, or a strip/fan of three or more
// vertices.
bool DriverPcoArrayTopologyIsExpandable(const DriverCommand &command) {
  // An indexed draw assembles its primitives from the index buffer.
  const std::uint32_t count =
      command.indexed != 0 ? command.index_count : command.vertex_count;
  switch (command.primitive_mode) {
    case kPipePrimPoints:
      return count >= 1U;
    case kPipePrimLines:
      return count >= 2U && count % 2U == 0U;
    case kPipePrimLineLoop:
    case kPipePrimLineStrip:
      return count >= 2U;
    case kPipePrimTriangles:
      return count >= 3U && count % 3U == 0U;
    case kPipePrimTriangleStrip:
    case kPipePrimTriangleFan:
      return count >= 3U;
    default:
      return false;
  }
}

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
  // Scale is half the viewport extent, which is the attachment only when the
  // draw renders to the whole of it.
  const std::array<std::uint32_t, 3> viewport_bits =
      PcoViewportBits(command.width, command.height);
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
      !PcoSingleDrawResolutionSupported(command) ||
      !DriverPcoColorAttachmentFormatSupported(command.format) ||
      command.clear_color_bits != kOpaqueBlack || command.first_vertex != 0 ||
      command.instance_count != 1 || command.indexed > 1 ||
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
      !ViewportScaleMatches(command.viewport_scale_bits, viewport_bits) ||
      command.front_ccw > 1 ||
      (command.cull_face != 0 && command.cull_face != 2) ||
      command.fill_front != 0 || command.fill_back != 0 ||
      command.scissor != 0 || command.rasterizer_discard != 0 ||
      command.multisample != 0 || command.half_pixel_center != 1 ||
      command.bottom_edge_rule > 1 || command.clip_halfz != 0 ||
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
  // Scale is half the viewport extent, which is the attachment only when the
  // draw renders to the whole of it.
  const std::array<std::uint32_t, 3> viewport_bits =
      PcoViewportBits(command.width, command.height);
  const bool conditionals_layout =
      command.vertex_stride == kDriverPcoPositionVertexStride &&
      command.vertex_pco_abi.vertex_inputs == 4;
  // Untextured position/colour layout: six floats for a vec2 position, eight
  // for a vec4 one.
  const bool generic_attribute_layout =
      command.vertex_attribute_count != 0 &&
      command.vertex_pco_abi.vertex_inputs ==
          command.vertex_attribute_count * 4U;
  const bool color_layout =
      generic_attribute_layout ||
      (((command.vertex_stride == 6U * sizeof(float) &&
         command.vertex_pco_abi.shareds == 0) ||
        command.vertex_stride == 8U * sizeof(float)) &&
       command.vertex_pco_abi.vertex_inputs == 8 &&
       command.varying_output_count == 4 &&
       command.fragment_varying_count == 16);
  const bool lit_mesh_layout =
      command.vertex_stride == kDriverPcoPositionNormalVertexStride &&
      command.vertex_pco_abi.vertex_inputs == 8 &&
      !color_layout;
  const bool texture_layout =
      command.vertex_stride ==
          kDriverPcoPositionNormalTexcoordVertexStride &&
      command.vertex_pco_abi.vertex_inputs == 12;
  if (!PcoSingleDrawResolutionSupported(command) ||
      !DriverPcoColorAttachmentFormatSupported(command.format) ||
      command.clear_color_bits != kOpaqueBlack ||
      (!conditionals_layout && !lit_mesh_layout && !texture_layout &&
       !color_layout) ||
      command.vertex_count == 0 ||
      !DriverPcoArrayTopologyIsExpandable(command) ||
      command.first_vertex != 0 || command.instance_count != 1 ||
      command.indexed > 1 ||
      command.vertex_pco.empty() || command.fragment_pco.empty() ||
      command.vertex_pco.size() > kDriverPcoMaximumBinaryBytes ||
      command.fragment_pco.size() > kDriverPcoMaximumBinaryBytes ||
      command.vertex_shared.size() != command.vertex_pco_abi.shareds ||
      command.fragment_shared.size() != command.fragment_pco_abi.shareds ||
      !DriverPcoStageAbiIsBounded(command.vertex_pco_abi, color_layout) ||
      !DriverPcoStageAbiIsBounded(command.fragment_pco_abi, color_layout)) {
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
      (color_layout &&
       (command.varying_output_start != 4 ||
        command.varying_output_count != 4 ||
        command.fragment_position_count != 4 ||
        command.fragment_varying_start != 4 ||
        command.fragment_varying_count != 16)) ||
      !ViewportScaleMatches(command.viewport_scale_bits, viewport_bits) ||
      command.front_ccw > 1 ||
      (!color_layout && command.cull_face != 2) ||
      command.fill_front != 0 || command.fill_back != 0 ||
      command.scissor != 0 || command.rasterizer_discard != 0 ||
      command.multisample != 0 || command.half_pixel_center != 1 ||
      command.bottom_edge_rule > 1 || command.clip_halfz != 0 ||
      command.depth_clip_near != 1 || command.depth_clip_far != 1 ||
      command.depth_clamp != 0 || command.sample_mask != UINT32_MAX ||
      command.color_mask != 0x0f || command.blend_enable != 0 ||
      command.dither != 1 ||
      (!color_layout &&
       (command.depth_enable != 1 || command.depth_write != 1 ||
        command.depth_func != 3 ||
        command.depth_clear_bits != UINT32_C(0x3f800000) ||
        command.depth_format == 0))) {
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
                               std::uint64_t gpu_address,
                               std::uint32_t descriptor_start = 0) {
  if (!shared || descriptor_set >= kPcoMaximumTextureDescriptorSets ||
      gpu_address == 0 || (gpu_address & 3U) != 0) {
    throw std::runtime_error(
        "Submitter PCO descriptor relocation arguments are invalid");
  }
  const std::size_t base =
      descriptor_start + static_cast<std::size_t>(descriptor_set) *
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
  // Spilling past the slot would land on the next submission's vertices, which
  // reads as corruption rather than as the size problem it is.
  if (bytes.size() > kDriverSequenceVertexAddressStride) {
    throw std::runtime_error(
        "Submitter vertex buffer is larger than the address slot it owns: "
        "bytes=" + std::to_string(bytes.size()) +
        " slot=" + std::to_string(kDriverSequenceVertexAddressStride));
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
  if (command.indexed != 0) {
    // Vertex fetch walks the index buffer and expands the topology itself, so
    // the vertex stream is forwarded untouched and index reuse stays visible.
    result.vertices = command.raw_vertex_data;
    result.input_primitives =
        command.primitive_mode == kPipePrimTriangles
            ? command.index_count / 3U
            : (command.index_count >= 3U ? command.index_count - 2U : 0U);
    result.emitted_primitives = result.input_primitives;
    return result;
  }
  if (command.primitive_mode == kPipePrimTriangles) {
    result.vertices = command.raw_vertex_data;
    result.input_primitives = command.vertex_count / 3U;
    result.emitted_primitives = result.input_primitives;
    return result;
  }
  if (command.primitive_mode <= kPipePrimLineStrip) {
    // Lines and points enter setup as the degenerate triangles the topology
    // expansion encodes them as -- a line as (a, b, b) and a point as
    // (p, p, p) -- so ClipCull reads endpoint a from vertex 0 and endpoint b
    // from vertex 1 and widens them into a real screen-space quad.
    //
    // That encoding repeats whole vertices, twice within a primitive and again
    // between the adjacent segments of a strip or loop, so the source vertex
    // behind each copy is recorded here for the same reason a strip or fan
    // records it: vertex fetch shades one lane per distinct source, and
    // vs_invocations counts the vertices the draw submitted rather than the
    // expansion's duplication.
    if (command.vertex_stride == 0)
      throw std::runtime_error("Submitter driver PCO topology is empty");
    const std::uint32_t vertices_per_primitive =
        command.primitive_mode == kPipePrimPoints ? 1U : 2U;
    std::vector<std::uint32_t> starts;
    if (command.primitive_mode == kPipePrimPoints) {
      for (std::uint32_t v = 0; v < command.vertex_count; ++v)
        starts.push_back(v);
    } else if (command.primitive_mode == kPipePrimLines) {
      for (std::uint32_t v = 0; v + 1 < command.vertex_count; v += 2)
        starts.push_back(v);
    } else {
      for (std::uint32_t v = 0; v + 1 < command.vertex_count; ++v)
        starts.push_back(v);
    }
    const bool closes_loop =
        command.primitive_mode == kPipePrimLineLoop && command.vertex_count >= 2;
    result.input_primitives =
        static_cast<std::uint32_t>(starts.size()) + (closes_loop ? 1U : 0U);
    const auto append_vertex = [&](std::uint32_t vertex) {
      const std::size_t begin =
          static_cast<std::size_t>(vertex) * command.vertex_stride;
      const std::size_t end = begin + command.vertex_stride;
      if (end > command.raw_vertex_data.size())
        throw std::runtime_error("Submitter PCO topology index is out of range");
      result.vertices.insert(result.vertices.end(),
                             command.raw_vertex_data.begin() + begin,
                             command.raw_vertex_data.begin() + end);
      result.source_vertices.push_back(vertex);
    };
    for (std::uint32_t start : starts) {
      const std::uint32_t second =
          vertices_per_primitive == 1U ? start : start + 1U;
      append_vertex(start);
      append_vertex(second);
      append_vertex(second);
    }
    if (closes_loop) {
      append_vertex(command.vertex_count - 1U);
      append_vertex(0U);
      append_vertex(0U);
    }
    result.emitted_primitives = result.input_primitives;
    if (result.source_vertices.size() !=
            static_cast<std::uint64_t>(result.emitted_primitives) * 3U ||
        result.vertices.size() !=
            static_cast<std::uint64_t>(result.emitted_primitives) * 3U *
                command.vertex_stride) {
      throw std::runtime_error(
          "Submitter PCO topology expansion accounting is inconsistent");
    }
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
    result.source_vertices.push_back(vertex);
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
      result.source_vertices.size() !=
          static_cast<std::uint64_t>(result.emitted_primitives) * 3U ||
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
                      const std::vector<T> &indices, PipelineState *state,
                      std::uint64_t gpu_address) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (!state || indices.empty() ||
      indices.size() > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::runtime_error("Submitter index buffer size is invalid");
  }
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(indices.size()) * sizeof(T);
  if (bytes > kDriverSequenceVertexAddressStride) {
    throw std::runtime_error(
        "Submitter index buffer is larger than the address slot it owns: "
        "bytes=" + std::to_string(bytes) +
        " slot=" + std::to_string(kDriverSequenceVertexAddressStride));
  }
  if (memory) {
    HostWriteArray(*memory, gpu_address, indices);
    state->index_buffer_gpu_address = gpu_address;
    state->index_buffer_bytes = indices.size() * sizeof(T);
  } else {
    state->vertex_indices = StoreNewArray(pool, indices);
  }
}

PrimitiveTopology DriverPcoTopologyFor(std::uint32_t primitive_mode) {
  switch (primitive_mode) {
    case kPipePrimPoints:
      return PrimitiveTopology::kPoints;
    case kPipePrimLines:
      return PrimitiveTopology::kLines;
    case kPipePrimLineLoop:
      return PrimitiveTopology::kLineLoop;
    case kPipePrimLineStrip:
      return PrimitiveTopology::kLineStrip;
    case kPipePrimTriangles:
      return PrimitiveTopology::kTriangleList;
    case kPipePrimTriangleStrip:
      return PrimitiveTopology::kTriangleStrip;
    case kPipePrimTriangleFan:
      return PrimitiveTopology::kTriangleFan;
    case 10: return PrimitiveTopology::kLinesAdjacency;
    case 11: return PrimitiveTopology::kLineStripAdjacency;
    case 12: return PrimitiveTopology::kTrianglesAdjacency;
    case 13: return PrimitiveTopology::kTriangleStripAdjacency;
    case 14: return PrimitiveTopology::kPatches;
    default:
      throw std::runtime_error("Submitter driver PCO topology is unsupported");
  }
}

GeometryInputTopology GeometryInputTopologyFor(std::uint32_t mode) {
  switch (mode) {
    case 0: return GeometryInputTopology::kPoints;
    case 1: return GeometryInputTopology::kLines;
    case 2: return GeometryInputTopology::kLineLoop;
    case 3: return GeometryInputTopology::kLineStrip;
    case 4: return GeometryInputTopology::kTriangles;
    case 5: return GeometryInputTopology::kTriangleStrip;
    case 6: return GeometryInputTopology::kTriangleFan;
    case 10: return GeometryInputTopology::kLinesAdjacency;
    case 11: return GeometryInputTopology::kLineStripAdjacency;
    case 12: return GeometryInputTopology::kTrianglesAdjacency;
    case 13: return GeometryInputTopology::kTriangleStripAdjacency;
    default: throw std::runtime_error("Submitter invalid GS input topology");
  }
}

std::vector<GeometryInputPrimitive> GeometryInputsFor(const DriverCommand &command) {
  const std::uint32_t count = command.indexed ? command.index_count : command.vertex_count;
  const std::uint32_t per_instance = command.geometry_vertices_per_instance;
  if (!per_instance || count % per_instance || command.first_index)
    throw std::runtime_error("Submitter invalid GS input instance range");
  std::vector<GeometryInputPrimitive> result(count);
  std::vector<std::uint32_t> occurrences(per_instance);
  std::size_t total = 0;
  for (std::uint32_t instance = 0; instance < count / per_instance; ++instance) {
    for (std::uint32_t i = 0; i < per_instance; ++i)
      occurrences[i] = instance * per_instance + i;
    std::size_t written = 0;
    const auto status = AssembleGeometryInputPrimitives(
        GeometryInputTopologyFor(command.primitive_mode), occurrences.data(),
        occurrences.size(), false, UINT32_MAX, instance,
        result.data() + total, result.size() - total, written);
    if (status != GeometryEmissionStatus::kSuccess)
      throw std::runtime_error(std::string("Submitter GS input assembly: ") +
                               GeometryEmissionStatusName(status));
    total += written;
  }
  result.resize(total);
  for (const auto &primitive : result)
    if (primitive.vertex_count != command.geometry_input_primitive_vertices)
      throw std::runtime_error("Submitter GS input shader/topology mismatch");
  return result;
}

IndexFormat DriverPcoIndexFormatFor(std::uint32_t index_size) {
  switch (index_size) {
    case 1:
      return IndexFormat::kUint8;
    case 2:
      return IndexFormat::kUint16;
    case 4:
      return IndexFormat::kUint32;
    default:
      throw std::runtime_error("Submitter driver PCO index size is invalid");
  }
}

// Republish the capsule's index payload as the typed buffer vertex fetch reads.
void StoreDriverPcoIndexBuffer(MemoryPool &pool, GpuMemorySystem *memory,
                               const DriverCommand &command,
                               PipelineState *state,
                               std::uint64_t gpu_address) {
  const std::uint64_t index_end =
      static_cast<std::uint64_t>(command.first_index) + command.index_count;
  if (command.index_size == 0 ||
      index_end * command.index_size != command.raw_index_data.size()) {
    throw std::runtime_error("Submitter driver PCO index payload is invalid");
  }
  const std::size_t count = static_cast<std::size_t>(index_end);
  if (command.index_size == 1) {
    std::vector<std::uint8_t> indices(command.raw_index_data.begin(),
                                      command.raw_index_data.end());
    StoreIndexBuffer(pool, memory, indices, state, gpu_address);
  } else if (command.index_size == 2) {
    std::vector<std::uint16_t> indices(count);
    std::memcpy(indices.data(), command.raw_index_data.data(),
                command.raw_index_data.size());
    StoreIndexBuffer(pool, memory, indices, state, gpu_address);
  } else {
    std::vector<std::uint32_t> indices(count);
    std::memcpy(indices.data(), command.raw_index_data.data(),
                command.raw_index_data.size());
    StoreIndexBuffer(pool, memory, indices, state, gpu_address);
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

VertexAttributeBinding MakeDriverPcoFloat4Binding(
    std::uint32_t offset_bytes, std::uint32_t stride_bytes,
    std::uint16_t destination_register) {
  VertexAttributeBinding binding;
  binding.buffer_index = 0;
  binding.offset_bytes = offset_bytes;
  binding.stride_bytes = stride_bytes;
  binding.destination_register = destination_register;
  binding.component_type = VertexComponentType::kFloat32;
  binding.source_components = 4;
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
                     sc_core::sc_event *sequence_completion, ModelJob *job)
    : sc_module(name), pool_(pool), options_(options), memory_(memory),
      sequence_completion_(sequence_completion), job_(job) {
  SC_THREAD(Run);
}

/*
 * Outer loop for a model that outlives one submission.
 *
 * Without a job slot this is a single pass, exactly as before.  With one, the
 * thread has to survive between flushes: it waits for work, adopts that
 * flush's options, drains them, and returns to the wait.  A failure is handed
 * to the job rather than thrown out of the process, because an exception
 * escaping a SystemC process leaves the kernel with no way to run the next
 * flush -- and the driver, which is still holding a mapped surface, with no
 * way to hear why.
 */
void Submitter::Run() {
  if (!job_) {
    RunJob();
    return;
  }
  for (;;) {
    wait(job_->start);
    if (!job_->running || job_->submitted)
      continue;
    options_ = job_->options;
    fifo_stalls_ = 0;
    try {
      RunJob();
    } catch (const std::exception &error) {
      job_->Fail(error.what());
    }
    job_->submitted = true;
  }
}

void Submitter::RunJob() {
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
        !driver_pco_triangles_command))) {
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
  std::vector<std::pair<const DriverShaderImage *, std::uint64_t>> sequence_image_storage;
  DriverPcoExternalTextureAllocation sequence_external_allocation;
  if (driver_pco_sequence_command &&
      !ResolveSequenceAttachmentAddresses(options_.driver_commands,
                                          &sequence_color_addresses,
                                          &sequence_depth_addresses)) {
    throw std::runtime_error(
        "Submitter sequence attachment dependencies do not fit the address "
        "map");
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
        const std::uint32_t samples = texture.sample_count ? texture.sample_count : 1U;
        const auto &shared = texture.stage == DriverPcoShaderStage::kVertex
                                 ? command.vertex_shared
                                 : texture.stage == DriverPcoShaderStage::kGeometry
                                     ? command.geometry_shared : command.fragment_shared;
        const std::size_t descriptor_end =
            (static_cast<std::size_t>(texture.descriptor_set) + 1U) *
                kPcoTextureDescriptorDwordCount +
                (texture.stage == DriverPcoShaderStage::kGeometry ? 4U : 0U);
        if ((texture.stage != DriverPcoShaderStage::kVertex &&
             texture.stage != DriverPcoShaderStage::kFragment &&
             texture.stage != DriverPcoShaderStage::kGeometry) ||
            (samples != 1U && samples != 2U && samples != 4U && samples != 8U) ||
            descriptor_end > shared.size() ||
            (1U << (shared[descriptor_end - kPcoTextureDescriptorDwordCount + 1U] >> 30U)) != samples ||
            (samples > 1U &&
             (texture.source != DriverPcoTextureSource::kExternalPayload ||
              texture.texture_kind > 1U || texture.mip_count != 1U))) {
          throw std::runtime_error("Submitter PCO texture sample count/descriptor layout mismatch");
        }
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
    state.attachment_layers = command.framebuffer_layers ? command.framebuffer_layers : 1;
    state.layered_framebuffer = command.framebuffer_layers != 0;
    if (state.attachment_layers > 256)
      throw std::runtime_error("Submitter framebuffer layer count is unsupported");
    state.workload_class = WorkloadClass(options_.test_case);
    state.sequence = frame;
    state.functional_case = functional_case;
    state.stage = PipelineStage::kSubmitted;
    state.raster_state.sample_count = command.raster_samples ? command.raster_samples : 1;
    state.raster_state.sample_mask = driver_pco_triangles_command ?
        command.sample_mask : UINT32_MAX;
    if (command.sample_frequency > 1)
      throw std::runtime_error("Submitter sample frequency is invalid");
    state.raster_state.sample_frequency = command.sample_frequency;
    state.raster_state.shader_writes_memory = command.fragment_image_write_mask != 0;
    state.raster_state.shader_early_tests = command.fragment_early_tests;
    state.raster_state.multisample_enable = driver_pco_triangles_command ?
        (command.multisample ? 1 : 0) : 1;
    state.raster_state.alpha_to_coverage = command.alpha_to_coverage;
    state.raster_state.alpha_to_coverage_dither = command.alpha_to_coverage_dither;
    state.raster_state.alpha_to_one = command.alpha_to_one;
    state.memory_mode = options_.memory_mode;
    state.cache_bypass = options_.cache_bypass ? 1U : 0U;
    /*
     * Establish the stored pixel width before resolving sequence LOADs.  An
     * aliased integer attachment can be 8 or 16 bytes per pixel; deriving its
     * readback size from the zero-initialized state made the Submitter fetch
     * only four bytes per pixel, then the PBE (after this field was populated
     * below) correctly rejected the truncated LOAD.
     */
    if (driver_pco_triangles_command) {
      state.color_attachment_raw_dwords =
          ColorAttachmentRawDwords(command.format);
      state.color_attachment_float32 =
          command.format == "PIPE_FORMAT_R32G32B32A32_FLOAT" ? 1U : 0U;
    }
    // Clear-only commands use the same physical packed storage as draws.
    state.color_attachment_packed_unorm = PackedUnormFormatFromName(command.format);

    // Colour attachments this draw writes.  Attachment 0 keeps whatever
    // address the single-target paths already chose; the rest are placed in
    // their own DRAM slots so no two attachments of a pass overlap.
    state.render_target_count =
        command.render_target_count == 0 ? 1U : command.render_target_count;
    if (state.render_target_count > kMaxRenderTargets)
      throw std::runtime_error("Submitter render target count is unsupported");
    for (std::uint32_t target = 1; target < state.render_target_count;
         ++target) {
      const std::uint64_t color_owner = driver_pco_sequence_command ?
          (sequence_color_addresses[submission] - kDriverPcoSequenceColorAddressBase) /
              kDriverPcoSequenceAttachmentStride : submission;
      const std::uint64_t slot =
          color_owner * kMaxRenderTargets + target;
      state.extra_framebuffer_gpu_address[target - 1] =
          kDriverPcoMrtColorAddressBase + slot * kDriverPcoSequenceAttachmentStride;
      /* Every attachment of a pass stores the same pixel width. */
      state.extra_framebuffer_bytes[target - 1] =
          static_cast<std::uint64_t>(state.width) * state.height *
          state.raster_state.sample_count * state.attachment_layers *
          ColorAttachmentBytesPerPixel(state.color_attachment_raw_dwords,
                                       state.color_attachment_float32);
    }
    if (driver_pco_sequence_command) {
      state.framebuffer_gpu_address = sequence_color_addresses[submission];
      const std::uint64_t color_bytes =
          static_cast<std::uint64_t>(state.width) * state.height *
          state.raster_state.sample_count * state.attachment_layers *
          ColorAttachmentBytesPerPixel(state.color_attachment_raw_dwords,
                                       state.color_attachment_float32);
      const std::uint64_t all_color_bytes = color_bytes * state.render_target_count;
      const auto color_address = [&](unsigned target) {
        return target == 0 ? state.framebuffer_gpu_address :
                            state.extra_framebuffer_gpu_address[target - 1];
      };
      if (!command.initial_color_attachment_bytes.empty()) {
        if (command.color_attachment_source_command_index !=
                kDriverPcoNewAttachment ||
            command.initial_color_attachment_bytes.size() != all_color_bytes ||
            color_bytes > kDriverPcoSequenceAttachmentStride) {
          throw std::runtime_error(
              "Submitter initial color attachment contract is invalid");
        }
        // A host snapshot establishes input storage only.  Reading it through
        // the memory system records the dependency and feeds the same PBE
        // LOAD path used by attachments produced by an earlier draw.
        for (unsigned target = 0; target < state.render_target_count; ++target)
          memory_->HostWrite(color_address(target),
              command.initial_color_attachment_bytes.data() + target * color_bytes,
              static_cast<std::size_t>(color_bytes));
      }
      if (command.color_attachment_source_command_index !=
              kDriverPcoNewAttachment ||
          !command.initial_color_attachment_bytes.empty()) {
        if (color_bytes == 0 || color_bytes > kDriverPcoSequenceAttachmentStride ||
            all_color_bytes > std::numeric_limits<std::size_t>::max()) {
          throw std::runtime_error(
              "Submitter aliased color attachment byte size is invalid");
        }
        std::vector<std::uint8_t> all_color_load(static_cast<std::size_t>(all_color_bytes));
        for (unsigned target = 0; target < state.render_target_count; ++target) {
          if (!memory_->backing().Contains(color_address(target), static_cast<std::size_t>(color_bytes)))
            throw std::runtime_error("Submitter aliased color attachment is absent from DRAM");
          MemoryReadResult color_load = memory_->Readback(color_address(target),
              static_cast<std::size_t>(color_bytes), MemoryClient::kFramebufferReadback);
          if (color_load.data.size() != color_bytes)
            throw std::runtime_error("Submitter aliased color attachment readback is truncated");
          sequence_dependency_stats += color_load.stats;
          std::copy(color_load.data.begin(), color_load.data.end(),
                    all_color_load.begin() + target * color_bytes);
        }
        state.color_attachment_load = StoreNewArray(pool_, all_color_load);
        state.color_attachment_load_enable = 1;
        state.color_attachment_load_bytes = all_color_bytes;
      }
      if (command.depth_format != 0) {
        state.depth_attachment_format = command.depth_format;
        state.depth_attachment_gpu_address =
            sequence_depth_addresses[submission];
        const std::size_t depth_bytes_per_pixel =
            DepthAttachmentBytesPerPixel(command.depth_format);
        const std::uint64_t depth_bytes =
            static_cast<std::uint64_t>(state.width) * state.height *
            depth_bytes_per_pixel * state.raster_state.sample_count * state.attachment_layers;
        if (depth_bytes == 0 ||
            depth_bytes > kDriverPcoSequenceAttachmentStride ||
            depth_bytes > std::numeric_limits<std::size_t>::max()) {
          throw std::runtime_error(
              "Submitter depth attachment byte size is invalid");
        }
        if (!command.initial_depth_attachment_bytes.empty()) {
          if (command.depth_attachment_source_command_index != kDriverPcoNewAttachment ||
              command.initial_depth_attachment_bytes.size() != depth_bytes)
            throw std::runtime_error("Submitter initial depth attachment contract is invalid");
          memory_->HostWrite(state.depth_attachment_gpu_address,
              command.initial_depth_attachment_bytes.data(),
              command.initial_depth_attachment_bytes.size());
        }
        if (command.depth_attachment_source_command_index != kDriverPcoNewAttachment ||
            !command.initial_depth_attachment_bytes.empty()) {
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
      if (command.depth_format != 0) {
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
      if (!command.stream_output.bindings.empty()) {
        if (!memory_ || submission >= kDriverSequenceAddressSlots ||
            command.stream_output.bindings.size() > 64 ||
            command.stream_output.targets.size() > 4)
          throw std::runtime_error("Submitter stream output contract is invalid");
        std::vector<StreamOutputBinding> bindings;
        for (const auto &b : command.stream_output.bindings)
          bindings.push_back({b.output_dword, b.num_components, b.output_buffer,
                              b.dst_offset_dwords, b.stream});
        std::vector<StreamOutputTarget> targets;
        for (const auto &source : command.stream_output.targets) {
          if (source.bytes.empty() || source.bytes.size() > kStreamOutputResourceAddressStride)
            throw std::runtime_error("Submitter stream output resource extent is invalid");
          StreamOutputTarget target;
          target.output_buffer = source.output_buffer;
          target.resource_token = source.resource_token;
          target.target_token = source.target_token;
          target.bytes_size = source.bytes.size();
          target.buffer_offset = source.buffer_offset;
          target.buffer_size = source.buffer_size;
          target.internal_offset = source.internal_offset;
          target.stride_dwords = source.stride_dwords;
          const auto alias = std::find_if(targets.begin(), targets.end(), [&](const auto &prior) {
            return prior.resource_token == target.resource_token;
          });
          if (alias != targets.end()) {
            target.gpu_address = alias->gpu_address;
          } else {
            target.gpu_address = kStreamOutputGpuAddressBase +
                (submission * 4U + targets.size()) * kStreamOutputResourceAddressStride;
            memory_->HostWrite(target.gpu_address, source.bytes.data(), source.bytes.size());
          }
          targets.push_back(target);
        }
        state.stream_output_bindings = StoreNewArray(pool_, bindings);
        state.stream_output_targets = StoreNewArray(pool_, targets);
      }
      if (!command.tessellation.control_pco.empty()) {
        const auto &source = command.tessellation;
        const auto count = command.indexed ? command.index_count : command.vertex_count;
        if (!memory_ || !command.geometry_pco.empty() || command.primitive_mode != 14 ||
            !source.input_vertices || source.input_vertices > kTessellationTaskWidth ||
            !source.vertices_per_instance || count % source.vertices_per_instance ||
            count > kTessellationMaxPatches * kTessellationTaskWidth)
          throw std::runtime_error("Submitter tessellation patch input contract is invalid");
        const auto patches_per_instance = source.vertices_per_instance / source.input_vertices;
        const std::uint64_t patch_count = std::uint64_t(patches_per_instance) *
            (count / source.vertices_per_instance);
        if (patch_count > kTessellationMaxPatches)
          throw std::runtime_error("Submitter tessellation patch count exceeds storage bound");
        TessellationState tess;
        tess.control_code = StoreNewArray(pool_, source.control_pco);
        tess.evaluation_code = StoreNewArray(pool_, source.evaluation_pco);
        tess.control_abi = source.control_abi;
        tess.evaluation_abi = source.evaluation_abi;
        state.tessellation_output_dwords = source.evaluation_abi.vertex_outputs;
        tess.input_vertices = source.input_vertices;
        tess.output_vertices = source.output_vertices;
        tess.vertices_per_instance = source.vertices_per_instance;
        tess.input_stride_dwords = source.input_stride_dwords;
        tess.output_vertex_stride_dwords = source.output_vertex_stride_dwords;
        tess.per_vertex_offset_dwords = source.per_vertex_offset_dwords;
        tess.patch_stride_dwords = source.patch_stride_dwords;
        tess.control_barrier_count = source.control_barrier_count;
        tess.domain = static_cast<TessellationDomain>(source.domain);
        tess.spacing = static_cast<TessellationSpacing>(source.spacing);
        tess.clockwise = source.clockwise;
        tess.point_mode = source.point_mode;
        tess.input_address = kTessellationGpuAddressBase + submission * kTessellationDrawAddressStride;
        tess.output_address = tess.input_address + kTessellationPatchAddressStride;
        tess.domain_address = tess.output_address + UINT64_C(0x4000000);
        std::vector<TessellationPatch> patches;
        patches.reserve(patch_count);
        for (std::uint32_t instance = 0; instance < count / source.vertices_per_instance; ++instance) {
          for (std::uint32_t primitive = 0; primitive < patches_per_instance; ++primitive) {
            TessellationPatch patch;
            patch.primitive_id = primitive;
            patch.instance_id = instance;
            patch.first_occurrence = instance * source.vertices_per_instance + primitive * source.input_vertices;
            patch.input_vertices = source.input_vertices;
            patch.output_address = tess.output_address + patches.size() * kTessellationPatchAddressStride;
            patches.push_back(patch);
          }
        }
        tess.patches = StoreNewArray(pool_, patches);
        state.tessellation_state = StoreNewArray(pool_, std::vector<TessellationState>{tess});
      }
      if (!command.geometry_pco.empty()) {
        state.geometry_code = StoreNewArray(pool_, command.geometry_pco);
        state.geometry_pco_abi = command.geometry_pco_abi;
        state.geometry_input_primitive_vertices = command.geometry_input_primitive_vertices;
        state.geometry_output_topology = DriverPcoTopologyFor(command.geometry_output_primitive);
        state.geometry_max_vertices = command.geometry_max_vertices;
        state.geometry_invocations = command.geometry_invocations;
        state.geometry_input_stride_dwords = command.geometry_input_stride_dwords;
        state.geometry_vertices_per_instance = command.geometry_vertices_per_instance;
        state.geometry_input_buffer_gpu_address = kGeometryPrimitiveGpuAddressBase +
            submission * kGeometryPrimitiveGpuAddressStride;
        state.geometry_layer_output_start = command.geometry_layer_output_start;
        state.geometry_layer_output_count = command.geometry_layer_output_count;
        state.geometry_primitive_id_output_start = command.geometry_primitive_id_output_start;
        state.geometry_primitive_id_output_count = command.geometry_primitive_id_output_count;
        state.geometry_input_primitives = StoreNewArray(pool_, GeometryInputsFor(command));
      }
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
      state.driver_varying_bindings_explicit = command.explicit_varying_bindings;
      state.driver_varying_binding_count = command.varying_bindings.size();
      for (std::size_t target = 0;
           target < state.fragment_output_mask.size(); ++target) {
        state.fragment_output_mask[target] =
            command.fragment_output_mask[target];
      }
      /*
       * How many raw 32-bit channels the colour attachment stores, from the
       * format the driver stated.  dEQP's shader executor renders a scalar
       * result into R32_UINT, a vec2 into RG32UI and a vec3 or vec4 into
       * RGBA32UI -- GLES has no three-channel integer target -- so this is
       * also what widens the framebuffer past four bytes a pixel.
       */
      state.color_attachment_raw_dwords =
          ColorAttachmentRawDwords(command.format);
      // An sRGB-encoded eight-bit colour target: same byte layout as RGBA8, but
      // the PBE applies the sRGB transfer on write and blends in linear space.
      state.color_is_srgb =
          (command.format == "PIPE_FORMAT_R8G8B8A8_SRGB" ||
           command.format == "PIPE_FORMAT_B8G8R8A8_SRGB")
              ? 1U
              : 0U;
      state.vertex_sampled_texture_count =
          command.vertex_sampled_texture_count;
      state.geometry_sampled_texture_count = command.geometry_sampled_texture_count;
      state.sampled_texture_count =
          command.fragment_sampled_texture_count != 0 ||
                  !command.sampled_textures.empty()
              ? command.fragment_sampled_texture_count
              : command.sampled_texture_count;
    }
    state.raster_state.sample_count = command.raster_samples ? command.raster_samples : 1;
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
    const bool driver_clear_like = driver_clear;
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
      state.raster_state.stencil.test_enable =
          static_cast<std::uint8_t>(command.stencil_enable);
      state.raster_state.stencil.clear_stencil = command.stencil_clear;
      if (!command.attachment_clears.empty()) {
        std::vector<AttachmentClearRect> clears;
        clears.reserve(command.attachment_clears.size());
        for (const DriverAttachmentClear &clear : command.attachment_clears) {
          AttachmentClearRect rect;
          rect.x = clear.x;
          rect.y = clear.y;
          rect.width = clear.width;
          rect.height = clear.height;
          rect.aspects = clear.aspects;
          rect.depth_bits = clear.depth_bits;
          rect.stencil_value = clear.stencil_value;
          clears.push_back(rect);
        }
        state.attachment_clears = StoreNewArray(pool_, clears);
      }
      {
        StencilFaceState *faces[2] = {&state.raster_state.stencil.front,
                                      &state.raster_state.stencil.back};
        for (std::size_t face = 0; face < 2; ++face) {
          faces[face]->compare_op =
              static_cast<DepthCompareOp>(command.stencil_func[face]);
          faces[face]->fail_op =
              static_cast<StencilOp>(command.stencil_fail_op[face]);
          faces[face]->depth_fail_op =
              static_cast<StencilOp>(command.stencil_depth_fail_op[face]);
          faces[face]->pass_op =
              static_cast<StencilOp>(command.stencil_pass_op[face]);
          faces[face]->value_mask = command.stencil_value_mask[face];
          faces[face]->write_mask = command.stencil_write_mask[face];
          faces[face]->reference = command.stencil_ref[face];
        }
      }
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
      for (std::size_t channel = 0; channel < 4; ++channel) {
        state.raster_state.blend.constant_color_bits[channel] =
            command.blend_constant_color_bits[channel];
      }
      state.raster_state.face_cull.enable = command.cull_face == 0 ? 0U : 1U;
      state.raster_state.face_cull.mode =
          command.cull_face == 1
              ? CullFaceMode::kFront
              : command.cull_face == 3 ? CullFaceMode::kFrontAndBack
                                       : CullFaceMode::kBack;
      state.raster_state.face_cull.front_face =
          FrontFaceFromDriverCommand(command.front_ccw,
                                     command.viewport_scale_bits[1]);
      state.raster_state.bottom_edge_rule =
          static_cast<std::uint8_t>(command.bottom_edge_rule);
      state.raster_state.color_mask =
          static_cast<std::uint8_t>(command.color_mask);
      state.raster_state.scissor.enable =
          command.scissor == 0 ? 0U : 1U;
      state.raster_state.scissor.x0 = command.scissor_x;
      state.raster_state.scissor.y0 = command.scissor_y;
      state.raster_state.scissor.x1 = command.scissor_x + command.scissor_width;
      state.raster_state.scissor.y1 =
          command.scissor_y + command.scissor_height;
      // An unstated width is the GLES default of one device pixel.
      state.raster_state.line_width =
          command.line_width_bits == 0
              ? 1.0F
              : FloatFromBits(command.line_width_bits);
      state.raster_state.point_size =
          command.point_size_bits == 0
              ? 1.0F
              : FloatFromBits(command.point_size_bits);
      state.raster_state.point_size_output_start =
          command.point_size_output_start;
      state.raster_state.point_size_output_count =
          command.point_size_output_count;
      // The viewport transform the draw states.  Clip/cull falls back to the
      // whole attachment when the scale is unstated.
      for (std::size_t axis = 0; axis < 3; ++axis) {
        state.raster_state.viewport_scale[axis] =
            FloatFromBits(command.viewport_scale_bits[axis]);
        state.raster_state.viewport_translate[axis] =
            FloatFromBits(command.viewport_translate_bits[axis]);
      }
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
        driver_pco_triangles && command.geometry_pco.empty() && command.tessellation.control_pco.empty()
            ? ExpandDriverPcoTopology(command)
            : DriverPcoTopologyExpansion{};
    const std::vector<std::uint8_t> &expanded_pco_vertices =
        command.geometry_pco.empty() && command.tessellation.control_pco.empty()
            ? expanded_pco.vertices : command.raw_vertex_data;
    if (driver_triangle) {
      vertex_buffer = DriverTriangleFloat2Vertices(command);
      const std::vector<std::uint16_t> indices = {0, 1, 2};
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_index = 0;
      state.draw.index_count = static_cast<std::uint32_t>(indices.size());
      state.draw.base_vertex = 0;
      state.draw.index_format = IndexFormat::kUint16;
      StoreIndexBuffer(pool_, memory_, indices, &state,
                       SequenceBufferAddress(
                           kBuiltinIndexBufferGpuAddress, submission,
                           "index"));
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
      StoreIndexBuffer(pool_, memory_, indices, &state,
                       SequenceBufferAddress(
                           kBuiltinIndexBufferGpuAddress, submission,
                           "index"));
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
      StoreIndexBuffer(pool_, memory_, indices, &state,
                       SequenceBufferAddress(
                           kBuiltinIndexBufferGpuAddress, submission,
                           "index"));
    } else if (driver_pco_triangles && command.indexed != 0) {
      // Hand the real index buffer to vertex fetch: it walks the indices,
      // expands the topology and reuses post-transform vertices, so the
      // reported vs_invocations reflect actual shading work.
      state.draw.topology = DriverPcoTopologyFor(command.primitive_mode);
      state.draw.first_vertex = 0;
      state.draw.vertex_count = command.vertex_count;
      state.draw.first_index = command.first_index;
      state.draw.index_count = command.index_count;
      state.draw.base_vertex = command.base_vertex;
      state.draw.index_format = DriverPcoIndexFormatFor(command.index_size);
      StoreDriverPcoIndexBuffer(pool_, memory_, command, &state,
                                SequenceBufferAddress(
                                    kBuiltinIndexBufferGpuAddress,
                                    submission, "index"));
    } else if (driver_pco_triangles) {
      state.draw.topology = command.geometry_pco.empty() && command.tessellation.control_pco.empty()
          ? PrimitiveTopology::kTriangleList : DriverPcoTopologyFor(command.primitive_mode);
      state.draw.first_vertex = 0;
      state.draw.vertex_count = command.geometry_pco.empty() && command.tessellation.control_pco.empty()
          ? static_cast<std::uint32_t>(expanded_pco_vertices.size() / command.vertex_stride)
          : command.vertex_count;
      state.draw.index_format = IndexFormat::kNone;
      if (!expanded_pco.source_vertices.empty()) {
        if (expanded_pco.source_vertices.size() != state.draw.vertex_count)
          throw std::runtime_error(
              "Submitter PCO source vertex map does not cover the draw");
        state.expanded_source_vertices =
            StoreNewArray(pool_, expanded_pco.source_vertices);
      }
      // The expansion above already encoded lines and points as degenerate
      // triangles, so record what was submitted for ClipCull to widen.
      state.source_topology = DriverPcoTopologyFor(command.primitive_mode);
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
    const bool empty_geometry_attributes = driver_pco_triangles &&
        (!command.geometry_pco.empty() || !command.tessellation.control_pco.empty()) && command.vertex_attribute_count == 0 &&
        command.vertex_pco_abi.vertex_inputs == 0 && command.vertex_stride == 0 &&
        expanded_pco_vertices.empty();
    std::vector<VertexBufferResource> vertex_resources;
    if (!empty_geometry_attributes) {
      const VertexBufferResource vertex_resource =
        driver_pco_triangles
            ? StoreRawVertexBuffer(pool_, expanded_pco_vertices,
                                   SequenceBufferAddress(
                                       kBuiltinVertexBufferGpuAddress,
                                       submission, "vertex"),
                                   memory_)
            : StoreFloat2VertexBuffer(pool_, vertex_buffer,
                                      kBuiltinVertexBufferGpuAddress, memory_);
      vertex_resources.push_back(vertex_resource);
    }
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
        const bool color_layout =
            command.vertex_stride == 6U * sizeof(float) &&
            command.vertex_pco_abi.vertex_inputs == 8 &&
            command.varying_output_count == 4 &&
            command.fragment_varying_count == 16 &&
            command.vertex_pco_abi.shareds == 0;
        // The same layout with a full-width vec4 position.
        const bool color4_layout =
            command.vertex_stride == 8U * sizeof(float) &&
            command.vertex_pco_abi.vertex_inputs == 8 &&
            command.varying_output_count == 4 &&
            command.fragment_varying_count == 16;
        const bool terrain_main_layout =
            command.vertex_stride == 11U * sizeof(float) &&
            command.vertex_pco_abi.vertex_inputs == 16;
        if (empty_geometry_attributes) {
          // No fabricated VBO or attribute: the shader has no VTXIN reads.
          state.driver_describes_attributes = 1;
        } else if (command.vertex_attribute_count != 0) {
          // The capsule states each attribute's width, so build the bindings
          // it describes rather than inferring a layout from the stride.
          state.driver_describes_attributes = 1;
          std::uint32_t offset_bytes = 0;
          for (std::uint32_t attribute = 0;
               attribute < command.vertex_attribute_count; ++attribute) {
            const std::uint32_t components =
                command.vertex_attribute_components[attribute];
            if (components == 0 || components > 4) {
              throw std::runtime_error(
                  "Submitter driver PCO attribute width is invalid");
            }
            VertexAttributeBinding binding;
            binding.buffer_index = 0;
            binding.offset_bytes = offset_bytes;
            binding.stride_bytes = command.vertex_stride;
            binding.destination_register =
                static_cast<std::uint16_t>(attribute * 4U);
            /*
             * The stream carries the attribute's own bytes and the shader's
             * own unpack decodes them, so every word here is opaque: fetch
             * copies it verbatim.  Reading it as a float and writing the bits
             * back would be exact only by accident -- a packed pair of bytes
             * is a denormal, and four that happen to set the exponent are a
             * NaN, neither of which survives a round trip through `float`.
             */
            binding.component_type = VertexComponentType::kUint32;
            binding.source_components =
                static_cast<std::uint8_t>(components);
            binding.destination_components = 4;
            binding.normalized = 0;
            binding.integer = 1U;
            binding.instance_divisor = 0;
            bindings.push_back(binding);
            offset_bytes += components * sizeof(float);
          }
          if (offset_bytes != command.vertex_stride) {
            throw std::runtime_error(
                "Submitter driver PCO attribute widths do not fill the stride");
          }
        } else if (color_layout) {
          bindings = {
              MakeDriverPcoFloat2Binding(0, command.vertex_stride, 0),
              MakeDriverPcoFloat4Binding(
                  2U * sizeof(float), command.vertex_stride, 4),
          };
        } else if (color4_layout) {
          bindings = {
              MakeDriverPcoFloat4Binding(0, command.vertex_stride, 0),
              MakeDriverPcoFloat4Binding(
                  4U * sizeof(float), command.vertex_stride, 4),
          };
        } else if (terrain_main_layout) {
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
      if (varying_count == 0 &&
          VaryingCoefficientDwordCount(state) != kCoefficientSetDwordCount)
        throw std::runtime_error("Submitter varying count is invalid");
      std::vector<ShaderVaryingBinding> linkages;
      linkages.reserve(varying_count);
      for (std::uint32_t varying = 0; varying < varying_count; ++varying) {
        ShaderVaryingBinding linkage;
        if (driver_pco_triangles && command.explicit_varying_bindings) {
          const auto &b = command.varying_bindings.at(varying);
          linkage.vertex_output_base = static_cast<std::uint16_t>(b.output_dword);
          linkage.coefficient_set_base = static_cast<std::uint16_t>(b.coefficient_dword / 4);
          linkage.w_coefficient_set = 0;
          linkage.component_count = static_cast<std::uint8_t>(b.num_components);
        } else if (driver_pco_triangles) {
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
        // A flat varying is not interpolated: its coefficient set carries the
        // provoking vertex's value.  The capsule states which are flat because
        // the model cannot tell from the linkage alone, and assuming smooth
        // made a flat integer read back as the plane's first term.
        const bool flat = command.explicit_varying_bindings
            ? command.varying_bindings.at(varying).flat != 0
            : (command.varying_flat_mask & (1U << varying)) != 0;
        linkage.interpolation = flat
                ? InterpolationMode::kFlat
                : InterpolationMode::kSmooth;
        const char *linkage_refusal = nullptr;
        if (!IsExactVaryingBinding(state, linkage, varying,
                                   &linkage_refusal)) {
          throw std::runtime_error(
              "Submitter varying linkage is invalid: field=" +
              std::string(linkage_refusal ? linkage_refusal : "unknown") +
              " varying=" + std::to_string(varying));
        }
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
      std::vector<std::uint32_t> geometry_shared = command.geometry_shared;
      std::vector<std::uint32_t> control_shared = command.tessellation.control_shared;
      std::vector<std::uint32_t> evaluation_shared = command.tessellation.evaluation_shared;
      std::vector<UniformBufferResource> control_uniform_buffers;
      std::vector<UniformBufferResource> evaluation_uniform_buffers;
      std::string uniform_error;
      if (!ValidateDriverUniformBuffers(command, &uniform_error) ||
          !ValidateDriverShaderImages(command, &uniform_error))
        throw std::runtime_error(uniform_error);
      state.fragment_image_descriptor_start = command.fragment_image_descriptor_start;
      state.fragment_image_descriptor_count = command.fragment_image_descriptor_count;
      state.fragment_image_read_mask = command.fragment_image_read_mask;
      state.fragment_image_write_mask = command.fragment_image_write_mask;
      if (!command.fragment_images.empty()) {
        if (!driver_pco_sequence_command || !memory_)
          throw std::runtime_error("Submitter fragment images require sequence GPU memory");
        std::vector<ShaderImageResource> resources;
        for (const auto &image : command.fragment_images) {
          ShaderImageResource resource;
          resource.resource_token = image.resource_token;
          resource.bytes = image.bytes.size(); resource.offset = image.offset;
          resource.image_slot = image.image_slot; resource.format = image.format;
          resource.access = image.access; resource.width = image.width;
          resource.height = image.height; resource.depth = image.depth;
          resource.row_stride = image.row_stride; resource.layer_stride = image.layer_stride;
          resource.texel_bytes = image.texel_bytes;
          const auto alias = std::find_if(sequence_image_storage.begin(), sequence_image_storage.end(),
              [&](const auto &prior) { return prior.first->resource_token == image.resource_token; });
          if (alias != sequence_image_storage.end()) {
            if (alias->first->bytes != image.bytes)
              throw std::runtime_error("Submitter image resource changed without a sequence boundary");
            resource.gpu_address = alias->second;
          } else {
            if (sequence_image_storage.size() >= kDriverSequenceAddressSlots * kMaximumFragmentImages)
              throw std::runtime_error("Submitter fragment image address slots exhausted");
            resource.gpu_address = kShaderImageGpuAddressBase +
                sequence_image_storage.size() * kMaximumFragmentImageBytes;
            memory_->HostWrite(resource.gpu_address, image.bytes.data(), image.bytes.size());
            sequence_image_storage.emplace_back(&image, resource.gpu_address);
          }
          resource.readback = StoreNewArray(pool_, image.bytes);
          const auto address = resource.gpu_address + resource.offset;
          const auto word = command.fragment_image_descriptor_start + image.image_slot * 8U;
          fragment_shared.at(word) = static_cast<std::uint32_t>(address);
          fragment_shared.at(word + 1) = static_cast<std::uint32_t>(address >> 32);
          resources.push_back(resource);
        }
        state.fragment_image_resources = StoreNewArray(pool_, resources);
      }
      if (!command.uniform_buffers.empty()) {
        if (!driver_pco_sequence_command || !memory_)
          throw std::runtime_error("Submitter uniform buffers require sequence GPU memory");
        std::vector<UniformBufferResource> vertex_uniform_buffers;
        std::vector<UniformBufferResource> fragment_uniform_buffers;
        std::vector<UniformBufferResource> geometry_uniform_buffers;
        for (const auto &buffer : command.uniform_buffers) {
          const bool vertex_stage = buffer.stage == DriverPcoShaderStage::kVertex;
          const bool geometry_stage = buffer.stage == DriverPcoShaderStage::kGeometry;
          const bool control_stage = buffer.stage == DriverPcoShaderStage::kTessellationControl;
          const bool evaluation_stage = buffer.stage == DriverPcoShaderStage::kTessellationEvaluation;
          const auto &abi = control_stage ? command.tessellation.control_abi
                           : evaluation_stage ? command.tessellation.evaluation_abi
                           : geometry_stage ? command.geometry_pco_abi
                           : vertex_stage ? command.vertex_pco_abi
                                         : command.fragment_pco_abi;
          auto &shared = control_stage ? control_shared : evaluation_stage ? evaluation_shared
                         : geometry_stage ? geometry_shared
                         : vertex_stage ? vertex_shared_words : fragment_shared;
          UniformBufferResource resource;
          resource.gpu_address = SequenceUniformBufferAddress(
              submission, buffer.stage, buffer.block_index);
          resource.bytes = buffer.bytes.size();
          resource.block_index = buffer.block_index;
          resource.descriptor_shared_start = abi.uniform_buffer_descriptor_start +
              buffer.block_index * kUniformBufferDescriptorDwordCount;
          HostWriteArray(*memory_, resource.gpu_address, buffer.bytes);
          const std::size_t word = resource.descriptor_shared_start;
          shared[word] = static_cast<std::uint32_t>(resource.gpu_address);
          shared[word + 1] = static_cast<std::uint32_t>(resource.gpu_address >> 32U);
          // Size and zero dynamic offset were validated against the snapshot.
          (control_stage ? control_uniform_buffers : evaluation_stage ? evaluation_uniform_buffers
           : geometry_stage ? geometry_uniform_buffers
           : vertex_stage ? vertex_uniform_buffers : fragment_uniform_buffers)
              .push_back(resource);
        }
        if (!vertex_uniform_buffers.empty())
          state.vertex_uniform_buffer_resources = StoreNewArray(pool_, vertex_uniform_buffers);
        if (!fragment_uniform_buffers.empty())
          state.fragment_uniform_buffer_resources = StoreNewArray(pool_, fragment_uniform_buffers);
        if (!geometry_uniform_buffers.empty())
          state.geometry_uniform_buffer_resources = StoreNewArray(pool_, geometry_uniform_buffers);
      }
      if (HasPoolHandle(state.tessellation_state)) {
        auto tess = LoadArray<TessellationState>(pool_, state.tessellation_state);
        tess[0].control_shared = StoreNewArray(pool_, control_shared);
        tess[0].evaluation_shared = StoreNewArray(pool_, evaluation_shared);
        if (!control_uniform_buffers.empty())
          tess[0].control_uniform_buffers = StoreNewArray(pool_, control_uniform_buffers);
        if (!evaluation_uniform_buffers.empty())
          tess[0].evaluation_uniform_buffers = StoreNewArray(pool_, evaluation_uniform_buffers);
        StoreArray(pool_, state.tessellation_state, tess);
      }
      if (!command.sampled_textures.empty()) {
        if (!driver_pco_sequence_command || !memory_ ||
            command.sampled_textures.size() !=
                command.sampled_texture_count ||
            command.vertex_sampled_texture_count +
                    command.fragment_sampled_texture_count + command.geometry_sampled_texture_count !=
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
        std::vector<TextureResource> geometry_resources(command.geometry_sampled_texture_count);
        std::vector<SamplerState> geometry_samplers(command.geometry_sampled_texture_count);
        std::vector<bool> geometry_present(command.geometry_sampled_texture_count, false);
        std::vector<bool> vertex_present(command.vertex_sampled_texture_count,
                                         false);
        std::vector<bool> fragment_present(
            command.fragment_sampled_texture_count, false);
        for (const DriverPcoSampledTexture &texture :
             command.sampled_textures) {
          const bool vertex_stage =
              texture.stage == DriverPcoShaderStage::kVertex;
          const bool geometry_stage = texture.stage == DriverPcoShaderStage::kGeometry;
          auto &resources =
              vertex_stage ? vertex_resources : geometry_stage ? geometry_resources : fragment_resources;
          auto &samplers =
              vertex_stage ? vertex_samplers : geometry_stage ? geometry_samplers : fragment_samplers;
          auto &present =
              vertex_stage ? vertex_present : geometry_stage ? geometry_present : fragment_present;
          if (texture.declared_bytes_size == 0 ||
              texture.declared_bytes_size >
                  std::numeric_limits<std::uint32_t>::max() ||
              (texture.source != DriverPcoTextureSource::kExternalPayload &&
               texture.declared_bytes_size > kDriverPcoSequenceAttachmentStride) ||
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
                submission, texture.stage, texture.descriptor_set,
                texture.declared_bytes_size, &sequence_external_allocation);
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
              vertex_stage ? &vertex_shared_words : geometry_stage ? &geometry_shared : &fragment_shared,
              texture.descriptor_set, gpu_address, geometry_stage ? 4U : 0U);

          TextureResource resource;
          resource.gpu_address = gpu_address;
          resource.byte_size =
              static_cast<std::uint32_t>(texture.declared_bytes_size);
          resource.mip_count = static_cast<std::uint8_t>(texture.mip_count);
          resource.sample_count = static_cast<std::uint8_t>(
              texture.sample_count ? texture.sample_count : 1U);
          resource.layer_count =
              static_cast<std::uint16_t>(texture.layers == 0U ? 1U
                                                              : texture.layers);
          resource.dimension_type =
              texture.texture_kind == 1U
                  ? TextureDimensionType::k2DArray
              : texture.texture_kind == 2U
                  ? TextureDimensionType::k3D
              : texture.texture_kind == 3U
                  ? TextureDimensionType::kCube
                  : TextureDimensionType::k2D;
          resource.format =
              texture.format == "PIPE_FORMAT_R32G32B32A32_UINT"
                  ? TextureFormat::kRgba32Uint
              : texture.format == "PIPE_FORMAT_R32G32B32A32_SINT"
                  ? TextureFormat::kRgba32Sint
              : texture.format == "PIPE_FORMAT_R32G32B32A32_FLOAT"
                  ? TextureFormat::kRgba32Float
              : texture.format == "PIPE_FORMAT_Z32_UNORM"
                  ? TextureFormat::kZ32Unorm
                  : texture.format == "PIPE_FORMAT_Z24_UNORM_S8_UINT"
                        ? TextureFormat::kZ24UnormS8Uint
                  : texture.format == "PIPE_FORMAT_R8G8B8A8_UNORM"
                        ? TextureFormat::kRgba8Unorm
                        // B8G8R8A8 storage; the descriptor swizzle presents it
                        // as RGBA and the texture unit swaps red/blue at fetch.
                        : texture.format == "PIPE_FORMAT_B8G8R8A8_UNORM"
                              ? TextureFormat::kBgra8Unorm
                        : texture.format == "PIPE_FORMAT_R8G8B8X8_UNORM"
                              ? TextureFormat::kRgbx8Unorm
                        // Same four stored bytes as RGBA8; the texture unit
                        // decodes R, G and B through the sRGB curve.
                        : texture.format == "PIPE_FORMAT_R8G8B8A8_SRGB"
                              ? TextureFormat::kRgba8Srgb
                        // Packed / wide colour: the texture unit unpacks the
                        // little-endian bit layout on its float datapath.
                        : (texture.format == "PIPE_FORMAT_R5G6B5_UNORM" ||
                           texture.format == "PIPE_FORMAT_B5G6R5_UNORM")
                              ? TextureFormat::kRgb565Unorm
                        : texture.format == "PIPE_FORMAT_R10G10B10A2_UNORM"
                              ? TextureFormat::kRgb10A2Unorm
                        : texture.format == "PIPE_FORMAT_B10G10R10A2_UNORM"
                              ? TextureFormat::kBgr10A2Unorm
                        : texture.format == "PIPE_FORMAT_R8G8B8A8_SNORM"
                              ? TextureFormat::kRgba8Snorm
                        : texture.format == "PIPE_FORMAT_R16G16B16A16_FLOAT"
                              ? TextureFormat::kRgba16Float
                        : texture.format == "PIPE_FORMAT_R11G11B10_FLOAT"
                              ? TextureFormat::kR11fG11fB10f
                        : texture.format == "PIPE_FORMAT_R9G9B9E5_FLOAT"
                              ? TextureFormat::kRgb9e5Float
                        // ASTC arrives compressed: the texture unit decodes
                        // the blocks, which is where the hardware does it.
                        : texture.format.rfind("PIPE_FORMAT_ASTC_", 0) == 0
                              ? (texture.format.size() > 5 &&
                                 texture.format.compare(
                                     texture.format.size() - 5, 5,
                                     "_SRGB") == 0
                                     ? TextureFormat::kAstcLdrSrgb
                                     : TextureFormat::kAstcLdr)
                        : throw std::runtime_error(
                              "Submitter PCO sequence texture format is "
                              "unsupported: " + texture.format);
          if (resource.format == TextureFormat::kAstcLdr ||
              resource.format == TextureFormat::kAstcLdrSrgb) {
            // The footprint the format names, from the same fourteen Rogue
            // TEXSTATE declares.  An ASTC name outside them is not a
            // footprint this model has, and saying so beats guessing 4x4.
            struct AstcFootprintName {
              const char *format;
              std::uint8_t width;
              std::uint8_t height;
            };
            static constexpr AstcFootprintName kFootprints[] = {
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
            bool found = false;
            for (const AstcFootprintName &entry : kFootprints) {
              if (texture.format == entry.format) {
                resource.block_width = entry.width;
                resource.block_height = entry.height;
                found = true;
                break;
              }
            }
            if (!found) {
              throw std::runtime_error(
                  "Submitter PCO sequence ASTC footprint is unsupported: " +
                  texture.format);
            }
          }
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
          if (resource.sample_count > 1U) {
            if (resource.block_width != 1U || resource.block_height != 1U ||
                resource.format == TextureFormat::kAstcLdr ||
                resource.format == TextureFormat::kAstcLdrSrgb ||
                texture.layers > UINT16_MAX || !resource.layer_count ||
                !resource.mip[0].width || !resource.mip[0].height ||
                resource.mip[0].offset_bytes != 0 ||
                resource.mip[0].row_pitch_bytes <
                    static_cast<std::uint64_t>(resource.mip[0].width) *
                        TextureBytesPerTexel(resource.format) * resource.sample_count ||
                resource.mip[0].row_pitch_bytes >
                    resource.byte_size / resource.mip[0].height / resource.layer_count ||
                static_cast<std::uint64_t>(resource.mip[0].row_pitch_bytes) *
                    resource.mip[0].height * resource.layer_count != resource.byte_size) {
              throw std::runtime_error("Submitter PCO multisample texture byte layout is invalid");
            }
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
          // Capsule address modes: 0 clamp to edge, 1 repeat, 2 mirrored.
          const auto capsule_wrap = [](std::uint32_t value) {
            return value == 1U   ? TextureWrapMode::kRepeat
                   : value == 2U ? TextureWrapMode::kMirroredRepeat
                                 : TextureWrapMode::kClampToEdge;
          };
          sampler.wrap_u = capsule_wrap(texture.wrap_u);
          sampler.wrap_v = capsule_wrap(texture.wrap_v);
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
            std::find(geometry_present.begin(), geometry_present.end(), false) != geometry_present.end() ||
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
        if (!geometry_resources.empty()) {
          state.geometry_texture_resources = StoreNewArray(pool_, geometry_resources);
          state.geometry_sampler_states = StoreNewArray(pool_, geometry_samplers);
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
      if (!geometry_shared.empty())
        state.geometry_shared_registers = StoreNewArray(pool_, geometry_shared);
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
    if (!driver_pco_triangles) {
      // The built-in fragment programs above export one complete vec4 to
      // target zero. Declare their ABI just as the driver does; a zero mask
      // in a real driver command still means that target is not written.
      state.fragment_output_mask[0] = 0xf;
    }
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
