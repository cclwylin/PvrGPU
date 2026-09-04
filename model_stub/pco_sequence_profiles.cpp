#include "pco_sequence_profiles.h"

#include "model_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace pvrgpu::stub {
namespace {

constexpr char kDriverCommandSchema[] = "pvrgpu.driver-command.v1";
constexpr char kDriverCommandProducer[] = "pvrgpu-gallium-driver";
constexpr char kDrawPcoTriangles[] = "draw_pco_triangles";
constexpr char kDrawPcoSequence[] = "draw_pco_sequence";
constexpr char kRgba8[] = "PIPE_FORMAT_R8G8B8A8_UNORM";
constexpr char kRgbx8[] = "PIPE_FORMAT_R8G8B8X8_UNORM";
constexpr char kZ32[] = "PIPE_FORMAT_Z32_UNORM";
constexpr char kRefractCase[] = "refract.refract.capture.1";
constexpr char kShadowCase[] = "shadow.shadow.capture.1";
constexpr char kTerrainCase[] = "terrain.terrain.capture.1";

constexpr std::uint32_t kTriangles = 4;
constexpr std::uint32_t kTriangleStrip = 5;
constexpr std::uint32_t kCullFront = 1;
constexpr std::uint32_t kCullBack = 2;
constexpr std::uint32_t kBlendAdd = 0;
constexpr std::uint32_t kBlendZero = 0;
constexpr std::uint32_t kBlendOne = 1;
constexpr std::uint32_t kBlendSourceAlpha = 2;
constexpr std::uint32_t kDepthLequal = 3;
constexpr std::uint32_t kFilterNearest = 0;
constexpr std::uint32_t kFilterLinear = 1;
constexpr std::uint32_t kMipNone = 0;
constexpr std::uint32_t kMipLinear = 1;
constexpr std::uint32_t kWrapClamp = 0;
constexpr std::uint32_t kWrapRepeat = 1;
constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

constexpr std::array<std::uint32_t, 4> kOpaqueBlack = {
    0, 0, 0, UINT32_C(0x3f800000)};
constexpr std::array<std::uint32_t, 4> kTerrainClear = {
    UINT32_C(0x3f533333), UINT32_C(0x3f3e147b),
    UINT32_C(0x3f1e6666), UINT32_C(0x3f800000)};

constexpr std::uint32_t FullMipCount(std::uint32_t width,
                                     std::uint32_t height) {
  std::uint32_t count = 0;
  while (width != 0 || height != 0) {
    ++count;
    width >>= 1U;
    height >>= 1U;
  }
  return count;
}

constexpr std::uint64_t TightRgba8MipBytes(std::uint32_t width,
                                           std::uint32_t height,
                                           std::uint32_t mip_count) {
  std::uint64_t bytes = 0;
  for (std::uint32_t level = 0; level < mip_count; ++level) {
    bytes += static_cast<std::uint64_t>(width) * height * 4U;
    width = width > 1U ? width >> 1U : 1U;
    height = height > 1U ? height >> 1U : 1U;
  }
  return bytes;
}

struct SequenceResolutionProfile {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t offscreen_width;
  std::uint32_t offscreen_height;
  std::array<std::uint32_t, 3> viewport;
  std::array<std::uint32_t, 3> offscreen_viewport;
  std::uint32_t output_mip_count;
  std::uint32_t offscreen_mip_count;
  std::uint64_t output_rgba8_bytes;
  std::uint64_t offscreen_rgba8_bytes;
  std::uint64_t offscreen_depth_bytes;
  std::uint32_t output_max_lod_u4_6;
  std::uint32_t offscreen_max_lod_u4_6;
  std::uint64_t refract_fragment_shared_fnv;
  std::uint64_t shadow_fragment_shared_fnv;
  std::uint64_t terrain_output_fragment_shared_fnv;
  std::array<std::uint64_t, 8> terrain_fragment_pco_fnv;
  std::uint32_t refract_drawlists;
  std::uint64_t refract_ps_invocations;
  std::uint64_t refract_texel_fetches;
  std::uint32_t shadow_drawlists;
  std::uint64_t shadow_ps_invocations;
  std::uint64_t shadow_texel_fetches;
  std::uint32_t terrain_drawlists;
  std::uint64_t terrain_ps_invocations;
  std::uint64_t terrain_texel_fetches;
};

constexpr std::array<SequenceResolutionProfile, 2>
    kSequenceResolutionProfiles = {{
        {80,
         60,
         160,
         120,
         {UINT32_C(0x42200000), UINT32_C(0x41f00000),
          UINT32_C(0x3f000000)},
         {UINT32_C(0x42a00000), UINT32_C(0x42700000),
          UINT32_C(0x3f000000)},
         FullMipCount(80, 60),
         FullMipCount(160, 120),
         TightRgba8MipBytes(80, 60, FullMipCount(80, 60)),
         TightRgba8MipBytes(160, 120, FullMipCount(160, 120)),
         UINT64_C(160) * 120U * 4U,
         (FullMipCount(80, 60) - 1U) * 64U,
         (FullMipCount(160, 120) - 1U) * 64U,
         UINT64_C(0x26536c76cbc158b5),
         UINT64_C(0x5d306f7625b3e88e),
         UINT64_C(0x2d423f9c5838f4fd),
         {UINT64_C(0x9e1c3ea2dfa1d8a5), UINT64_C(0x9711b79a7b5b63a6),
          UINT64_C(0x4fecdd1ce1feb997), UINT64_C(0x956d5ea59737b66f),
          UINT64_C(0x76fac56a9fbc5918), UINT64_C(0x412b9e844c3de073),
          UINT64_C(0xab0dfc14e6aa5116), UINT64_C(0xd0b9eb8de7e641d2)},
         9,
         15675,
         130272,
         3,
         3463,
         2104,
         42,
         329330,
         5413856},
        {800,
         600,
         1600,
         1200,
         {UINT32_C(0x43c80000), UINT32_C(0x43960000),
          UINT32_C(0x3f000000)},
         {UINT32_C(0x44480000), UINT32_C(0x44160000),
          UINT32_C(0x3f000000)},
         FullMipCount(800, 600),
         FullMipCount(1600, 1200),
         TightRgba8MipBytes(800, 600, FullMipCount(800, 600)),
         TightRgba8MipBytes(1600, 1200, FullMipCount(1600, 1200)),
         UINT64_C(1600) * 1200U * 4U,
         (FullMipCount(800, 600) - 1U) * 64U,
         (FullMipCount(1600, 1200) - 1U) * 64U,
         UINT64_C(0x440e2cf4d71a3f5c),
         UINT64_C(0x77d13d3fd210cd96),
         UINT64_C(0x33d7c2aad6bb3b8a),
         {UINT64_C(0x9e1c3ea2dfa1d8a5), UINT64_C(0x9711b79a7b5b63a6),
          UINT64_C(0x4fecdd1ce1feb997), UINT64_C(0x6ad4537c64c80942),
          UINT64_C(0x76fac56a9fbc5918), UINT64_C(0x412b9e844c3de073),
          UINT64_C(0x1d6737c7f69c0953), UINT64_C(0xb41e711d1ef41b5a)},
         12,
         1568167,
         8261280,
         3,
         348735,
         168960,
         51,
         2658615,
         91927520},
    }};

static_assert(FullMipCount(1600, 1200) == 11);
static_assert(TightRgba8MipBytes(1600, 1200, 11) == UINT64_C(10239756));
static_assert(FullMipCount(800, 600) == 10);
static_assert(TightRgba8MipBytes(800, 600, 10) == UINT64_C(2559756));

const SequenceResolutionProfile *SequenceResolutionProfileFor(
    const DriverCommand &logical) {
  for (const SequenceResolutionProfile &profile :
       kSequenceResolutionProfiles) {
    if (logical.framebuffer_width == profile.width &&
        logical.framebuffer_height == profile.height &&
        logical.width == profile.width && logical.height == profile.height) {
      return &profile;
    }
  }
  return nullptr;
}

bool Reject(std::string *error, const std::string &reason) {
  if (error)
    *error = reason;
  return false;
}

std::string DrawReason(const char *profile, std::size_t ordinal,
                       const char *field) {
  return std::string(profile) + " PCO draw " + std::to_string(ordinal) +
         " has invalid " + field;
}

template <typename T>
std::uint64_t Fnv1a64(const std::vector<T> &payload) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(payload.data());
  const std::size_t byte_count = payload.size() * sizeof(T);
  std::uint64_t hash = kFnvOffset;
  for (std::size_t index = 0; index < byte_count; ++index) {
    hash ^= bytes[index];
    hash *= kFnvPrime;
  }
  return hash;
}

bool RootPayloadIsEmpty(const DriverCommand &command) {
  return command.raw_vertex_data.empty() && command.vertex_pco.empty() &&
         command.fragment_pco.empty() && command.vertex_shared.empty() &&
         command.fragment_shared.empty() && command.sampled_textures.empty() &&
         command.sampled_texture_count == 0 &&
         command.vertex_sampled_texture_count == 0 &&
         command.fragment_sampled_texture_count == 0 &&
         command.sampled_texture_bytes.empty() &&
         command.declared_sampled_texture_bytes_size == 0 &&
         command.declared_raw_vertex_data_size == 0 &&
         command.declared_vertex_pco_size == 0 &&
         command.declared_fragment_pco_size == 0;
}

bool SequenceEnvelopeMatches(const Options &options, const char *case_name,
                             std::size_t physical_draw_count,
                             const SequenceResolutionProfile **resolution,
                             std::string *error) {
  const DriverCommand &logical = options.driver_command;
  const SequenceResolutionProfile *profile =
      SequenceResolutionProfileFor(logical);
  if (!logical.enabled || logical.schema != kDriverCommandSchema ||
      logical.producer != kDriverCommandProducer ||
      logical.command != kDrawPcoSequence || logical.test_case != case_name ||
      logical.frame != 1 || !profile || logical.format != kRgba8 ||
      !RootPayloadIsEmpty(logical) || !resolution) {
    return Reject(error, std::string(case_name) +
                             " PCO logical command envelope is invalid");
  }
  if (options.driver_commands.size() != physical_draw_count) {
    return Reject(error, std::string(case_name) + " PCO sequence requires " +
                             std::to_string(physical_draw_count) +
                             " physical draws");
  }
  *resolution = profile;
  return true;
}

/*
 * Input-assembly totals a sequence's draws add up to, and the rule that
 * everything downstream of clipping is left for SystemC to measure.
 *
 * These profiles used to pin the whole counter set to constants recorded from
 * a golden run, which meant a capture reported a rasterization result whether
 * or not the model produced one.  The contract now is narrower and checkable:
 * the driver states what the draws submit, and states nothing else.
 */
bool LogicalUnusedCountersAreZero(const DriverCommand &command);

bool SequenceLogicalCountersAreDerived(const Options &options,
                                       std::string *error) {
  const DriverCommand &logical = options.driver_command;
  if (logical.draw_count != options.driver_commands.size())
    return Reject(error, "sequence draw count does not match its draws");

  std::uint64_t vertices = 0;
  std::uint64_t primitives = 0;
  bool any_indexed = false;
  for (const DriverCommand &draw : options.driver_commands) {
    const std::uint64_t assembled =
        draw.indexed != 0 ? draw.index_count : draw.vertex_count;
    any_indexed = any_indexed || draw.indexed != 0;
    vertices += assembled;
    switch (draw.primitive_mode) {
      case 0U: primitives += assembled; break;
      case 1U: primitives += assembled / 2U; break;
      case 2U: primitives += assembled; break;
      case 3U: primitives += assembled >= 2U ? assembled - 1U : 0U; break;
      case 4U: primitives += assembled / 3U; break;
      case 5U:
      case 6U: primitives += assembled >= 3U ? assembled - 2U : 0U; break;
      default:
        return Reject(error, "sequence draw has an unknown topology");
    }
  }

  if (logical.ia_vertices != vertices || logical.ia_primitives != primitives ||
      logical.clip_invocations != primitives) {
    return Reject(error,
                  "sequence input-assembly totals do not match its draws");
  }
  // Vertex shading follows the vertex count exactly unless the post-transform
  // cache decides it, which only an indexed draw can do.
  const std::uint64_t expected_vs = any_indexed ? 0U : vertices;
  if (logical.vs_invocations != expected_vs)
    return Reject(error, "sequence vertex-shading total is not derived");
  if (logical.clip_primitives != 0 || logical.setup_triangles != 0 ||
      logical.ps_invocations != 0 || logical.semantic_texel_fetches != 0) {
    return Reject(error,
                  "sequence states a rasterization result the model measures");
  }
  return LogicalUnusedCountersAreZero(logical);
}

bool LogicalUnusedCountersAreZero(const DriverCommand &command) {
  return command.index_count == 0 && command.unique_vertices == 0 &&
         command.primitive_count == 0 && command.gs_invocations == 0 &&
         command.gs_primitives == 0 && command.hs_invocations == 0 &&
         command.ds_invocations == 0 && command.cs_invocations == 0;
}

bool NestedCountersAreZero(const DriverCommand &command) {
  return command.draw_count == 0 && command.index_count == 0 &&
         command.unique_vertices == 0 && command.primitive_count == 0 &&
         command.clip_primitives == 0 && command.setup_triangles == 0 &&
         command.semantic_texel_fetches == 0 && command.ia_vertices == 0 &&
         command.ia_primitives == 0 && command.vs_invocations == 0 &&
         command.gs_invocations == 0 && command.gs_primitives == 0 &&
         command.clip_invocations == 0 && command.ps_invocations == 0 &&
         command.hs_invocations == 0 && command.ds_invocations == 0 &&
         command.cs_invocations == 0;
}

struct DrawSpec {
  const char *case_name;
  std::uint32_t width;
  std::uint32_t height;
  std::array<std::uint32_t, 4> clear_color;
  std::uint32_t vertex_stride;
  std::uint32_t vertex_count;
  std::uint32_t primitive_mode;
  std::uint64_t vertex_fnv;
  std::size_t vertex_pco_bytes;
  std::uint64_t vertex_pco_fnv;
  std::size_t fragment_pco_bytes;
  std::uint64_t fragment_pco_fnv;
  DriverPcoStageAbi vertex_abi;
  DriverPcoStageAbi fragment_abi;
  std::size_t vertex_shared_dwords;
  std::uint64_t vertex_shared_fnv;
  std::size_t fragment_shared_dwords;
  std::uint64_t fragment_shared_fnv;
  std::uint32_t varying_components;
  std::array<std::uint32_t, 3> viewport;
  std::uint32_t cull_face;
  std::uint32_t color_mask;
  std::uint32_t blend_enable;
  std::uint32_t blend_source_factor;
  std::uint32_t blend_destination_factor;
  std::uint32_t depth_enable;
  std::uint32_t depth_write;
  std::uint32_t depth_func;
  std::uint32_t depth_clear_bits;
  std::uint32_t depth_format;
  std::uint32_t color_attachment_source;
  std::uint32_t depth_attachment_source;
  std::uint32_t vertex_texture_count;
  std::uint32_t fragment_texture_count;
};

bool DrawHeaderAndPayloadMatch(const DriverCommand &command,
                               const DrawSpec &spec, const char *profile,
                               std::size_t ordinal, std::string *error) {
  const std::uint64_t expected_vertex_bytes =
      static_cast<std::uint64_t>(spec.vertex_stride) * spec.vertex_count;
  if (!command.enabled || command.schema != kDriverCommandSchema ||
      command.producer != kDriverCommandProducer ||
      command.command != kDrawPcoTriangles ||
      command.test_case != spec.case_name || command.frame != 1 ||
      command.framebuffer_width != spec.width ||
      command.framebuffer_height != spec.height ||
      command.width != spec.width || command.height != spec.height ||
      command.format != kRgba8 || command.clear_color_bits != spec.clear_color) {
    return Reject(error, DrawReason(profile, ordinal, "command envelope"));
  }
  if (command.vertex_stride != spec.vertex_stride ||
      command.vertex_count != spec.vertex_count || command.first_vertex != 0 ||
      command.instance_count != 1 ||
      command.primitive_mode != spec.primitive_mode || command.indexed != 0 ||
      expected_vertex_bytes > std::numeric_limits<std::size_t>::max() ||
      command.raw_vertex_data.size() != expected_vertex_bytes ||
      command.declared_raw_vertex_data_size != expected_vertex_bytes ||
      Fnv1a64(command.raw_vertex_data) != spec.vertex_fnv) {
    return Reject(error, DrawReason(profile, ordinal, "VBO/topology"));
  }
  const bool vertex_pco_matches =
      (command.vertex_pco.size() == spec.vertex_pco_bytes &&
       command.declared_vertex_pco_size == spec.vertex_pco_bytes &&
       Fnv1a64(command.vertex_pco) == spec.vertex_pco_fnv) ||
      (std::strcmp(spec.case_name, kShadowCase) == 0 && ordinal == 2 &&
       command.vertex_pco.size() == 744U &&
       command.declared_vertex_pco_size == 744U &&
       Fnv1a64(command.vertex_pco) == UINT64_C(0xe6bc6969c1a52652));
  if (!vertex_pco_matches ||
      command.fragment_pco.size() != spec.fragment_pco_bytes ||
      command.declared_fragment_pco_size != spec.fragment_pco_bytes ||
      Fnv1a64(command.fragment_pco) != spec.fragment_pco_fnv ||
      !DriverPcoStageAbiMatches(command.vertex_pco_abi, spec.vertex_abi) ||
      !DriverPcoStageAbiMatches(command.fragment_pco_abi,
                                spec.fragment_abi)) {
    return Reject(error, DrawReason(profile, ordinal, "PCO binary/ABI"));
  }
  if (command.vertex_shared.size() != spec.vertex_shared_dwords ||
      (spec.vertex_shared_dwords != 0 &&
       Fnv1a64(command.vertex_shared) != spec.vertex_shared_fnv) ||
      command.fragment_shared.size() != spec.fragment_shared_dwords ||
      (spec.fragment_shared_dwords != 0 &&
       Fnv1a64(command.fragment_shared) != spec.fragment_shared_fnv)) {
    return Reject(error, DrawReason(profile, ordinal, "shared payload"));
  }
  if (command.position_output_start != 0 ||
      command.position_output_count != 4 || command.varying_output_start != 4 ||
      command.varying_output_count != spec.varying_components ||
      command.fragment_position_start != 0 ||
      command.fragment_position_count != 4 ||
      command.fragment_varying_start != 4 ||
      command.fragment_varying_count != spec.varying_components * 4U) {
    return Reject(error, DrawReason(profile, ordinal, "stage linkage"));
  }
  return true;
}

bool DrawFixedFunctionStateMatches(const DriverCommand &command,
                                   const DrawSpec &spec, const char *profile,
                                   std::size_t ordinal, std::string *error) {
  if (command.viewport_scale_bits != spec.viewport ||
      command.viewport_translate_bits != spec.viewport ||
      command.front_ccw != 0 || command.cull_face != spec.cull_face ||
      command.fill_front != 0 || command.fill_back != 0 ||
      command.scissor != 0 || command.rasterizer_discard != 0 ||
      command.multisample != 0 || command.half_pixel_center != 1 ||
      command.bottom_edge_rule != 0 || command.clip_halfz != 0 ||
      command.depth_clip_near != 1 || command.depth_clip_far != 1 ||
      command.depth_clamp != 0 || command.sample_mask != UINT32_MAX ||
      command.color_mask != spec.color_mask ||
      command.blend_enable != spec.blend_enable ||
      command.blend_rgb_equation != kBlendAdd ||
      command.blend_alpha_equation != kBlendAdd ||
      command.blend_source_rgb_factor != spec.blend_source_factor ||
      command.blend_destination_rgb_factor !=
          spec.blend_destination_factor ||
      command.blend_source_alpha_factor != spec.blend_source_factor ||
      command.blend_destination_alpha_factor !=
          spec.blend_destination_factor ||
      command.dither != 1 || command.depth_enable != spec.depth_enable ||
      command.depth_write != spec.depth_write ||
      command.depth_func != spec.depth_func ||
      command.depth_clear_bits != spec.depth_clear_bits ||
      command.depth_format != spec.depth_format ||
      command.color_attachment_source_command_index !=
          spec.color_attachment_source ||
      command.depth_attachment_source_command_index !=
          spec.depth_attachment_source) {
    return Reject(error,
                  DrawReason(profile, ordinal, "raster/blend/depth state"));
  }
  const std::uint32_t total_textures =
      spec.vertex_texture_count + spec.fragment_texture_count;
  if (command.sampled_texture_count != total_textures ||
      command.sampled_textures.size() != total_textures ||
      command.vertex_sampled_texture_count != spec.vertex_texture_count ||
      command.fragment_sampled_texture_count !=
          spec.fragment_texture_count ||
      !command.sampled_texture_bytes.empty() ||
      command.declared_sampled_texture_bytes_size != 0 ||
      command.sampled_texture_width != 0 ||
      command.sampled_texture_height != 0 ||
      command.sampled_texture_row_pitch != 0 ||
      !command.sampled_texture_format.empty() ||
      command.sampled_texture_mip_count != 0 ||
      !NestedCountersAreZero(command)) {
    return Reject(error,
                  DrawReason(profile, ordinal, "resource/counter state"));
  }
  return true;
}

bool DrawMatches(const DriverCommand &command, const DrawSpec &spec,
                 const char *profile, std::size_t ordinal,
                 std::string *error) {
  return DrawHeaderAndPayloadMatch(command, spec, profile, ordinal, error) &&
         DrawFixedFunctionStateMatches(command, spec, profile, ordinal, error);
}

bool MipChainMatches(const DriverPcoSampledTexture &texture,
                     std::uint32_t width, std::uint32_t height,
                     std::uint32_t mip_count) {
  if (mip_count == 0 || mip_count > texture.mip.size() ||
      texture.mip_count != mip_count)
    return false;
  std::uint64_t offset = 0;
  for (std::size_t level = 0; level < texture.mip.size(); ++level) {
    const auto &mip = texture.mip[level];
    if (level < mip_count) {
      const std::uint32_t level_width = width >> level ? width >> level : 1U;
      const std::uint32_t level_height =
          height >> level ? height >> level : 1U;
      const std::uint32_t pitch = level_width * 4U;
      if (offset > UINT32_MAX || mip.width != level_width ||
          mip.height != level_height || mip.row_pitch_bytes != pitch ||
          mip.offset_bytes != offset) {
        return false;
      }
      offset += static_cast<std::uint64_t>(pitch) * level_height;
    } else if (mip.width != 0 || mip.height != 0 ||
               mip.row_pitch_bytes != 0 || mip.offset_bytes != 0) {
      return false;
    }
  }
  return offset == texture.declared_bytes_size;
}

bool TextureMatches(const DriverPcoSampledTexture &texture,
                    DriverPcoTextureSource source,
                    DriverPcoShaderStage stage, std::uint32_t producer,
                    std::uint32_t descriptor_set, std::uint32_t width,
                    std::uint32_t height, std::uint32_t mip_count,
                    std::uint64_t declared_size, std::uint32_t mip_filter,
                    std::uint32_t wrap, std::uint32_t max_lod,
                    std::uint64_t payload_fnv = 0,
                    bool external_rgbx_allowed = false,
                    const char *required_format = kRgba8) {
  const bool external = source == DriverPcoTextureSource::kExternalPayload;
  const bool nearest = std::string_view(required_format) == kZ32;
  const bool format_ok =
      texture.format == required_format ||
      (external && external_rgbx_allowed && texture.format == kRgbx8);
  const bool payload_ok =
      external ? texture.bytes.size() == declared_size && payload_fnv != 0 &&
                     Fnv1a64(texture.bytes) == payload_fnv
               : texture.bytes.empty() && payload_fnv == 0;
  return texture.source == source && texture.stage == stage &&
         texture.producer_command_index == producer &&
         texture.descriptor_set == descriptor_set && texture.binding == 0 &&
         format_ok && payload_ok &&
         texture.declared_bytes_size == declared_size &&
         MipChainMatches(texture, width, height, mip_count) &&
         texture.min_filter == (nearest ? kFilterNearest : kFilterLinear) &&
         texture.mag_filter == (nearest ? kFilterNearest : kFilterLinear) &&
         texture.mip_filter == mip_filter && texture.wrap_u == wrap &&
         texture.wrap_v == wrap && texture.normalized_coordinates == 1 &&
         texture.min_lod_u4_6 == 0 && texture.max_lod_u4_6 == max_lod;
}

constexpr DriverPcoStageAbi kRefractPrepassVertexAbi = {
    10, 8, 7, 0, 16, 0, 16, 0};
constexpr DriverPcoStageAbi kRefractPrepassFragmentAbi = {
    3, 0, 0, 16, 0, 0, 0, 0};
constexpr DriverPcoStageAbi kRefractCompositeVertexAbi = {
    20, 8, 15, 0, 64, 0, 64, 0};
constexpr DriverPcoStageAbi kRefractCompositeFragmentAbi = {
    21, 0, 0, 48, 60, 0, 0, 0};

constexpr std::array<DrawSpec, 2> kRefractDraws = {{
    {kRefractCase,
     160,
     120,
     kOpaqueBlack,
     24,
     208998,
     kTriangles,
     kDriverPcoRefractVertexFnv1a64,
     432,
     kDriverPcoRefractPrepassVertexPcoFnv1a64,
     56,
     kDriverPcoRefractPrepassFragmentPcoFnv1a64,
     kRefractPrepassVertexAbi,
     kRefractPrepassFragmentAbi,
     16,
     kDriverPcoRefractPrepassSharedFnv1a64,
     0,
     0,
     3,
     {UINT32_C(0x42a00000), UINT32_C(0x42700000),
      UINT32_C(0x3f000000)},
     kCullFront,
     0x0f,
     0,
     kBlendOne,
     kBlendZero,
     1,
     1,
     kDepthLequal,
     UINT32_C(0x3f800000),
     kDriverPcoDepthFormatZ32Unorm,
     kDriverPcoNewAttachment,
     kDriverPcoNewAttachment,
     0,
     0},
    {kRefractCase,
     80,
     60,
     kOpaqueBlack,
     24,
     208998,
     kTriangles,
     kDriverPcoRefractVertexFnv1a64,
     1536,
     kDriverPcoRefractCompositeVertexPcoFnv1a64,
     5072,
     kDriverPcoRefractCompositeFragmentPcoFnv1a64,
     kRefractCompositeVertexAbi,
     kRefractCompositeFragmentAbi,
     64,
     kDriverPcoRefractCompositeSharedFnv1a64,
     60,
     kDriverPcoRefractFragmentSharedFnv1a64,
     11,
     {UINT32_C(0x42200000), UINT32_C(0x41f00000),
      UINT32_C(0x3f000000)},
     kCullBack,
     0x0f,
     0,
     kBlendOne,
     kBlendZero,
     1,
     1,
     kDepthLequal,
     UINT32_C(0x3f800000),
     kDriverPcoDepthFormatZ24X8Unorm,
     kDriverPcoNewAttachment,
     kDriverPcoNewAttachment,
     0,
     3},
}};

DrawSpec RefractDrawSpec(std::size_t ordinal,
                         const SequenceResolutionProfile &resolution) {
  DrawSpec spec = kRefractDraws[ordinal];
  const bool prepass = ordinal == 0;
  spec.width = prepass ? resolution.offscreen_width : resolution.width;
  spec.height = prepass ? resolution.offscreen_height : resolution.height;
  spec.viewport =
      prepass ? resolution.offscreen_viewport : resolution.viewport;
  if (!prepass)
    spec.fragment_shared_fnv = resolution.refract_fragment_shared_fnv;
  return spec;
}

bool RefractResourcesMatch(const std::vector<DriverCommand> &commands,
                           const SequenceResolutionProfile &resolution,
                           std::string *error) {
  if (!commands[0].sampled_textures.empty() ||
      commands[1].sampled_textures.size() != 3) {
    return Reject(error, "Refract PCO resource cardinality is invalid");
  }
  const auto &depth = commands[1].sampled_textures[0];
  const auto &color = commands[1].sampled_textures[1];
  const auto &external = commands[1].sampled_textures[2];
  if (!TextureMatches(depth,
                      DriverPcoTextureSource::kPreviousDepthAttachment,
                      DriverPcoShaderStage::kFragment, 0, 0,
                      resolution.offscreen_width,
                      resolution.offscreen_height, 1,
                      resolution.offscreen_depth_bytes, kMipNone, kWrapClamp,
                      0, 0, false, kZ32) ||
      !TextureMatches(color,
                      DriverPcoTextureSource::kPreviousColorAttachment,
                      DriverPcoShaderStage::kFragment, 0, 1,
                      resolution.offscreen_width,
                      resolution.offscreen_height,
                      resolution.offscreen_mip_count,
                      resolution.offscreen_rgba8_bytes, kMipLinear,
                      kWrapClamp, resolution.offscreen_max_lod_u4_6) ||
      !TextureMatches(external,
                      DriverPcoTextureSource::kExternalPayload,
                      DriverPcoShaderStage::kFragment, 0, 2, 512, 512, 1,
                      1048576, kMipNone, kWrapClamp, 0,
                      kDriverPcoRefractExternalFnv1a64)) {
    return Reject(error, "Refract PCO sampled-resource contract is invalid");
  }
  return true;
}

bool RefractSupported(const Options &options, std::string *error) {
  const SequenceResolutionProfile *resolution = nullptr;
  if (!SequenceEnvelopeMatches(options, kRefractCase, kRefractDraws.size(),
                               &resolution, error)) {
    return false;
  }
  const DriverCommand &logical = options.driver_command;
  if (logical.clear_color_bits != kOpaqueBlack ||
      !SequenceLogicalCountersAreDerived(options, error)) {
    return Reject(error,
                  error && !error->empty()
                      ? "Refract PCO logical counters are invalid: " + *error
                      : std::string("Refract PCO logical counters are invalid"));
  }
  for (std::size_t ordinal = 0; ordinal < kRefractDraws.size(); ++ordinal) {
    const DrawSpec spec = RefractDrawSpec(ordinal, *resolution);
    if (!DrawMatches(options.driver_commands[ordinal], spec,
                     "Refract", ordinal, error)) {
      return false;
    }
  }
  return RefractResourcesMatch(options.driver_commands, *resolution, error);
}

constexpr std::uint64_t kShadowMeshVertexFnv =
    UINT64_C(0x8ed96606410719eb);
constexpr std::uint64_t kShadowMaskVertexFnv =
    UINT64_C(0xa5f3d6f332c36765);
constexpr std::uint64_t kShadowDepthSharedFnv =
    UINT64_C(0x4f2ffc57463c4710);
constexpr std::uint64_t kShadowMaskSharedFnv =
    UINT64_C(0x9f324a4e93b44b28);
constexpr std::uint64_t kShadowSceneSharedFnv =
    UINT64_C(0xbf09db1f576ac6d9);
constexpr std::uint64_t kShadowFragmentSharedFnv =
    UINT64_C(0x5d306f7625b3e88e);

constexpr std::array<DrawSpec, 3> kShadowDraws = {{
    {kShadowCase,
     160,
     120,
     kOpaqueBlack,
     24,
     21516,
     kTriangles,
     kShadowMeshVertexFnv,
     432,
     UINT64_C(0x6e9ad97e49eca9fe),
     56,
     UINT64_C(0xa55a28d91b0f4b9e),
     {10, 8, 7, 0, 16, 0, 16, 0},
     {3, 0, 0, 16, 0, 0, 0, 0},
     16,
     kShadowDepthSharedFnv,
     0,
     0,
     3,
     {UINT32_C(0x42a00000), UINT32_C(0x42700000),
      UINT32_C(0x3f000000)},
     kCullBack,
     0,
     0,
     kBlendOne,
     kBlendZero,
     1,
     1,
     kDepthLequal,
     UINT32_C(0x3f800000),
     kDriverPcoDepthFormatZ32Unorm,
     kDriverPcoNewAttachment,
     kDriverPcoNewAttachment,
     0,
     0},
    {kShadowCase,
     80,
     60,
     kOpaqueBlack,
     8,
     4,
     kTriangleStrip,
     kShadowMaskVertexFnv,
     464,
     UINT64_C(0x79b5f95f5c89ad6c),
     216,
     UINT64_C(0x1ac54b25af8de102),
     {16, 4, 8, 0, 32, 0, 32, 0},
     {11, 0, 0, 20, 20, 0, 0, 0},
     32,
     kShadowMaskSharedFnv,
     20,
     kShadowFragmentSharedFnv,
     4,
     {UINT32_C(0x42200000), UINT32_C(0x41f00000),
      UINT32_C(0x3f000000)},
     kCullBack,
     0x0f,
     0,
     kBlendOne,
     kBlendZero,
     1,
     1,
     kDepthLequal,
     UINT32_C(0x3f800000),
     kDriverPcoDepthFormatZ24X8Unorm,
     kDriverPcoNewAttachment,
     kDriverPcoNewAttachment,
     0,
     1},
    {kShadowCase,
     80,
     60,
     kOpaqueBlack,
     24,
     21516,
     kTriangles,
     kShadowMeshVertexFnv,
     728,
     UINT64_C(0x385c48c6c28cd9fc),
     56,
     UINT64_C(0x24f632ab8095faeb),
     {11, 8, 5, 0, 32, 0, 32, 0},
     {1, 0, 0, 8, 0, 0, 0, 0},
     32,
     kShadowSceneSharedFnv,
     0,
     0,
     1,
     {UINT32_C(0x42200000), UINT32_C(0x41f00000),
      UINT32_C(0x3f000000)},
     kCullBack,
     0x0f,
     0,
     kBlendOne,
     kBlendZero,
     1,
     1,
     kDepthLequal,
     UINT32_C(0x3f800000),
     kDriverPcoDepthFormatZ24X8Unorm,
     1,
     1,
     0,
     0},
}};

DrawSpec ShadowDrawSpec(std::size_t ordinal,
                        const SequenceResolutionProfile &resolution) {
  DrawSpec spec = kShadowDraws[ordinal];
  const bool depth = ordinal == 0;
  spec.width = depth ? resolution.offscreen_width : resolution.width;
  spec.height = depth ? resolution.offscreen_height : resolution.height;
  spec.viewport = depth ? resolution.offscreen_viewport : resolution.viewport;
  if (ordinal == 1)
    spec.fragment_shared_fnv = resolution.shadow_fragment_shared_fnv;
  return spec;
}

bool ShadowSupported(const Options &options, std::string *error) {
  const SequenceResolutionProfile *resolution = nullptr;
  if (!SequenceEnvelopeMatches(options, kShadowCase, kShadowDraws.size(),
                               &resolution, error)) {
    return false;
  }
  const DriverCommand &logical = options.driver_command;
  if (logical.clear_color_bits != kOpaqueBlack ||
      !SequenceLogicalCountersAreDerived(options, error)) {
    return Reject(error,
                  error && !error->empty()
                      ? "Shadow PCO logical counters are invalid: " + *error
                      : std::string("Shadow PCO logical counters are invalid"));
  }
  for (std::size_t ordinal = 0; ordinal < kShadowDraws.size(); ++ordinal) {
    const DrawSpec spec = ShadowDrawSpec(ordinal, *resolution);
    if (!DrawMatches(options.driver_commands[ordinal], spec,
                     "Shadow", ordinal, error)) {
      return false;
    }
  }
  if (!options.driver_commands[0].sampled_textures.empty() ||
      options.driver_commands[1].sampled_textures.size() != 1 ||
      !options.driver_commands[2].sampled_textures.empty() ||
      !TextureMatches(
          options.driver_commands[1].sampled_textures[0],
          DriverPcoTextureSource::kPreviousDepthAttachment,
          DriverPcoShaderStage::kFragment, 0, 0,
          resolution->offscreen_width, resolution->offscreen_height, 1,
          resolution->offscreen_depth_bytes, kMipNone, kWrapClamp, 0, 0,
          false, kZ32)) {
    return Reject(error, "Shadow PCO sampled-depth contract is invalid");
  }
  return true;
}

constexpr std::array<std::uint64_t, 8> kTerrainVertexFnv = {
    UINT64_C(0x137ad857d68f72e5), UINT64_C(0x137ad857d68f72e5),
    UINT64_C(0xc33cf9ea6c986551), UINT64_C(0x137ad857d68f72e5),
    UINT64_C(0x137ad857d68f72e5), UINT64_C(0x137ad857d68f72e5),
    UINT64_C(0x137ad857d68f72e5), UINT64_C(0x137ad857d68f72e5)};
constexpr std::array<std::uint64_t, 8> kTerrainVertexSharedFnv = {
    UINT64_C(0x48fff97294e45f55), UINT64_C(0x15e8065d3d6b1b55),
    UINT64_C(0x798ce5dd9c33fa18), UINT64_C(0x15e8065d3d6b1b55),
    UINT64_C(0x15e8065d3d6b1b55), UINT64_C(0x15e8065d3d6b1b55),
    UINT64_C(0x15e8065d3d6b1b55), UINT64_C(0x15e8065d3d6b1b55)};
constexpr std::array<std::uint64_t, 8> kTerrainFragmentSharedFnv = {
    UINT64_C(0x62101b5902762818), UINT64_C(0x4e1ccea0e6192d58),
    UINT64_C(0x1369112ad898bbfd), UINT64_C(0x2d423f9c5838f4fd),
    UINT64_C(0x21d394b1ca541e48), UINT64_C(0x4755794a96dd0179),
    UINT64_C(0x2d423f9c5838f4fd), UINT64_C(0x2d423f9c5838f4fd)};
constexpr std::array<std::uint64_t, 7> kTerrainMainTextureFnv = {
    0,
    0,
    UINT64_C(0xa69ccd9838551cb3),
    UINT64_C(0x777443d6a3c0ceeb),
    UINT64_C(0xd510ff3e570680dd),
    0,
    UINT64_C(0x3964257e9bde4861),
};
constexpr std::array<std::size_t, 8> kTerrainVertexPcoBytes = {
    200, 192, 2680, 192, 192, 192, 192, 192};
constexpr std::array<std::uint64_t, 8> kTerrainVertexPcoFnv = {
    UINT64_C(0x9abe96cad5fe9f4e), UINT64_C(0x081618f544cc6abe),
    UINT64_C(0x8d0f6d4b38cdecf4), UINT64_C(0x081618f544cc6abe),
    UINT64_C(0x081618f544cc6abe), UINT64_C(0x081618f544cc6abe),
    UINT64_C(0x081618f544cc6abe), UINT64_C(0x081618f544cc6abe)};
constexpr std::array<std::size_t, 8> kTerrainFragmentPcoBytes = {
    38832, 1208, 7328, 1920, 1880, 448, 3528, 3504};
constexpr std::array<DriverPcoStageAbi, 8> kTerrainVertexAbi = {{
    {7, 4, 6, 0, 8, 0, 8, 0},
    {6, 4, 6, 0, 8, 0, 8, 0},
    {44, 16, 18, 0, 96, 40, 56, 0},
    {6, 4, 6, 0, 8, 0, 8, 0},
    {6, 4, 6, 0, 8, 0, 8, 0},
    {6, 4, 6, 0, 8, 0, 8, 0},
    {6, 4, 6, 0, 8, 0, 8, 0},
    {6, 4, 6, 0, 8, 0, 8, 0},
}};
constexpr std::array<DriverPcoStageAbi, 8> kTerrainFragmentAbi = {{
    {50, 0, 0, 12, 4, 0, 4, 0},
    {24, 0, 0, 12, 28, 20, 8, 0},
    {33, 0, 0, 60, 164, 100, 64, 0},
    {30, 0, 0, 12, 20, 20, 0, 0},
    {30, 0, 0, 12, 20, 20, 0, 0},
    {17, 0, 0, 12, 24, 20, 4, 0},
    {35, 0, 0, 12, 20, 20, 0, 0},
    {35, 0, 0, 12, 20, 20, 0, 0},
}};
constexpr std::array<std::size_t, 8> kTerrainVertexSharedDwords = {
    8, 8, 96, 8, 8, 8, 8, 8};
constexpr std::array<std::size_t, 8> kTerrainFragmentSharedDwords = {
    4, 28, 164, 20, 20, 24, 20, 20};

DrawSpec TerrainDrawSpec(std::size_t ordinal,
                         const SequenceResolutionProfile &resolution) {
  const bool main = ordinal == 2;
  const bool extent_256 = ordinal == 0 || ordinal == 1 || ordinal == 3 ||
                          ordinal == 4;
  const std::uint32_t width = extent_256 ? 256U : resolution.width;
  const std::uint32_t height = extent_256 ? 256U : resolution.height;
  const bool depth = ordinal == 2 || ordinal == 5 || ordinal == 7;
  const bool z24 = ordinal == 7;
  const bool d6 = ordinal == 5;
  const bool output_descriptor = ordinal == 3 || ordinal == 6 || ordinal == 7;
  const std::uint32_t vertex_textures = main ? 2U : 0U;
  const std::uint32_t fragment_textures =
      ordinal == 0 ? 0U : (main ? 5U : 1U);
  return {kTerrainCase,
          width,
          height,
          ordinal <= 1 ? kOpaqueBlack : kTerrainClear,
          main ? 44U : 12U,
          main ? 393216U : 6U,
          kTriangles,
          kTerrainVertexFnv[ordinal],
          kTerrainVertexPcoBytes[ordinal],
          kTerrainVertexPcoFnv[ordinal],
          kTerrainFragmentPcoBytes[ordinal],
          resolution.terrain_fragment_pco_fnv[ordinal],
          kTerrainVertexAbi[ordinal],
          kTerrainFragmentAbi[ordinal],
          kTerrainVertexSharedDwords[ordinal],
          kTerrainVertexSharedFnv[ordinal],
          kTerrainFragmentSharedDwords[ordinal],
          output_descriptor ? resolution.terrain_output_fragment_shared_fnv :
                              kTerrainFragmentSharedFnv[ordinal],
          main ? 14U : 2U,
          extent_256 ? std::array<std::uint32_t, 3>{
                           UINT32_C(0x43000000), UINT32_C(0x43000000),
                           UINT32_C(0x3f000000)} :
                       resolution.viewport,
          kCullBack,
          0x0f,
          d6 ? 1U : 0U,
          d6 ? kBlendSourceAlpha : kBlendOne,
          d6 ? kBlendOne : kBlendZero,
          depth ? 1U : 0U,
          depth ? 1U : 0U,
          depth ? kDepthLequal : 0U,
          (ordinal == 2 || ordinal == 7) ? UINT32_C(0x3f800000) : 0U,
          depth ? (z24 ? kDriverPcoDepthFormatZ24X8Unorm
                       : kDriverPcoDepthFormatZ16Unorm)
                : 0U,
          d6 ? 2U : kDriverPcoNewAttachment,
          d6 ? 2U : kDriverPcoNewAttachment,
          vertex_textures,
          fragment_textures};
}

bool TerrainResourcesMatch(const std::vector<DriverCommand> &commands,
                           const SequenceResolutionProfile &resolution,
                           std::string *error) {
  if (!commands[0].sampled_textures.empty())
    return Reject(error, "Terrain D1 must not have sampled resources");
  if (commands[1].sampled_textures.size() != 1 ||
      !TextureMatches(commands[1].sampled_textures[0],
                      DriverPcoTextureSource::kPreviousColorAttachment,
                      DriverPcoShaderStage::kFragment, 0, 0, 256, 256, 1,
                      262144, kMipNone, kWrapClamp, 0)) {
    return Reject(error, "Terrain D2 sampled-resource contract is invalid");
  }
  const auto &main = commands[2].sampled_textures;
  const bool main_ok =
      main.size() == 7 &&
      TextureMatches(main[0],
                     DriverPcoTextureSource::kPreviousColorAttachment,
                     DriverPcoShaderStage::kVertex, 0, 0, 256, 256, 1,
                     262144, kMipNone, kWrapClamp, 0) &&
      TextureMatches(main[1],
                     DriverPcoTextureSource::kPreviousColorAttachment,
                     DriverPcoShaderStage::kVertex, 1, 1, 256, 256, 1,
                     262144, kMipNone, kWrapClamp, 0) &&
      TextureMatches(main[2], DriverPcoTextureSource::kExternalPayload,
                     DriverPcoShaderStage::kFragment, 0, 0, 512, 512,
                     10, 1398100, kMipLinear, kWrapRepeat, 576,
                     kTerrainMainTextureFnv[2], true) &&
      DriverPcoTerrainExternalPayloadHashMatches(2,
                                                  Fnv1a64(main[2].bytes)) &&
      TextureMatches(main[3], DriverPcoTextureSource::kExternalPayload,
                     DriverPcoShaderStage::kFragment, 0, 1, 512, 512,
                     10, 1398100, kMipLinear, kWrapRepeat, 576,
                     kTerrainMainTextureFnv[3], true) &&
      DriverPcoTerrainExternalPayloadHashMatches(3,
                                                  Fnv1a64(main[3].bytes)) &&
      TextureMatches(main[4], DriverPcoTextureSource::kExternalPayload,
                     DriverPcoShaderStage::kFragment, 0, 2, 512, 512,
                     10, 1398100, kMipLinear, kWrapRepeat, 576,
                     kTerrainMainTextureFnv[4], true) &&
      DriverPcoTerrainExternalPayloadHashMatches(4,
                                                  Fnv1a64(main[4].bytes)) &&
      TextureMatches(main[5],
                     DriverPcoTextureSource::kPreviousColorAttachment,
                     DriverPcoShaderStage::kFragment, 0, 3, 256, 256, 1,
                     262144, kMipNone, kWrapClamp, 0) &&
      TextureMatches(main[6], DriverPcoTextureSource::kExternalPayload,
                     DriverPcoShaderStage::kFragment, 0, 4, 512, 512,
                     10, 1398100, kMipLinear, kWrapRepeat, 576,
                     kTerrainMainTextureFnv[6], true) &&
      DriverPcoTerrainExternalPayloadHashMatches(6,
                                                  Fnv1a64(main[6].bytes));
  if (!main_ok)
    return Reject(error, "Terrain D3 stage-banked resource contract is invalid");

  struct PreviousSpec {
    std::uint32_t producer;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t mips;
    std::uint64_t bytes;
    std::uint32_t max_lod;
  };
  const std::array<PreviousSpec, 5> previous = {{
      {2, resolution.width, resolution.height,
       resolution.output_mip_count, resolution.output_rgba8_bytes,
       resolution.output_max_lod_u4_6},
      {3, 256, 256, 9, 349524, 512},
      {4, 256, 256, 9, 349524, 512},
      {5, resolution.width, resolution.height,
       resolution.output_mip_count, resolution.output_rgba8_bytes,
       resolution.output_max_lod_u4_6},
      {6, resolution.width, resolution.height,
       resolution.output_mip_count, resolution.output_rgba8_bytes,
       resolution.output_max_lod_u4_6},
  }};
  for (std::size_t offset = 0; offset < previous.size(); ++offset) {
    const std::size_t ordinal = offset + 3;
    const auto &spec = previous[offset];
    if (commands[ordinal].sampled_textures.size() != 1 ||
        !TextureMatches(
            commands[ordinal].sampled_textures[0],
            DriverPcoTextureSource::kPreviousColorAttachment,
            DriverPcoShaderStage::kFragment, spec.producer, 0, spec.width,
            spec.height, spec.mips, spec.bytes, kMipLinear, kWrapClamp,
            spec.max_lod)) {
      return Reject(error, "Terrain D" + std::to_string(ordinal + 1) +
                               " sampled-resource contract is invalid");
    }
  }
  return true;
}

bool TerrainPhysicalContractMatches(const Options &options,
                                    const SequenceResolutionProfile &resolution,
                                    std::string *error) {
  for (std::size_t ordinal = 0; ordinal < 8; ++ordinal) {
    const DrawSpec spec = TerrainDrawSpec(ordinal, resolution);
    if (!DrawMatches(options.driver_commands[ordinal], spec, "Terrain",
                     ordinal, error)) {
      return false;
    }
  }
  return TerrainResourcesMatch(options.driver_commands, resolution, error);
}

bool TerrainLogicalCountersMatch(const Options &options,
                                 std::string *error) {
  return options.driver_command.clear_color_bits == kTerrainClear &&
         SequenceLogicalCountersAreDerived(options, error);
}

bool TerrainSupported(const Options &options, std::string *error) {
  const SequenceResolutionProfile *resolution = nullptr;
  if (!SequenceEnvelopeMatches(options, kTerrainCase, 8, &resolution, error))
    return false;
  if (!TerrainLogicalCountersMatch(options, error))
    return Reject(error,
                  error && !error->empty()
                      ? "Terrain PCO logical counters are invalid: " + *error
                      : std::string("Terrain PCO logical counters are invalid"));
  return TerrainPhysicalContractMatches(options, *resolution, error);
}

}  // namespace

bool DriverPcoTerrainExternalPayloadHashMatches(
    std::size_t texture_index, std::uint64_t payload_hash) {
  return texture_index < kTerrainMainTextureFnv.size() &&
         kTerrainMainTextureFnv[texture_index] != 0 &&
         payload_hash == kTerrainMainTextureFnv[texture_index];
}

bool DriverPcoTerrainFragmentBinaryHashMatches(
    std::uint32_t width, std::uint32_t height, std::size_t draw_index,
    std::uint64_t binary_hash) {
  if (draw_index >= kTerrainFragmentPcoBytes.size())
    return false;
  for (const SequenceResolutionProfile &profile :
       kSequenceResolutionProfiles) {
    if (profile.width == width && profile.height == height) {
      return profile.terrain_fragment_pco_fnv[draw_index] == binary_hash;
    }
  }
  return false;
}

// A native PCO sequence that is described by its shape instead of a captured
// case name: every nested draw uses the untextured position/colour vertex
// layout, writes the same render target, and continues from the surface the
// previous ordinal produced.  SystemC rasterizes the geometry and measures the
// counters itself, so no profile constants are involved.
bool GenericColorSequenceSupported(const Options &options, std::string *error) {
  const DriverCommand &logical = options.driver_command;
  if (!logical.enabled || logical.schema != kDriverCommandSchema ||
      logical.producer != kDriverCommandProducer ||
      logical.command != kDrawPcoSequence || logical.frame != 1 ||
      logical.format != kRgba8 || !RootPayloadIsEmpty(logical)) {
    return Reject(error, "generic PCO logical command envelope is invalid");
  }
  // The viewport may cover part of the attachment; it just has to fit.
  if (logical.framebuffer_width == 0 || logical.framebuffer_height == 0 ||
      logical.framebuffer_width > 4096 || logical.framebuffer_height > 4096 ||
      logical.width == 0 || logical.height == 0 ||
      logical.width > logical.framebuffer_width ||
      logical.height > logical.framebuffer_height) {
    return Reject(error, "generic PCO sequence render target is invalid");
  }
  if (options.driver_commands.empty()) {
    return Reject(error, "generic PCO sequence has no physical draws");
  }

  for (std::size_t ordinal = 0; ordinal < options.driver_commands.size();
       ++ordinal) {
    const DriverCommand &draw = options.driver_commands[ordinal];
    if (draw.command != kDrawPcoTriangles || draw.test_case != logical.test_case ||
        draw.format != kRgba8 || draw.frame != 1 ||
        draw.framebuffer_width != logical.framebuffer_width ||
        draw.framebuffer_height != logical.framebuffer_height ||
        draw.width != logical.width || draw.height != logical.height ||
        draw.width > draw.framebuffer_width ||
        draw.height > draw.framebuffer_height) {
      return Reject(error, "generic PCO sequence draw envelope is invalid");
    }
    // Untextured position/colour layout: float2 position + float4 colour.
    const bool describes_attributes =
        draw.vertex_attribute_count != 0 &&
        draw.vertex_pco_abi.vertex_inputs ==
            draw.vertex_attribute_count * 4U;
    if ((!describes_attributes &&
         ((draw.vertex_stride != 6U * sizeof(float) &&
           draw.vertex_stride != 8U * sizeof(float)) ||
          draw.vertex_pco_abi.vertex_inputs != 8)) ||
        !draw.sampled_textures.empty() || draw.sampled_texture_count != 0) {
      return Reject(error, "generic PCO sequence draw is not the colour layout");
    }
    // Topologies the submitter expands for setup.  An indexed draw assembles
    // its primitives from the index buffer.  Lines and points are included:
    // the model widens them into real screen-space geometry.
    const std::uint32_t assembled =
        draw.indexed != 0 ? draw.index_count : draw.vertex_count;
    bool topology_assembles = false;
    switch (draw.primitive_mode) {
      case 0U:
        topology_assembles = assembled >= 1U;
        break;
      case 1U:
        topology_assembles = assembled >= 2U && assembled % 2U == 0U;
        break;
      case 2U:
      case 3U:
        topology_assembles = assembled >= 2U;
        break;
      case 4U:
        topology_assembles = assembled >= 3U && assembled % 3U == 0U;
        break;
      case 5U:
      case 6U:
        topology_assembles = assembled >= 3U;
        break;
      default:
        topology_assembles = false;
        break;
    }
    const bool topology_expandable = draw.indexed <= 1U &&
                                     draw.first_vertex == 0 &&
                                     draw.instance_count == 1 &&
                                     topology_assembles;
    if (!topology_expandable) {
      return Reject(error, "generic PCO sequence draw topology is invalid");
    }
    // The first draw starts from a clear; later draws continue from the
    // surface an earlier ordinal produced, never from a forward reference.
    const std::uint32_t expected_source =
        ordinal == 0 ? kDriverPcoNewAttachment
                     : static_cast<std::uint32_t>(ordinal - 1);
    if (draw.color_attachment_source_command_index != expected_source) {
      return Reject(error, "generic PCO sequence colour attachment chain is invalid");
    }
    const bool uses_depth = draw.depth_enable != 0 || draw.depth_write != 0;
    if (!uses_depth &&
        (draw.depth_format != 0 ||
         draw.depth_attachment_source_command_index !=
             kDriverPcoNewAttachment)) {
      return Reject(error, "generic PCO sequence depth attachment is invalid");
    }
    if (uses_depth &&
        draw.depth_attachment_source_command_index != expected_source) {
      return Reject(error, "generic PCO sequence depth attachment chain is invalid");
    }
  }
  return true;
}

bool DriverPcoSequenceSupported(const Options &options, std::string *error) {
  if (error)
    error->clear();
  const DriverCommand &logical = options.driver_command;
  if (logical.command != kDrawPcoSequence) {
    return Reject(error, "command is not a native PCO sequence");
  }
  if (logical.test_case == kRefractCase)
    return RefractSupported(options, error);
  if (logical.test_case == kShadowCase)
    return ShadowSupported(options, error);
  if (logical.test_case == kTerrainCase)
    return TerrainSupported(options, error);
  return GenericColorSequenceSupported(options, error);
}

}  // namespace pvrgpu::stub
