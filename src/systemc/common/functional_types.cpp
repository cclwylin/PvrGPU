// Shared functional payload helpers. PipelineStageName is used for fail-fast
// module diagnostics; ReleaseFunctionalPayloads owns the complete MemoryPool
// lifetime join at the PBE/JSON reporter boundary.
// 中文：集中處理 case mapping、stage 診斷及所有 MemoryPool handle 的釋放。
#include "common/functional_types.h"

#include "common/pipeline_state.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace pvrgpu::stub {

std::size_t DepthAttachmentBytesPerPixel(std::uint32_t format) {
  if (format == kDriverPcoDepthFormatZ16Unorm)
    return sizeof(std::uint16_t);
  if (format == kDriverPcoDepthFormatZ24X8Unorm ||
      format == kDriverPcoDepthFormatZ32Unorm) {
    return sizeof(std::uint32_t);
  }
  throw std::runtime_error("unsupported native depth attachment format");
}

std::uint32_t EncodeDepthAttachmentUnorm(float depth,
                                         std::uint32_t format) {
  if (!std::isfinite(depth) || depth < 0.0F || depth > 1.0F)
    throw std::runtime_error("native depth value is outside [0, 1]");
  const std::uint64_t maximum =
      format == kDriverPcoDepthFormatZ16Unorm
          ? UINT64_C(0xffff)
          : format == kDriverPcoDepthFormatZ24X8Unorm
                ? UINT64_C(0xffffff)
                : format == kDriverPcoDepthFormatZ32Unorm
                      ? UINT64_C(0xffffffff)
                      : throw std::runtime_error(
                            "unsupported native depth attachment format");
  const long double scaled = static_cast<long double>(depth) * maximum;
  const long double integral = std::floor(scaled);
  std::uint64_t encoded = static_cast<std::uint64_t>(integral);
  const long double fraction = scaled - integral;
  if (fraction > 0.5L || (fraction == 0.5L && (encoded & 1U) != 0))
    ++encoded;
  if (encoded > maximum)
    encoded = maximum;
  return static_cast<std::uint32_t>(encoded);
}

float DecodeDepthAttachmentUnorm(std::uint32_t encoded,
                                 std::uint32_t format) {
  const std::uint32_t maximum =
      format == kDriverPcoDepthFormatZ16Unorm
          ? UINT32_C(0xffff)
          : format == kDriverPcoDepthFormatZ24X8Unorm
                ? UINT32_C(0xffffff)
                : format == kDriverPcoDepthFormatZ32Unorm
                      ? UINT32_MAX
                      : throw std::runtime_error(
                            "unsupported native depth attachment format");
  if ((encoded & ~maximum) != 0)
    throw std::runtime_error("native depth attachment has nonzero padding");
  return static_cast<float>(static_cast<double>(encoded) /
                            static_cast<double>(maximum));
}

std::vector<std::uint32_t> DecodeDepthAttachmentUnormBytes(
    const std::vector<std::uint8_t> &bytes, std::uint32_t format) {
  const std::size_t bytes_per_pixel = DepthAttachmentBytesPerPixel(format);
  if (bytes.empty() || bytes.size() % bytes_per_pixel != 0)
    throw std::runtime_error("native depth attachment byte count is invalid");
  std::vector<std::uint32_t> encoded(bytes.size() / bytes_per_pixel, 0);
  for (std::size_t pixel = 0; pixel < encoded.size(); ++pixel) {
    std::memcpy(&encoded[pixel], bytes.data() + pixel * bytes_per_pixel,
                bytes_per_pixel);
    (void)DecodeDepthAttachmentUnorm(encoded[pixel], format);
  }
  return encoded;
}

std::vector<std::uint8_t> EncodeDepthAttachmentUnormBytes(
    const std::vector<std::uint32_t> &encoded, std::uint32_t format) {
  const std::size_t bytes_per_pixel = DepthAttachmentBytesPerPixel(format);
  if (encoded.empty() ||
      encoded.size() >
          std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
    throw std::runtime_error("native depth attachment value count is invalid");
  }
  std::vector<std::uint8_t> bytes(encoded.size() * bytes_per_pixel, 0);
  for (std::size_t pixel = 0; pixel < encoded.size(); ++pixel) {
    (void)DecodeDepthAttachmentUnorm(encoded[pixel], format);
    std::memcpy(bytes.data() + pixel * bytes_per_pixel, &encoded[pixel],
                bytes_per_pixel);
  }
  return bytes;
}

FunctionalCase FunctionalCaseFromName(std::string_view name) {
  if (name == "fill_solid")
    return FunctionalCase::kFillSolid;
  if (name == "fill_solid_depth_neq")
    return FunctionalCase::kFillSolidDepthNotEqual;
  if (name == "fill_solid_depth_never")
    return FunctionalCase::kFillSolidDepthNever;
  if (name == "fill_solid_blended")
    return FunctionalCase::kFillSolidBlended;
  if (name == "triangle_setup")
    return FunctionalCase::kTriangleSetup;
  if (name == "triangle_setup_all_culled")
    return FunctionalCase::kTriangleSetupAllCulled;
  if (name == "triangle_setup_half_culled")
    return FunctionalCase::kTriangleSetupHalfCulled;
  if (name == "attribute_fetch_shader")
    return FunctionalCase::kAttributeFetchShader;
  if (name == "attribute_fetch_shader_2_attr")
    return FunctionalCase::kAttributeFetchShaderTwoAttribute;
  if (name == "attribute_fetch_shader_4_attr")
    return FunctionalCase::kAttributeFetchShaderFourAttribute;
  if (name == "attribute_fetch_shader_8_attr")
    return FunctionalCase::kAttributeFetchShaderEightAttribute;
  if (name == "varyings_shader_1")
    return FunctionalCase::kVaryingsShaderOne;
  if (name == "varyings_shader_2")
    return FunctionalCase::kVaryingsShaderTwo;
  if (name == "varyings_shader_4")
    return FunctionalCase::kVaryingsShaderFour;
  if (name == "varyings_shader_8")
    return FunctionalCase::kVaryingsShaderEight;
  if (name == "fill_tex_nearest")
    return FunctionalCase::kFillTexNearest;
  if (name == "fill_tex_bilinear")
    return FunctionalCase::kFillTexBilinear;
  if (name == "fill_tex_trilinear_linear_01")
    return FunctionalCase::kFillTexTrilinearLinear01;
  if (name == "fill_tex_trilinear_linear_04")
    return FunctionalCase::kFillTexTrilinearLinear04;
  if (name == "fill_tex_trilinear_linear_05")
    return FunctionalCase::kFillTexTrilinearLinear05;
  if (name == "driver_clear_color")
    return FunctionalCase::kDriverClearColor;
  if (name == "driver_triangle_solid")
    return FunctionalCase::kDriverTriangleSolid;
  if (name == "driver_indexed_quad")
    return FunctionalCase::kDriverIndexedQuad;
  if (name == "driver_textured_triangles")
    return FunctionalCase::kDriverTexturedTriangles;
  if (name == "driver_pco_triangles")
    return FunctionalCase::kDriverPcoTriangles;
  return FunctionalCase::kNone;
}

const char *FunctionalCaseName(FunctionalCase functional_case) {
  switch (functional_case) {
  case FunctionalCase::kFillSolid:
    return "fill_solid";
  case FunctionalCase::kFillSolidDepthNotEqual:
    return "fill_solid_depth_neq";
  case FunctionalCase::kFillSolidDepthNever:
    return "fill_solid_depth_never";
  case FunctionalCase::kFillSolidBlended:
    return "fill_solid_blended";
  case FunctionalCase::kTriangleSetup:
    return "triangle_setup";
  case FunctionalCase::kTriangleSetupAllCulled:
    return "triangle_setup_all_culled";
  case FunctionalCase::kTriangleSetupHalfCulled:
    return "triangle_setup_half_culled";
  case FunctionalCase::kAttributeFetchShader:
    return "attribute_fetch_shader";
  case FunctionalCase::kAttributeFetchShaderTwoAttribute:
    return "attribute_fetch_shader_2_attr";
  case FunctionalCase::kAttributeFetchShaderFourAttribute:
    return "attribute_fetch_shader_4_attr";
  case FunctionalCase::kAttributeFetchShaderEightAttribute:
    return "attribute_fetch_shader_8_attr";
  case FunctionalCase::kVaryingsShaderOne:
    return "varyings_shader_1";
  case FunctionalCase::kVaryingsShaderTwo:
    return "varyings_shader_2";
  case FunctionalCase::kVaryingsShaderFour:
    return "varyings_shader_4";
  case FunctionalCase::kVaryingsShaderEight:
    return "varyings_shader_8";
  case FunctionalCase::kFillTexNearest:
    return "fill_tex_nearest";
  case FunctionalCase::kFillTexBilinear:
    return "fill_tex_bilinear";
  case FunctionalCase::kFillTexTrilinearLinear01:
    return "fill_tex_trilinear_linear_01";
  case FunctionalCase::kFillTexTrilinearLinear04:
    return "fill_tex_trilinear_linear_04";
  case FunctionalCase::kFillTexTrilinearLinear05:
    return "fill_tex_trilinear_linear_05";
  case FunctionalCase::kDriverClearColor:
    return "driver_clear_color";
  case FunctionalCase::kDriverTriangleSolid:
    return "driver_triangle_solid";
  case FunctionalCase::kDriverIndexedQuad:
    return "driver_indexed_quad";
  case FunctionalCase::kDriverTexturedTriangles:
    return "driver_textured_triangles";
  case FunctionalCase::kDriverPcoTriangles:
    return "driver_pco_triangles";
  case FunctionalCase::kNone:
    return "none";
  }
  return "unknown";
}

bool IsFillSolidFamily(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kFillSolid ||
         functional_case == FunctionalCase::kFillSolidDepthNotEqual ||
         functional_case == FunctionalCase::kFillSolidDepthNever ||
         functional_case == FunctionalCase::kFillSolidBlended ||
         functional_case == FunctionalCase::kDriverClearColor;
}

bool IsTriangleSetupFamily(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kTriangleSetup ||
         functional_case == FunctionalCase::kTriangleSetupAllCulled ||
         functional_case == FunctionalCase::kTriangleSetupHalfCulled ||
         functional_case == FunctionalCase::kDriverTriangleSolid ||
         functional_case == FunctionalCase::kDriverIndexedQuad;
}

bool IsAttributeFetchFamily(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kAttributeFetchShader ||
         functional_case == FunctionalCase::kAttributeFetchShaderTwoAttribute ||
         functional_case == FunctionalCase::kAttributeFetchShaderFourAttribute ||
         functional_case == FunctionalCase::kAttributeFetchShaderEightAttribute;
}

bool IsVaryingsFamily(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kVaryingsShaderOne ||
         functional_case == FunctionalCase::kVaryingsShaderTwo ||
         functional_case == FunctionalCase::kVaryingsShaderFour ||
         functional_case == FunctionalCase::kVaryingsShaderEight;
}

bool IsTextureFamily(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kFillTexNearest ||
         functional_case == FunctionalCase::kFillTexBilinear ||
         functional_case == FunctionalCase::kFillTexTrilinearLinear01 ||
         functional_case == FunctionalCase::kFillTexTrilinearLinear04 ||
         functional_case == FunctionalCase::kFillTexTrilinearLinear05 ||
         functional_case == FunctionalCase::kDriverTexturedTriangles;
}

bool IsDriverPcoTrianglesCase(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kDriverPcoTriangles;
}

bool UsesTextureSampling(FunctionalCase functional_case) {
  return IsTextureFamily(functional_case);
}

bool UsesTextureSampling(const PipelineState &state) {
  return UsesTextureSampling(state, ShaderStage::kVertex) ||
         UsesTextureSampling(state, ShaderStage::kFragment);
}

bool UsesTextureSampling(const PipelineState &state, ShaderStage stage) {
  if (!IsDriverPcoTrianglesCase(state.functional_case)) {
    return stage == ShaderStage::kFragment &&
           UsesTextureSampling(state.functional_case);
  }
  return stage == ShaderStage::kVertex
             ? state.vertex_sampled_texture_count != 0
             : state.sampled_texture_count != 0;
}

bool UsesShaderVaryings(FunctionalCase functional_case) {
  return IsVaryingsFamily(functional_case) || IsTextureFamily(functional_case);
}

bool UsesShaderVaryings(const PipelineState &state) {
  if (!IsDriverPcoTrianglesCase(state.functional_case))
    return UsesShaderVaryings(state.functional_case);
  return state.varying_output_count != 0 ||
         state.fragment_varying_count != 0;
}

std::uint32_t VaryingVectorCount(FunctionalCase functional_case) {
  switch (functional_case) {
  case FunctionalCase::kVaryingsShaderOne:
    return 1;
  case FunctionalCase::kVaryingsShaderTwo:
    return 2;
  case FunctionalCase::kVaryingsShaderFour:
    return 4;
  case FunctionalCase::kVaryingsShaderEight:
    return 8;
  case FunctionalCase::kFillTexNearest:
  case FunctionalCase::kFillTexBilinear:
  case FunctionalCase::kFillTexTrilinearLinear01:
  case FunctionalCase::kFillTexTrilinearLinear04:
  case FunctionalCase::kFillTexTrilinearLinear05:
  case FunctionalCase::kDriverTexturedTriangles:
    return 1;
  default:
    return 0;
  }
}

std::uint32_t VaryingVectorCount(const PipelineState &state) {
  if (!IsDriverPcoTrianglesCase(state.functional_case))
    return VaryingVectorCount(state.functional_case);
  if (!UsesShaderVaryings(state))
    return 0;
  return (state.varying_output_count +
          kVaryingVectorComponentCount - 1U) /
         kVaryingVectorComponentCount;
}

std::uint32_t VaryingCoefficientSetCount(FunctionalCase functional_case) {
  if (IsTextureFamily(functional_case))
    return kFillTexNearestCoefficientSetCount;
  const std::uint32_t vectors = VaryingVectorCount(functional_case);
  return vectors == 0 ? 0 : 1 + vectors * kVaryingVectorComponentCount;
}

std::uint32_t VaryingCoefficientSetCount(const PipelineState &state) {
  if (!IsDriverPcoTrianglesCase(state.functional_case))
    return VaryingCoefficientSetCount(state.functional_case);
  const std::uint32_t dwords = VaryingCoefficientDwordCount(state);
  return dwords == 0 || dwords % kCoefficientSetDwordCount != 0
             ? 0
             : dwords / kCoefficientSetDwordCount;
}

std::uint32_t VaryingCoefficientDwordCount(FunctionalCase functional_case) {
  return VaryingCoefficientSetCount(functional_case) *
         kCoefficientSetDwordCount;
}

std::uint32_t VaryingCoefficientDwordCount(const PipelineState &state) {
  if (!IsDriverPcoTrianglesCase(state.functional_case))
    return VaryingCoefficientDwordCount(state.functional_case);
  if (!UsesShaderVaryings(state) || state.fragment_position_start != 0 ||
      state.fragment_position_count != kCoefficientSetDwordCount ||
      state.fragment_varying_start != state.fragment_position_count ||
      state.fragment_varying_count >
          std::numeric_limits<std::uint32_t>::max() -
              state.fragment_position_count) {
    return 0;
  }
  return state.fragment_position_count + state.fragment_varying_count;
}

std::uint32_t VaryingVertexOutputDwordCount(FunctionalCase functional_case) {
  if (IsTextureFamily(functional_case))
    return kFillTexNearestVertexOutputDwordCount;
  const std::uint32_t vectors = VaryingVectorCount(functional_case);
  return vectors == 0 ? 0 : 4 + vectors * kVaryingVectorComponentCount;
}

std::uint32_t VaryingVertexOutputDwordCount(const PipelineState &state) {
  if (!IsDriverPcoTrianglesCase(state.functional_case))
    return VaryingVertexOutputDwordCount(state.functional_case);
  if (!UsesShaderVaryings(state) || state.position_output_start != 0 ||
      state.position_output_count != 4 ||
      state.varying_output_start != state.position_output_count ||
      state.varying_output_count >
          std::numeric_limits<std::uint32_t>::max() -
              state.position_output_count) {
    return 0;
  }
  return state.position_output_count + state.varying_output_count;
}

bool IsExactVaryingBinding(FunctionalCase functional_case,
                           const ShaderVaryingBinding &binding,
                           std::size_t binding_index) {
  const std::uint32_t vectors = VaryingVectorCount(functional_case);
  if (vectors == 0 || binding_index >= vectors)
    return false;
  const std::uint32_t component_count =
      IsTextureFamily(functional_case)
          ? 2U
          : kVaryingVectorComponentCount;
  const std::uint32_t component_offset =
      static_cast<std::uint32_t>(binding_index) *
      component_count;
  return binding.vertex_output_base == 4 + component_offset &&
         binding.coefficient_set_base == 1 + component_offset &&
         binding.w_coefficient_set == 0 &&
         binding.component_count == component_count &&
         binding.interpolation == InterpolationMode::kSmooth &&
         binding.reserved[0] == 0 && binding.reserved[1] == 0;
}

bool IsExactVaryingBinding(const PipelineState &state,
                           const ShaderVaryingBinding &binding,
                           std::size_t binding_index) {
  if (!IsDriverPcoTrianglesCase(state.functional_case)) {
    return IsExactVaryingBinding(state.functional_case, binding,
                                 binding_index);
  }
  const std::uint32_t components = state.varying_output_count;
  const std::uint32_t binding_count = VaryingVectorCount(state);
  if (components == 0 || binding_count == 0 ||
      binding_index >= binding_count ||
      components > kDriverPcoMaximumVaryingComponents)
    return false;

  const std::uint32_t component_offset =
      static_cast<std::uint32_t>(binding_index) *
      kVaryingVectorComponentCount;
  const std::uint32_t binding_components =
      std::min(kVaryingVectorComponentCount, components - component_offset);
  return
         state.position_output_start == 0 &&
         state.position_output_count == 4 &&
         state.varying_output_start == 4 &&
         state.fragment_position_start == 0 &&
         state.fragment_position_count == kCoefficientSetDwordCount &&
         state.fragment_varying_start == kCoefficientSetDwordCount &&
         state.fragment_varying_count ==
             components * kCoefficientSetDwordCount &&
         state.vertex_pco_abi.vertex_outputs == 4 + components &&
         state.fragment_pco_abi.coefficients ==
             kCoefficientSetDwordCount +
                 components * kCoefficientSetDwordCount &&
         binding.vertex_output_base ==
             state.varying_output_start + component_offset &&
         binding.coefficient_set_base ==
             state.fragment_varying_start / kCoefficientSetDwordCount +
                 component_offset &&
         binding.w_coefficient_set ==
             state.fragment_position_start / kCoefficientSetDwordCount &&
         binding.component_count == binding_components &&
         binding.interpolation == InterpolationMode::kSmooth &&
         binding.reserved[0] == 0 && binding.reserved[1] == 0;
}

bool IsIndexedTriangleRasterCase(FunctionalCase functional_case) {
  return IsTriangleSetupFamily(functional_case) ||
         IsAttributeFetchFamily(functional_case) ||
         IsVaryingsFamily(functional_case);
}

bool RequiresBackCcwFaceCull(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kTriangleSetupAllCulled ||
         functional_case == FunctionalCase::kTriangleSetupHalfCulled ||
         IsAttributeFetchFamily(functional_case);
}

bool IsSolidColorRasterCase(FunctionalCase functional_case) {
  return IsFillSolidFamily(functional_case) ||
         IsTriangleSetupFamily(functional_case) ||
         IsAttributeFetchFamily(functional_case) ||
         IsDriverPcoTrianglesCase(functional_case);
}

bool IsRasterFunctionalCase(FunctionalCase functional_case) {
  return IsSolidColorRasterCase(functional_case) ||
         IsVaryingsFamily(functional_case) || IsTextureFamily(functional_case);
}

const char *PipelineStageName(PipelineStage stage) {
  switch (stage) {
  case PipelineStage::kSubmitted:
    return "submitted";
  case PipelineStage::kVdmComplete:
    return "vdm-complete";
  case PipelineStage::kVertexFetched:
    return "vertex-fetched";
  case PipelineStage::kVertexPdsReady:
    return "vertex-pds-ready";
  case PipelineStage::kVertexDecoded:
    return "vertex-decoded";
  case PipelineStage::kVertexIssued:
    return "vertex-issued";
  case PipelineStage::kVertexTexturePending:
    return "vertex-texture-pending";
  case PipelineStage::kVertexTextureSamplesReady:
    return "vertex-texture-samples-ready";
  case PipelineStage::kVertexShaded:
    return "vertex-shaded";
  case PipelineStage::kClipCullComplete:
    return "clip-cull-complete";
  case PipelineStage::kTiled:
    return "tiled";
  case PipelineStage::kParameterBufferReady:
    return "parameter-buffer-ready";
  case PipelineStage::kFragmentDecoded:
    return "fragment-decoded";
  case PipelineStage::kTilesScheduled:
    return "tiles-scheduled";
  case PipelineStage::kVisibilityReady:
    return "visibility-ready";
  case PipelineStage::kFragmentsReady:
    return "fragments-ready";
  case PipelineStage::kPdsReady:
    return "pds-ready";
  case PipelineStage::kFragmentIssued:
    return "fragment-issued";
  case PipelineStage::kFragmentTexturePending:
    return "fragment-texture-pending";
  case PipelineStage::kTextureSamplesReady:
    return "texture-samples-ready";
  case PipelineStage::kFragmentShaded:
    return "fragment-shaded";
  case PipelineStage::kTextureComplete:
    return "texture-complete";
  case PipelineStage::kPbeComplete:
    return "pbe-complete";
  case PipelineStage::kPixelDataMasterComplete:
    return "pixel-data-master-complete";
  case PipelineStage::kSlcComplete:
    return "slc-complete";
  case PipelineStage::kFramebufferReady:
    return "framebuffer-ready";
  }
  return "unknown";
}

void RequireStage(PipelineStage actual, PipelineStage expected,
                  const char *module_name) {
  if (actual == expected)
    return;
  std::ostringstream message;
  message << module_name << " expected pipeline stage "
          << PipelineStageName(expected) << ", got "
          << PipelineStageName(actual);
  throw std::runtime_error(message.str());
}

void ReleaseFunctionalPayloads(MemoryPool &pool, const PipelineState &state) {
  std::vector<PoolHandle> released;
  const auto same_handle = [](PoolHandle left, PoolHandle right) {
    return left.slot == right.slot && left.generation == right.generation;
  };
  const auto release_unique = [&](PoolHandle handle) {
    if (!HasPoolHandle(handle))
      return;
    for (const PoolHandle prior : released) {
      if (same_handle(prior, handle))
        return;
    }
    pool.Release(handle);
    released.push_back(handle);
  };

  // Resource-table entries uniquely own their bulk VBO payload. Retire those
  // nested payloads before releasing the table that describes them.
  if (HasPoolHandle(state.vertex_buffer_resources)) {
    const std::vector<VertexBufferResource> resources =
        LoadArray<VertexBufferResource>(pool, state.vertex_buffer_resources);
    for (const VertexBufferResource &resource : resources)
      release_unique(resource.data);
  }
  if (HasPoolHandle(state.texture_resources)) {
    const std::vector<TextureResource> resources =
        LoadArray<TextureResource>(pool, state.texture_resources);
    for (const TextureResource &resource : resources)
      release_unique(resource.data);
  }
  if (HasPoolHandle(state.vertex_texture_resources)) {
    const std::vector<TextureResource> resources = LoadArray<TextureResource>(
        pool, state.vertex_texture_resources);
    for (const TextureResource &resource : resources)
      release_unique(resource.data);
  }

  const PoolHandle handles[] = {
      state.drawlist_stats,
      state.vertex_buffer_resources,
      state.vertex_attribute_bindings,
      state.vertex_indices,
      state.vertex_lanes,
      state.vertex_lane_refs,
      state.vertex_shared_registers,
      state.shader_varying_bindings,
      state.vertex_texture_resources,
      state.vertex_sampler_states,
      state.texture_resources,
      state.sampler_states,
      state.fragment_shared_registers,
      state.vertex_code,
      state.vertex_instructions,
      state.fragment_code,
      state.fragment_instructions,
      state.raster_triangles,
      state.raster_vertex_outputs,
      state.tile_records,
      state.tile_primitive_refs,
      state.parameter_triangles,
      state.parameter_coefficients,
      state.fragment_candidates,
      state.color_attachment_load,
      state.depth_attachment_load,
      state.isp_depth_attachment,
      state.depth_attachment,
      state.fragment_invocations,
      state.fragment_shader_lanes,
      state.fragment_quads,
      state.usc_fragment_tasks,
      state.usc_coefficient_banks,
      state.texture_sample_requests,
      state.texture_sample_responses,
      state.vertex_continuations,
      state.fragment_continuations,
      state.fragment_outputs,
      state.pbe_framebuffer,
      state.slc_writeback_lines,
      state.dram_framebuffer,
  };
  for (const PoolHandle handle : handles)
    release_unique(handle);
}

std::size_t GetComponentTypeBytes(VertexComponentType type) {
  switch (type) {
    case VertexComponentType::kInt8:
    case VertexComponentType::kUint8:
      return 1;
    case VertexComponentType::kInt16:
    case VertexComponentType::kUint16:
    case VertexComponentType::kHalfFloat:
      return 2;
    case VertexComponentType::kFloat32:
    case VertexComponentType::kInt32:
    case VertexComponentType::kUint32:
      return 4;
    default:
      return 0;
  }
}

} // namespace pvrgpu::stub
