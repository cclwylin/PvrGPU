// Shared functional payload helpers. PipelineStageName is used for fail-fast
// module diagnostics; ReleaseFunctionalPayloads owns the complete MemoryPool
// lifetime join at the PBE/JSON reporter boundary.
// 中文：集中處理 case mapping、stage 診斷及所有 MemoryPool handle 的釋放。
#include "common/functional_types.h"

#include "common/pipeline_state.h"

#include <sstream>

namespace pvrgpu::stub {

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
  case FunctionalCase::kNone:
    return "none";
  }
  return "unknown";
}

bool IsFillSolidFamily(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kFillSolid ||
         functional_case == FunctionalCase::kFillSolidDepthNotEqual ||
         functional_case == FunctionalCase::kFillSolidDepthNever ||
         functional_case == FunctionalCase::kFillSolidBlended;
}

bool IsTriangleSetupFamily(FunctionalCase functional_case) {
  return functional_case == FunctionalCase::kTriangleSetup ||
         functional_case == FunctionalCase::kTriangleSetupAllCulled ||
         functional_case == FunctionalCase::kTriangleSetupHalfCulled;
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
         functional_case == FunctionalCase::kFillTexTrilinearLinear05;
}

bool UsesShaderVaryings(FunctionalCase functional_case) {
  return IsVaryingsFamily(functional_case) || IsTextureFamily(functional_case);
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
    return 1;
  default:
    return 0;
  }
}

std::uint32_t VaryingCoefficientSetCount(FunctionalCase functional_case) {
  if (IsTextureFamily(functional_case))
    return kFillTexNearestCoefficientSetCount;
  const std::uint32_t vectors = VaryingVectorCount(functional_case);
  return vectors == 0 ? 0 : 1 + vectors * kVaryingVectorComponentCount;
}

std::uint32_t VaryingCoefficientDwordCount(FunctionalCase functional_case) {
  return VaryingCoefficientSetCount(functional_case) *
         kCoefficientSetDwordCount;
}

std::uint32_t VaryingVertexOutputDwordCount(FunctionalCase functional_case) {
  if (IsTextureFamily(functional_case))
    return kFillTexNearestVertexOutputDwordCount;
  const std::uint32_t vectors = VaryingVectorCount(functional_case);
  return vectors == 0 ? 0 : 4 + vectors * kVaryingVectorComponentCount;
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
         IsAttributeFetchFamily(functional_case);
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
  case PipelineStage::kVertexDecoded:
    return "vertex-decoded";
  case PipelineStage::kVertexIssued:
    return "vertex-issued";
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

  const PoolHandle handles[] = {
      state.drawlist_stats,
      state.vertex_buffer_resources,
      state.vertex_attribute_bindings,
      state.vertex_indices,
      state.vertex_lanes,
      state.vertex_lane_refs,
      state.vertex_shared_registers,
      state.shader_varying_bindings,
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
      state.fragment_invocations,
      state.fragment_shader_lanes,
      state.fragment_quads,
      state.usc_fragment_tasks,
      state.usc_coefficient_banks,
      state.texture_sample_requests,
      state.texture_sample_responses,
      state.fragment_continuations,
      state.fragment_outputs,
      state.pbe_framebuffer,
      state.slc_writeback_lines,
      state.dram_framebuffer,
  };
  for (const PoolHandle handle : handles)
    release_unique(handle);
}

} // namespace pvrgpu::stub
