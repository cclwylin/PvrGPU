// Shared functional payload helpers. PipelineStageName is used for fail-fast
// module diagnostics; ReleaseFunctionalPayloads owns the complete MemoryPool
// lifetime join at the PBE/JSON reporter boundary.
// 中文：集中處理 case mapping、stage 診斷及所有 MemoryPool handle 的釋放。
#include "common/functional_types.h"

#include "common/pipeline_state.h"
#include "common/stream_output_types.h"
#include "common/tessellation_state.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace pvrgpu::stub {

std::int64_t QuantizeRasterSubpixel(float value) {
  const double scaled = static_cast<double>(value) * kSubpixelScale;
  if (!std::isfinite(scaled))
    throw std::overflow_error("raster fixed-point coordinate is non-finite");

  const double lower = std::floor(scaled);
  const double fraction = scaled - lower;
  double rounded = lower;
  if (fraction > 0.5 ||
      (fraction == 0.5 && std::fmod(std::fabs(lower), 2.0) == 1.0)) {
    rounded = lower + 1.0;
  }
  // INT64_MAX is not exactly representable as double: converting it yields
  // +2^63. Use a half-open upper bound so +2^63 cannot reach the
  // out-of-range double-to-integer conversion below; -2^63 remains valid.
  if (rounded < -0x1p63 || rounded >= 0x1p63) {
    throw std::overflow_error("raster fixed-point coordinate overflow");
  }
  return static_cast<std::int64_t>(rounded);
}

float SrgbChannelToLinear(std::uint8_t encoded) {
  const double value = static_cast<double>(encoded) / 255.0;
  const double linear = value <= 0.04045
                            ? value / 12.92
                            : std::pow((value + 0.055) / 1.055, 2.4);
  return static_cast<float>(linear);
}

std::uint8_t LinearChannelToSrgbUnorm8(float linear) {
  const double value = std::clamp(static_cast<double>(linear), 0.0, 1.0);
  const double encoded = value <= 0.0031308
                             ? value * 12.92
                             : 1.055 * std::pow(value, 1.0 / 2.4) - 0.055;
  const double scaled = std::clamp(encoded, 0.0, 1.0) * 255.0;
  return static_cast<std::uint8_t>(std::floor(scaled + 0.5));
}

std::size_t DepthAttachmentBytesPerPixel(std::uint32_t format) {
  if (format == kDriverPcoDepthFormatZ16Unorm)
    return sizeof(std::uint16_t);
  if (format == kDriverPcoDepthFormatZ24X8Unorm ||
      format == kDriverPcoDepthFormatZ24UnormS8Uint ||
      format == kDriverPcoDepthFormatZ32Unorm ||
      format == kDriverPcoDepthFormatZ32Float) {
    return sizeof(std::uint32_t);
  }
  if (format == kDriverPcoDepthFormatZ32FloatS8X24Uint)
    return 2U * sizeof(std::uint32_t);
  throw std::runtime_error("unsupported native depth attachment format");
}

std::uint32_t EncodeDepthAttachmentUnorm(float depth,
                                         std::uint32_t format) {
  if (!std::isfinite(depth) || depth < 0.0F || depth > 1.0F)
    throw std::runtime_error("native depth value is outside [0, 1]");
  if (format == kDriverPcoDepthFormatZ32Float ||
      format == kDriverPcoDepthFormatZ32FloatS8X24Uint) {
    // Nonnegative IEEE binary32 words compare in the same order as floats.
    // Canonicalize signed zero so the integer depth comparator also treats
    // -0 and +0 as equal; every other representable bit is retained exactly.
    if (depth == 0.0F)
      return 0;
    std::uint32_t encoded = 0;
    std::memcpy(&encoded, &depth, sizeof(encoded));
    return encoded;
  }
  const std::uint64_t maximum =
      format == kDriverPcoDepthFormatZ16Unorm
          ? UINT64_C(0xffff)
          : (format == kDriverPcoDepthFormatZ24X8Unorm ||
             format == kDriverPcoDepthFormatZ24UnormS8Uint)
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
  if (format == kDriverPcoDepthFormatZ32Float ||
      format == kDriverPcoDepthFormatZ32FloatS8X24Uint) {
    float depth = 0.0F;
    std::memcpy(&depth, &encoded, sizeof(depth));
    if (!std::isfinite(depth) || depth < 0.0F || depth > 1.0F)
      throw std::runtime_error("native floating-point depth is outside [0, 1]");
    return depth == 0.0F ? 0.0F : depth;
  }
  const std::uint32_t maximum =
      format == kDriverPcoDepthFormatZ16Unorm
          ? UINT32_C(0xffff)
          : (format == kDriverPcoDepthFormatZ24X8Unorm ||
             format == kDriverPcoDepthFormatZ24UnormS8Uint)
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

std::uint8_t ApplyStencilOp(StencilOp op, std::uint8_t stored,
                            std::uint8_t reference) {
  switch (op) {
    case StencilOp::kKeep:
      return stored;
    case StencilOp::kZero:
      return 0;
    case StencilOp::kReplace:
      return reference;
    case StencilOp::kIncrementClamp:
      return stored == 0xFFU ? stored : static_cast<std::uint8_t>(stored + 1U);
    case StencilOp::kDecrementClamp:
      return stored == 0 ? stored : static_cast<std::uint8_t>(stored - 1U);
    case StencilOp::kInvert:
      return static_cast<std::uint8_t>(~stored);
    case StencilOp::kIncrementWrap:
      return static_cast<std::uint8_t>(stored + 1U);
    case StencilOp::kDecrementWrap:
      return static_cast<std::uint8_t>(stored - 1U);
  }
  throw std::runtime_error("unsupported stencil operation");
}

bool StencilPass(DepthCompareOp op, std::uint8_t reference,
                 std::uint8_t stored) {
  switch (op) {
    case DepthCompareOp::kNever:
      return false;
    case DepthCompareOp::kLess:
      return reference < stored;
    case DepthCompareOp::kEqual:
      return reference == stored;
    case DepthCompareOp::kLessOrEqual:
      return reference <= stored;
    case DepthCompareOp::kGreater:
      return reference > stored;
    case DepthCompareOp::kNotEqual:
      return reference != stored;
    case DepthCompareOp::kGreaterOrEqual:
      return reference >= stored;
    case DepthCompareOp::kAlways:
      return true;
  }
  throw std::runtime_error("unsupported stencil compare operation");
}

bool DepthAttachmentHasStencil(std::uint32_t format) {
  return format == kDriverPcoDepthFormatZ24UnormS8Uint ||
         format == kDriverPcoDepthFormatZ32FloatS8X24Uint;
}

std::vector<std::uint32_t> DecodeDepthAttachmentUnormBytes(
    const std::vector<std::uint8_t> &bytes, std::uint32_t format,
    std::vector<std::uint8_t> *stencil) {
  const std::size_t bytes_per_pixel = DepthAttachmentBytesPerPixel(format);
  if (bytes.empty() || bytes.size() % bytes_per_pixel != 0)
    throw std::runtime_error("native depth attachment byte count is invalid");
  const bool has_stencil = DepthAttachmentHasStencil(format);
  std::vector<std::uint32_t> encoded(bytes.size() / bytes_per_pixel, 0);
  if (stencil)
    stencil->assign(encoded.size(), 0);
  for (std::size_t pixel = 0; pixel < encoded.size(); ++pixel) {
    std::uint32_t word = 0;
    const std::size_t offset = pixel * bytes_per_pixel;
    for (std::size_t byte = 0; byte < std::min(bytes_per_pixel, sizeof(word)); ++byte)
      word |= static_cast<std::uint32_t>(bytes[offset + byte]) << (byte * 8U);
    if (has_stencil) {
      const std::size_t stencil_byte =
          format == kDriverPcoDepthFormatZ32FloatS8X24Uint ? 4U : 3U;
      if (stencil)
        (*stencil)[pixel] = bytes[offset + stencil_byte];
      if (format == kDriverPcoDepthFormatZ24UnormS8Uint)
        word &= UINT32_C(0x00ffffff);
    }
    if ((format == kDriverPcoDepthFormatZ32Float ||
         format == kDriverPcoDepthFormatZ32FloatS8X24Uint) &&
        word == UINT32_C(0x80000000))
      word = 0;
    encoded[pixel] = word;
    (void)DecodeDepthAttachmentUnorm(encoded[pixel], format);
  }
  return encoded;
}

std::vector<std::uint8_t> EncodeDepthAttachmentUnormBytes(
    const std::vector<std::uint32_t> &encoded, std::uint32_t format,
    const std::vector<std::uint8_t> *stencil) {
  const std::size_t bytes_per_pixel = DepthAttachmentBytesPerPixel(format);
  if (encoded.empty() ||
      encoded.size() >
          std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
    throw std::runtime_error("native depth attachment value count is invalid");
  }
  const bool has_stencil = DepthAttachmentHasStencil(format);
  if (stencil && (!has_stencil || stencil->size() != encoded.size())) {
    throw std::runtime_error(
        "native depth attachment stencil plane does not match its depth plane");
  }
  std::vector<std::uint8_t> bytes(encoded.size() * bytes_per_pixel, 0);
  for (std::size_t pixel = 0; pixel < encoded.size(); ++pixel) {
    (void)DecodeDepthAttachmentUnorm(encoded[pixel], format);
    std::uint32_t word = encoded[pixel];
    if ((format == kDriverPcoDepthFormatZ32Float ||
         format == kDriverPcoDepthFormatZ32FloatS8X24Uint) &&
        word == UINT32_C(0x80000000))
      word = 0;
    const std::size_t offset = pixel * bytes_per_pixel;
    for (std::size_t byte = 0; byte < std::min(bytes_per_pixel, sizeof(word)); ++byte)
      bytes[offset + byte] = static_cast<std::uint8_t>(word >> (byte * 8U));
    if (has_stencil && stencil) {
      const std::size_t stencil_byte =
          format == kDriverPcoDepthFormatZ32FloatS8X24Uint ? 4U : 3U;
      bytes[offset + stencil_byte] = (*stencil)[pixel];
    }
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
         UsesTextureSampling(state, ShaderStage::kGeometry) ||
         UsesTextureSampling(state, ShaderStage::kFragment);
}

bool UsesTextureSampling(const PipelineState &state, ShaderStage stage) {
  if (!IsDriverPcoTrianglesCase(state.functional_case)) {
    return stage == ShaderStage::kFragment &&
           UsesTextureSampling(state.functional_case);
  }
  switch (stage) {
    case ShaderStage::kVertex: return state.vertex_sampled_texture_count != 0;
    case ShaderStage::kGeometry: return state.geometry_sampled_texture_count != 0;
    case ShaderStage::kFragment: return state.sampled_texture_count != 0;
    default: return false;
  }
}

bool UsesShaderVaryings(FunctionalCase functional_case) {
  return IsVaryingsFamily(functional_case) || IsTextureFamily(functional_case);
}

bool UsesShaderVaryings(const PipelineState &state) {
  if (!IsDriverPcoTrianglesCase(state.functional_case))
    return UsesShaderVaryings(state.functional_case);
  // A texture shader can address texels from gl_FragCoord or constants with
  // no user varying. Its declared position coefficient set still traverses
  // ParameterBuffer/PDS; an empty linkage is not an absent position plane.
  return state.varying_output_count != 0 ||
         state.fragment_varying_count != 0 ||
         UsesTextureSampling(state, ShaderStage::kFragment);
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
  if (state.driver_varying_bindings_explicit)
    return state.driver_varying_binding_count;
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
  /*
   * gl_PointSize sits between the position and the varyings when the shader
   * writes one, so the varyings do not always start at the end of the
   * position.  Requiring that they do reported a span of zero for every
   * point-sized shader, and the linkage check then refused its own ABI.
   */
  const std::uint32_t expected_varying_start =
      state.position_output_count + state.raster_state.point_size_output_count;
  if (!UsesShaderVaryings(state) || state.position_output_start != 0 ||
      state.position_output_count != 4 ||
      state.varying_output_start != expected_varying_start ||
      state.varying_output_count >
          std::numeric_limits<std::uint32_t>::max() -
              state.varying_output_start) {
    return 0;
  }
  return state.varying_output_start + state.varying_output_count;
}

std::uint32_t ActiveVertexOutputDwordCount(const PipelineState &state) {
  std::uint32_t dwords =
      UsesShaderVaryings(state) ? VaryingVertexOutputDwordCount(state) : 4U;
  if (state.raster_state.point_size_output_count != 0) {
    dwords = std::max(dwords, state.raster_state.point_size_output_start +
                                  state.raster_state.point_size_output_count);
  }
  if (HasPoolHandle(state.geometry_code)) {
    // GS keeps PrimitiveID/Layer after its packed user varyings. These raw
    // outputs still need transport when the fragment shader does not read
    // them as varyings; the raster boundary consumes their declared slots.
    for (const auto &range : {
             std::pair{state.geometry_primitive_id_output_start,
                        state.geometry_primitive_id_output_count},
             std::pair{state.geometry_layer_output_start,
                        state.geometry_layer_output_count}}) {
      if (!range.second) continue;
      if (range.first > kPcoVertexOutputRegisterCount ||
          range.second > kPcoVertexOutputRegisterCount - range.first)
        return kPcoVertexOutputRegisterCount + 1;
      dwords = std::max(dwords, range.first + range.second);
    }
  }
  return dwords;
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
                           std::size_t binding_index,
                           const char **out_refusal) {
  if (out_refusal)
    *out_refusal = nullptr;
  if (!IsDriverPcoTrianglesCase(state.functional_case)) {
    const bool exact =
        IsExactVaryingBinding(state.functional_case, binding, binding_index);
    if (!exact && out_refusal)
      *out_refusal = "fixture_profile_layout";
    return exact;
  }
  const std::uint32_t components = state.varying_output_count;
  const std::uint32_t binding_count = VaryingVectorCount(state);
  if (state.driver_varying_bindings_explicit) {
    const auto coefficient_count = state.fragment_pco_abi.coefficients / 4;
    const auto output_dwords = HasPoolHandle(state.tessellation_state)
        ? state.tessellation_output_dwords : HasPoolHandle(state.geometry_code)
        ? state.geometry_pco_abi.vertex_outputs : state.vertex_pco_abi.vertex_outputs;
    if (HasPoolHandle(state.geometry_code) &&
        binding.interpolation != InterpolationMode::kFlat) {
      for (const auto &range : {
               std::pair{state.geometry_primitive_id_output_start,
                         state.geometry_primitive_id_output_count},
               std::pair{state.geometry_layer_output_start,
                         state.geometry_layer_output_count}}) {
        if (range.second && std::uint64_t(binding.vertex_output_base) <
                                std::uint64_t(range.first) + range.second &&
            std::uint64_t(range.first) <
                std::uint64_t(binding.vertex_output_base) + binding.component_count) {
          if (out_refusal) *out_refusal = "geometry_integer_varying_not_flat";
          return false;
        }
      }
    }
    const bool valid = binding_index < binding_count && binding_count <= 64 &&
        binding.component_count >= 1 && binding.component_count <= 4 &&
        binding.vertex_output_base >= state.varying_output_start &&
        binding.vertex_output_base <= output_dwords &&
        binding.component_count <= output_dwords - binding.vertex_output_base &&
        binding.coefficient_set_base >= 1 && binding.coefficient_set_base <= coefficient_count &&
        binding.component_count <= coefficient_count - binding.coefficient_set_base &&
        binding.w_coefficient_set == 0 &&
        (binding.interpolation == InterpolationMode::kSmooth || binding.interpolation == InterpolationMode::kFlat) &&
        !binding.reserved[0] && !binding.reserved[1];
    if (!valid && out_refusal) *out_refusal = "explicit_varying_binding";
    return valid;
  }
  if (components == 0 || binding_count == 0 ||
      binding_index >= binding_count ||
      components > kDriverPcoMaximumVaryingComponents) {
    if (out_refusal)
      *out_refusal = "varying_component_range";
    return false;
  }

  const std::uint32_t component_offset =
      static_cast<std::uint32_t>(binding_index) *
      kVaryingVectorComponentCount;
  const std::uint32_t binding_components =
      std::min(kVaryingVectorComponentCount, components - component_offset);
  /*
   * gl_PointSize sits between the position and the varyings, so the varyings
   * do not always start at dword four.  Reading that as a constant rejected
   * every shader that writes a point size.
   */
  const std::uint32_t expected_varying_start =
      state.position_output_count + state.raster_state.point_size_output_count;
  const char *refusal = nullptr;
  if (state.position_output_start != 0)
    refusal = "position_output_start";
  else if (state.position_output_count != 4)
    refusal = "position_output_count";
  else if (state.varying_output_start != expected_varying_start)
    refusal = "varying_output_start";
  else if (state.fragment_position_start != 0)
    refusal = "fragment_position_start";
  else if (state.fragment_position_count != kCoefficientSetDwordCount)
    refusal = "fragment_position_count";
  else if (state.fragment_varying_start != kCoefficientSetDwordCount)
    refusal = "fragment_varying_start";
  else if (state.fragment_varying_count !=
           components * kCoefficientSetDwordCount)
    refusal = "fragment_varying_count";
  else if ((HasPoolHandle(state.tessellation_state)
                ? state.tessellation_output_dwords : HasPoolHandle(state.geometry_code)
                ? state.geometry_pco_abi.vertex_outputs
                : state.vertex_pco_abi.vertex_outputs) !=
           ActiveVertexOutputDwordCount(state))
    refusal = "vertex_outputs";
  else if (state.fragment_pco_abi.coefficients !=
           kCoefficientSetDwordCount + components * kCoefficientSetDwordCount)
    refusal = "fragment_coefficients";
  else if (binding.vertex_output_base !=
           state.varying_output_start + component_offset)
    refusal = "vertex_output_base";
  else if (binding.coefficient_set_base !=
           state.fragment_varying_start / kCoefficientSetDwordCount +
               component_offset)
    refusal = "coefficient_set_base";
  else if (binding.w_coefficient_set !=
           state.fragment_position_start / kCoefficientSetDwordCount)
    refusal = "w_coefficient_set";
  else if (binding.component_count != binding_components)
    refusal = "component_count";
  /*
   * What this gate guards is the layout: which vertex outputs and coefficient
   * sets a varying owns.  The interpolation mode is not part of that -- the
   * capsule states it per varying, and a flat one is as valid as a smooth one.
   */
  else if (binding.interpolation != InterpolationMode::kSmooth &&
           binding.interpolation != InterpolationMode::kFlat)
    refusal = "interpolation";
  else if (binding.reserved[0] != 0 || binding.reserved[1] != 0)
    refusal = "reserved";
  if (out_refusal)
    *out_refusal = refusal;
  return refusal == nullptr;
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
  case PipelineStage::kGeometryTexturePending:
    return "geometry-texture-pending";
  case PipelineStage::kGeometryTextureSamplesReady:
    return "geometry-texture-samples-ready";
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
  if (HasPoolHandle(state.geometry_texture_resources)) {
    for (const auto &resource : LoadArray<TextureResource>(pool, state.geometry_texture_resources))
      release_unique(resource.data);
  }

  if (HasPoolHandle(state.tessellation_state)) {
    const auto tess = LoadArray<TessellationState>(pool, state.tessellation_state);
    if (tess.size() != 1)
      throw std::runtime_error("Tessellation state ownership extent is invalid");
    for (const auto handle : {tess[0].control_code, tess[0].control_instructions,
         tess[0].control_shared, tess[0].control_uniform_buffers,
         tess[0].evaluation_code, tess[0].evaluation_instructions,
         tess[0].evaluation_shared, tess[0].evaluation_uniform_buffers,
         tess[0].patches, tess[0].domain_points, tess[0].domain_indices})
      release_unique(handle);
    release_unique(state.tessellation_state);
  }
  if (HasPoolHandle(state.stream_output_targets)) {
    for (const auto &target : LoadArray<StreamOutputTarget>(pool, state.stream_output_targets))
      release_unique(target.readback);
  }
  const PoolHandle handles[] = {
      state.drawlist_stats,
      state.vertex_buffer_resources,
      state.vertex_attribute_bindings,
      state.vertex_indices,
      state.vertex_lanes,
      state.vertex_lane_refs,
      state.stream_output_bindings,
      state.stream_output_targets,
      state.geometry_input_primitives,
      state.geometry_primitives,
      state.geometry_code,
      state.geometry_instructions,
      state.geometry_shared_registers,
      state.geometry_uniform_buffer_resources,
      state.expanded_source_vertices,
      state.vertex_shared_registers,
      state.vertex_uniform_buffer_resources,
      state.fragment_uniform_buffer_resources,
      state.shader_varying_bindings,
      state.vertex_texture_resources,
      state.vertex_sampler_states,
      state.geometry_texture_resources,
      state.geometry_sampler_states,
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
      state.isp_stencil_attachment,
      state.attachment_clears,
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
      /* The colour attachments past the first, read back alongside it. */
      state.extra_dram_framebuffer[0],
      state.extra_dram_framebuffer[1],
      state.extra_dram_framebuffer[2],
  };
  static_assert(kMaxRenderTargets == 4,
                "every extra colour attachment must be listed for release");
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
