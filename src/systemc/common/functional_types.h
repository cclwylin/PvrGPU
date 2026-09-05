// PvrGPU functional-pipeline payloads shared by the event-driven SystemC
// modules. FIFO (First-In, First-Out) links carry only MemoryPool handles;
// these trivially-copyable records hold the bulk vertex, index, tile, ISP and
// USC (Unified Shading Cluster) data addressed by those handles.
// 中文：此檔定義硬體 pipeline 共用的輕量 transaction/payload 格式；大型
// vertex/index/raster 資料留在 MemoryPool，不直接穿越 module FIFO。
#pragma once

#include "common/reference_uarch.h"
#include "common/shader_stage.h"
#include "memory_pool.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace pvrgpu::stub {

inline constexpr std::uint32_t kTileWidth = kReferenceUarch.tile_width;
inline constexpr std::uint32_t kTileHeight = kReferenceUarch.tile_height;
inline constexpr std::uint32_t kSubpixelBits = kReferenceUarch.subpixel_bits;
inline constexpr std::int64_t kSubpixelScale = 1LL << kSubpixelBits;

// Match llvmpipe SSE's round-to-nearest-even rule at the 24.8 boundary for
// post-clip viewport coordinates. In that bounded domain, llvmpipe's
// half-pixel translation and this model's translated sample coordinates are
// equivalent. Keep clip-area classification and parameter edge setup on one
// deterministic conversion instead of inheriting half-away library behavior.
std::int64_t QuantizeRasterSubpixel(float value);
// Keep aligned with shader/pco_iss.h's kPcoVertexInputCount; the two headers
// cannot include each other.  Sixteen bound attributes occupy four registers
// each, which is what the instruction encoding's six-bit index can name.
inline constexpr std::uint32_t kPcoVertexInputRegisterCount = 64;
// Public Rogue implementations expose at least 64 user vertex-output dwords.
// Keep this common payload limit aligned with shader/pco_iss.h's
// kPcoVertexOutputCount; the two headers cannot include each other.
inline constexpr std::uint32_t kPcoVertexOutputRegisterCount = 64;
inline constexpr std::uint32_t kCoefficientSetDwordCount = 4;
inline constexpr std::uint32_t kVaryingVectorComponentCount = 4;
inline constexpr std::uint32_t kVaryingsShaderOneCoefficientSetCount = 5;
inline constexpr std::uint32_t kVaryingsShaderOneCoefficientDwordCount =
    kCoefficientSetDwordCount * kVaryingsShaderOneCoefficientSetCount;
inline constexpr std::uint32_t kVaryingsShaderTwoCoefficientSetCount = 9;
inline constexpr std::uint32_t kVaryingsShaderTwoCoefficientDwordCount =
    kCoefficientSetDwordCount * kVaryingsShaderTwoCoefficientSetCount;
inline constexpr std::uint32_t kVaryingsShaderFourCoefficientSetCount = 17;
inline constexpr std::uint32_t kVaryingsShaderFourCoefficientDwordCount =
    kCoefficientSetDwordCount * kVaryingsShaderFourCoefficientSetCount;
inline constexpr std::uint32_t kVaryingsShaderEightCoefficientSetCount = 33;
inline constexpr std::uint32_t kVaryingsShaderEightCoefficientDwordCount =
    kCoefficientSetDwordCount * kVaryingsShaderEightCoefficientSetCount;
inline constexpr std::uint32_t kFillTexNearestCoefficientSetCount = 3;
inline constexpr std::uint32_t kFillTexNearestCoefficientDwordCount =
    kCoefficientSetDwordCount * kFillTexNearestCoefficientSetCount;
inline constexpr std::uint32_t kFillTexNearestVertexOutputDwordCount = 6;
inline constexpr std::uint32_t kFillTexNearestSharedDwordCount = 20;
inline constexpr std::uint32_t kMaximumTextureMipLevels = 15;
inline constexpr std::uint32_t kInvalidFragmentInvocationIndex =
    std::numeric_limits<std::uint32_t>::max();

enum class FunctionalCase : std::uint32_t {
  kNone = 0,
  kFillSolid = 1,
  kFillSolidDepthNotEqual = 2,
  kFillSolidDepthNever = 3,
  kFillSolidBlended = 4,
  kTriangleSetup = 5,
  kTriangleSetupAllCulled = 6,
  kTriangleSetupHalfCulled = 7,
  kAttributeFetchShader = 8,
  kAttributeFetchShaderTwoAttribute = 9,
  kAttributeFetchShaderFourAttribute = 10,
  kAttributeFetchShaderEightAttribute = 11,
  kVaryingsShaderOne = 12,
  kVaryingsShaderTwo = 13,
  kVaryingsShaderFour = 14,
  kVaryingsShaderEight = 15,
  kFillTexNearest = 16,
  kFillTexBilinear = 17,
  kFillTexTrilinearLinear01 = 18,
  kFillTexTrilinearLinear04 = 19,
  kFillTexTrilinearLinear05 = 20,
  kDriverClearColor = 21,
  kDriverTriangleSolid = 22,
  kDriverIndexedQuad = 23,
  kDriverTexturedTriangles = 24,
  kDriverPcoTriangles = 25,
};

struct PipelineState;

FunctionalCase FunctionalCaseFromName(std::string_view name);
const char *FunctionalCaseName(FunctionalCase functional_case);
bool IsFillSolidFamily(FunctionalCase functional_case);
bool IsTriangleSetupFamily(FunctionalCase functional_case);
bool IsAttributeFetchFamily(FunctionalCase functional_case);
bool IsVaryingsFamily(FunctionalCase functional_case);
bool IsTextureFamily(FunctionalCase functional_case);
bool IsDriverPcoTrianglesCase(FunctionalCase functional_case);
bool UsesTextureSampling(FunctionalCase functional_case);
bool UsesTextureSampling(const PipelineState &state);
bool UsesTextureSampling(const PipelineState &state, ShaderStage stage);
bool UsesShaderVaryings(FunctionalCase functional_case);
bool UsesShaderVaryings(const PipelineState &state);
bool IsIndexedTriangleRasterCase(FunctionalCase functional_case);
bool RequiresBackCcwFaceCull(FunctionalCase functional_case);
bool IsSolidColorRasterCase(FunctionalCase functional_case);
bool IsRasterFunctionalCase(FunctionalCase functional_case);
std::uint32_t VaryingVectorCount(FunctionalCase functional_case);
std::uint32_t VaryingVectorCount(const PipelineState &state);
std::uint32_t VaryingCoefficientSetCount(FunctionalCase functional_case);
std::uint32_t VaryingCoefficientSetCount(const PipelineState &state);
std::uint32_t VaryingCoefficientDwordCount(FunctionalCase functional_case);
std::uint32_t VaryingCoefficientDwordCount(const PipelineState &state);
std::uint32_t VaryingVertexOutputDwordCount(FunctionalCase functional_case);
std::uint32_t VaryingVertexOutputDwordCount(const PipelineState &state);

enum class PipelineStage : std::uint32_t {
  kSubmitted = 0,
  kVdmComplete,
  kVertexFetched,
  kVertexPdsReady,
  kVertexDecoded,
  kVertexIssued,
  kVertexTexturePending,
  kVertexTextureSamplesReady,
  kVertexShaded,
  kClipCullComplete,
  kTiled,
  kParameterBufferReady,
  kFragmentDecoded,
  kTilesScheduled,
  kVisibilityReady,
  kFragmentsReady,
  kPdsReady,
  kFragmentIssued,
  kFragmentTexturePending,
  kTextureSamplesReady,
  kFragmentShaded,
  kTextureComplete,
  kPbeComplete,
  kPixelDataMasterComplete,
  kSlcComplete,
  kFramebufferReady,
};

enum class PrimitiveTopology : std::uint32_t {
  kTriangleStrip = 0,
  kTriangleList = 1,
  kPoints = 2,
  kLines = 3,
  kLineStrip = 4,
  kLineLoop = 5,
  kTriangleFan = 6,
};

enum class IndexFormat : std::uint32_t {
  kNone = 0,
  kUint8 = 1,
  kUint16 = 2,
  kUint32 = 3,
};

enum class DepthCompareOp : std::uint32_t {
  kNever = 0,
  kLess,
  kEqual,
  kLessOrEqual,
  kGreater,
  kNotEqual,
  kGreaterOrEqual,
  kAlways,
};

enum class FragmentVisibility : std::uint8_t {
  kRejected = 0,
  kVisible = 1,
};

struct DrawCommand {
  PrimitiveTopology topology = PrimitiveTopology::kTriangleStrip;
  std::uint32_t first_vertex = 0;
  std::uint32_t vertex_count = 0;
  std::uint32_t first_index = 0;
  std::uint32_t index_count = 0;
  std::int32_t base_vertex = 0;
  IndexFormat index_format = IndexFormat::kNone;
};

// Per-DrawList PCO statistics keep static program composition separate
// from repeat/lane-expanded dynamic instruction executions.
struct DrawListShaderStats {
  std::uint64_t invocations = 0;
  std::uint32_t program_groups = 0;
  std::uint32_t program_instructions = 0;
  std::uint64_t program_alu_instructions = 0;
  std::uint64_t program_tex_instructions = 0;
  std::uint64_t program_memory_instructions = 0;
  std::uint64_t executed_alu_instructions = 0;
  std::uint64_t executed_tex_instructions = 0;
  std::uint64_t executed_memory_instructions = 0;
  std::uint8_t program_recorded = 0;
  std::uint8_t executions_recorded = 0;
  std::uint8_t reserved[6]{};
};

struct DrawListStats {
  std::uint32_t drawlist_index = 0;
  std::uint32_t draw_id = 0;
  DrawListShaderStats vertex;
  DrawListShaderStats fragment;
};

struct DepthState {
  DepthCompareOp compare_op = DepthCompareOp::kLess;
  float clear_depth = 1.0f;
  std::uint8_t test_enable = 0;
  std::uint8_t write_enable = 0;
  std::uint8_t reserved[2]{};
};

// Public pipe-format values are transported verbatim by the native sequence
// ABI. These helpers centralize the exact little-endian UNORM attachment
// conversion used by Submitter, ISP and FragmentFrontend.
std::size_t DepthAttachmentBytesPerPixel(std::uint32_t format);
std::uint32_t EncodeDepthAttachmentUnorm(float depth, std::uint32_t format);
float DecodeDepthAttachmentUnorm(std::uint32_t encoded,
                                 std::uint32_t format);
std::vector<std::uint32_t> DecodeDepthAttachmentUnormBytes(
    const std::vector<std::uint8_t> &bytes, std::uint32_t format);
std::vector<std::uint8_t> EncodeDepthAttachmentUnormBytes(
    const std::vector<std::uint32_t> &encoded, std::uint32_t format);

enum class BlendEquation : std::uint8_t {
  kAdd = 0,
  kSubtract = 1,
  kReverseSubtract = 2,
  kMin = 3,
  kMax = 4,
};

enum class BlendFactor : std::uint8_t {
  kZero = 0,
  kOne,
  kSourceAlpha,
  kOneMinusSourceAlpha,
  kSourceColor,
  kOneMinusSourceColor,
  kDestinationColor,
  kOneMinusDestinationColor,
  kDestinationAlpha,
  kOneMinusDestinationAlpha,
};

// GLES blend state is explicit for both RGB and alpha.  GLBench Fill.Solid
// blended uses ADD with SRC_ALPHA / ONE_MINUS_SRC_ALPHA for both paths.
struct BlendState {
  BlendEquation rgb_equation = BlendEquation::kAdd;
  BlendEquation alpha_equation = BlendEquation::kAdd;
  BlendFactor source_rgb_factor = BlendFactor::kOne;
  BlendFactor destination_rgb_factor = BlendFactor::kZero;
  BlendFactor source_alpha_factor = BlendFactor::kOne;
  BlendFactor destination_alpha_factor = BlendFactor::kZero;
  std::uint8_t enable = 0;
  std::uint8_t reserved = 0;
};

// GLES face-cull state is independent from triangle setup.  When enabled,
// the API defaults are GL_BACK and GL_CCW: clockwise triangles are back
// facing and are removed after homogeneous clipping.
enum class CullFaceMode : std::uint8_t {
  kFront = 0,
  kBack,
  kFrontAndBack,
};

enum class FrontFaceWinding : std::uint8_t {
  kClockwise = 0,
  kCounterClockwise,
};

struct FaceCullState {
  CullFaceMode mode = CullFaceMode::kBack;
  FrontFaceWinding front_face = FrontFaceWinding::kCounterClockwise;
  std::uint8_t enable = 0;
  std::uint8_t reserved = 0;
};

// Explicit state keeps HSR (Hidden Surface Removal) legality separate from
// shader decoding, so discard/depth/sample-mask/blend support can extend the
// same ISP payload contract instead of replacing it.
// The scissor rectangle in framebuffer pixels, half-open on the upper bound.
// A disabled scissor leaves rasterization bounded only by the render target.
struct ScissorState {
  std::uint8_t enable = 0;
  std::uint32_t x0 = 0;
  std::uint32_t y0 = 0;
  std::uint32_t x1 = 0;
  std::uint32_t y1 = 0;
};

struct RasterState {
  DepthState depth;
  BlendState blend;
  FaceCullState face_cull;
  ScissorState scissor;
  // Width a line or point rasterizes at, in device pixels.  GLES guarantees
  // 1.0; anything wider is widened into real screen-space geometry by
  // clip/cull rather than approximated.
  float line_width = 1.0f;
  float point_size = 1.0f;
  // Vertex output holding gl_PointSize.  A count of zero means every point
  // rasterizes at point_size instead of sizing itself.
  std::uint32_t point_size_output_start = 0;
  std::uint32_t point_size_output_count = 0;
  // Viewport transform applied to normalized device coordinates.  A draw that
  // renders to part of its attachment states a scale and offset that are not
  // half the surface, so clip/cull cannot derive them from the extent.  Zero
  // means unstated, and the full surface is used.
  float viewport_scale[3] = {0.0f, 0.0f, 0.0f};
  float viewport_translate[3] = {0.0f, 0.0f, 0.0f};
  float clear_color[4] = {0.0F, 0.0F, 0.0F, 1.0F};
  std::uint32_t sample_count = 1;
  std::uint8_t shader_may_discard = 0;
  std::uint8_t shader_writes_depth = 0;
  std::uint8_t shader_writes_sample_mask = 0;
  std::uint8_t depth_clamp_enable = 0;
  std::uint8_t color_mask = 0x0f;
};

struct InputVertex {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

// One uniquely owned MemoryPool payload represents one API vertex-buffer
// object. Attribute bindings refer to this table by index, so several shader
// inputs may read the same VBO without duplicating or aliasing its PoolHandle.
enum class VertexComponentType : std::uint8_t {
  kFloat32 = 0,
  kInt8 = 1,
  kUint8 = 2,
  kInt16 = 3,
  kUint16 = 4,
  kInt32 = 5,
  kUint32 = 6,
  kHalfFloat = 7,
};

struct VertexBufferResource {
  PoolHandle data;
  std::uint64_t gpu_address = 0;
  std::uint32_t byte_size = 0;
  std::uint32_t reserved = 0;
};

// Hardware-facing vertex input route. source_components are fetched from the
// VBO; destination_components also materialize the GLES defaults (0,0,0,1)
// into consecutive PCO VTXIN registers. Defaults are not memory traffic.
struct VertexAttributeBinding {
  std::uint32_t buffer_index = 0;
  std::uint32_t offset_bytes = 0;
  std::uint32_t stride_bytes = 0;
  std::uint16_t destination_register = 0;
  VertexComponentType component_type = VertexComponentType::kFloat32;
  std::uint8_t source_components = 0;
  std::uint8_t destination_components = 0;
  std::uint8_t normalized = 0;
  std::uint8_t integer = 0;
  std::uint8_t reserved[2]{};
  std::uint32_t instance_divisor = 0;
};

// PCO instruction execution uses raw 32-bit register contents. Float
// interpretation belongs only at the fixed-function clip/raster boundaries.
struct VertexLane {
  std::uint32_t vertex_input[kPcoVertexInputRegisterCount]{};
  std::uint32_t vertex_output[kPcoVertexOutputRegisterCount]{};
  std::uint8_t emitted = 0;
  std::uint8_t ended = 0;
  std::uint8_t reserved[2]{};
};

// One entry exists for every indexed input-assembler occurrence.  A cache hit
// may point several occurrences at one shaded lane; a cache miss owns a newly
// appended lane. 中文：每個 index occurrence 都保留 lane/vertex 對照，讓
// ClipCull 能依原始 primitive order 重組 triangle，而不是硬編 counter。
struct VertexLaneRef {
  std::uint32_t lane_index = 0;
  std::uint32_t vertex_index = 0;
};

// The vertex PDS supplies raw shared-register bits to the USC. Gate16 uses
// SH0 for GLBench's scale uniform; keeping the values in MemoryPool avoids a
// case-specific constant inside the shader executor.
struct ShaderSharedRegister {
  std::uint32_t value = 0;
};

enum class TextureFormat : std::uint8_t {
  kRgba8Unorm = 0,
  // The fourth stored byte is padding. The public image descriptor selects
  // SRC_ONE for alpha, so the texture datapath must never expose that byte.
  kRgbx8Unorm,
  // Public Rogue U32 sampled through SMP.FCNORM.  Refract binds this format
  // as a Z32_UNORM image with XXX1 swizzle; storage remains one little-endian
  // uint32 per texel and filtering is restricted to nearest.
  kZ32Unorm,
  // Depth-as-texture: a combined depth/stencil image sampled through a 2D
  // view, stored as Rogue ST8U24.  One little-endian uint32 per texel holds
  // the stencil in bits 24..31 and the depth in bits 0..23, normalized by
  // 2^24-1, exactly as the driver's clear path packs it.  GL swizzles it to
  // (depth, 0, 0, 1), and unlike kZ32Unorm it filters.
  kZ24UnormS8Uint,
};

enum class TextureLayout : std::uint8_t {
  kLinear = 0,
};

enum class TextureFilter : std::uint8_t {
  kNearest = 0,
  kLinear,
};

enum class TextureWrapMode : std::uint8_t {
  kRepeat = 0,
  kMirroredRepeat,
  kClampToEdge,
  kClampToBorder,
};

struct TextureMipLevel {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t row_pitch_bytes = 0;
  std::uint32_t offset_bytes = 0;
};

// One immutable sampled image. `data` owns the complete pre-resident mip
// allocation; TextureUnit uploads it to DRAM before issuing the first sample.
struct TextureResource {
  PoolHandle data;
  std::uint64_t gpu_address = 0;
  std::uint32_t byte_size = 0;
  std::uint8_t mip_count = 0;
  TextureFormat format = TextureFormat::kRgba8Unorm;
  TextureLayout layout = TextureLayout::kLinear;
  std::uint8_t descriptor_set = 0;
  std::uint8_t binding = 0;
  std::uint8_t reserved[3]{};
  TextureMipLevel mip[kMaximumTextureMipLevels]{};
};

struct SamplerState {
  TextureFilter min_filter = TextureFilter::kNearest;
  TextureFilter mag_filter = TextureFilter::kNearest;
  TextureFilter mip_filter = TextureFilter::kNearest;
  TextureWrapMode wrap_u = TextureWrapMode::kRepeat;
  TextureWrapMode wrap_v = TextureWrapMode::kRepeat;
  std::uint16_t min_lod_u4_6 = 0;
  std::uint16_t max_lod_u4_6 = 0;
  std::uint8_t normalized_coordinates = 1;
  std::uint8_t base_mip_level = 0;
  std::uint8_t descriptor_set = 0;
  std::uint8_t binding = 0;
  std::uint8_t reserved[2]{};
};

// Interpolation qualifier carried by the exact VS-to-FS linkage. The binding
// payload is emitted by command/shader setup and consumed independently by
// ClipCull and ParameterBuffer; no shader value is precomputed in the front
// end. FITR/FITRP execution remains in the USC ISS.
enum class InterpolationMode : std::uint8_t {
  kSmooth = 0,
  kNoPerspective,
  kFlat,
};

struct ShaderVaryingBinding {
  std::uint16_t vertex_output_base = 0;
  std::uint16_t coefficient_set_base = 0;
  std::uint16_t w_coefficient_set = 0;
  std::uint8_t component_count = 0;
  InterpolationMode interpolation = InterpolationMode::kSmooth;
  std::uint8_t reserved[2]{};
};

// The current public GLBench varying gates use contiguous smooth vec4
// linkages: VTXOUT4... feed coefficient sets after shared position-W set 0.
// Keeping the contract in one helper lets every hardware module validate the
// same producer/consumer ABI without branching on a shader name or color.
bool IsExactVaryingBinding(FunctionalCase functional_case,
                           const ShaderVaryingBinding &binding,
                           std::size_t binding_index);
bool IsExactVaryingBinding(const PipelineState &state,
                           const ShaderVaryingBinding &binding,
                           std::size_t binding_index);

struct PrimitiveKey {
  std::uint64_t submit_ordinal = 0;
  std::uint32_t draw_id = 0;
  std::uint32_t api_primitive_id = 0;
  std::uint32_t instance_id = 0;
  std::uint16_t clip_piece = 0;
  std::uint16_t layer = 0;
};

struct RasterTriangle {
  PrimitiveKey key;
  float x[3]{};
  float y[3]{};
  float window_z[3]{};
  float reciprocal_w[3]{};
  // Three consecutive vertices live in the flat raster_vertex_outputs pool
  // payload. Offsets, never nested PoolHandles, preserve single ownership.
  std::uint32_t first_vertex_output_dword = 0;
  std::uint16_t vertex_output_stride_dwords = 0;
  std::uint8_t front_facing = 0;
  // Degenerate post-clip fan triangles remain serialized for setup/counter
  // identity but are not referenced by the tiler.
  std::uint8_t rasterizable = 0;
  // Fast-path triangles that reach fixed setup before face culling remain
  // serialized for exact setup accounting but never enter a tile bin.
  std::uint8_t face_culled = 0;
  // RasterTriangle is normalized independently for the model's edge walker.
  // Keep Mesa's actual setup-JIT vertex order so binary32 coefficient
  // construction still uses the same anchor/subtraction sequence.  The three
  // entries are a permutation of the serialized raster vertices.
  std::uint8_t setup_vertex_order[3]{};
};

// TileRecord owns a contiguous range in TilePrimitiveRef. This preserves
// primitive identity and submission order, unlike the former 32-bit mask.
struct TileRecord {
  std::uint32_t x0 = 0;
  std::uint32_t y0 = 0;
  std::uint32_t x1 = 0;
  std::uint32_t y1 = 0;
  std::uint32_t first_primitive_ref = 0;
  std::uint32_t primitive_ref_count = 0;
};

struct TilePrimitiveRef {
  std::uint32_t parameter_index = 0;
  std::uint32_t reserved = 0;
  std::uint64_t submit_ordinal = 0;
};

struct EdgeEquation {
  std::int64_t a = 0;
  std::int64_t b = 0;
  std::int64_t c = 0;
  std::uint8_t inclusive = 0;
  std::uint8_t reserved[7]{};
};

struct ParameterTriangle {
  PrimitiveKey key;
  EdgeEquation edge[3];
  std::int64_t signed_area = 0;
  std::int32_t min_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_x = 0;
  std::int32_t max_y = 0;
  float window_z[3]{};
  // Native driver-PCO depth follows llvmpipe's position-slot setup ABI, not
  // the 24.8 coverage barycentrics above.  A/B/C are raw binary32 dwords for
  // the plane evaluated as fma(B, y, fma(A, x, C)); PAD is kept explicit so
  // the serialized payload has the same fail-closed shape as varying planes.
  std::uint32_t depth_plane[4]{};
  std::uint8_t depth_plane_valid = 0;
  std::uint8_t depth_plane_reserved[3]{};
  // Contiguous A/B/C/PAD sets in parameter_coefficients. A non-rasterizable
  // placeholder owns no sets and therefore has coefficient_set_count == 0.
  std::uint32_t first_coefficient_set = 0;
  std::uint16_t coefficient_set_count = 0;
  std::uint8_t front_facing = 0;
  std::uint8_t rasterizable = 0;
  std::uint8_t face_culled = 0;
  std::uint8_t reserved[3]{};
};

inline bool HasCanonicalDepthPlaneMetadata(
    FunctionalCase functional_case, const ParameterTriangle &triangle) {
  const bool expected = IsDriverPcoTrianglesCase(functional_case) &&
                        triangle.rasterizable != 0;
  if ((triangle.depth_plane_valid != 0) != expected ||
      triangle.depth_plane_valid > 1 || triangle.depth_plane[3] != 0 ||
      triangle.depth_plane_reserved[0] != 0 ||
      triangle.depth_plane_reserved[1] != 0 ||
      triangle.depth_plane_reserved[2] != 0) {
    return false;
  }
  if (!expected) {
    return triangle.depth_plane[0] == 0 && triangle.depth_plane[1] == 0 &&
           triangle.depth_plane[2] == 0;
  }
  for (std::size_t component = 0; component < 3; ++component) {
    if ((triangle.depth_plane[component] & UINT32_C(0x7f800000)) ==
        UINT32_C(0x7f800000)) {
      return false;
    }
  }
  return true;
}

// One public Rogue coefficient set consists of raw IEEE-754 binary32 plane
// coefficients A/B/C plus the architectural padding dword.
struct ParameterCoefficientSet {
  std::uint32_t a = 0;
  std::uint32_t b = 0;
  std::uint32_t c = 0;
  std::uint32_t pad = 0;
};

// ISP records every covered sample candidate, rather than a union mask. Opaque
// HSR marks one final owner visible per sample; blending keeps every passing
// candidate visible in submit order.
struct FragmentCandidate {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t primitive_id = 0;
  std::uint32_t parameter_index = 0;
  std::uint64_t submit_ordinal = 0;
  float depth = 0.0f;
  float barycentric[3]{};
  std::uint8_t sample_mask = 0;
  FragmentVisibility visibility = FragmentVisibility::kRejected;
  std::uint8_t reserved[2]{};
};

struct FragmentInvocation {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t primitive_id = 0;
  std::uint32_t parameter_index = 0;
  std::uint64_t submit_ordinal = 0;
  std::uint32_t quad_id = 0;
  std::uint8_t quad_lane = 0;
  std::uint8_t sample_mask = 0;
  std::uint8_t reserved[2]{};
  float depth = 0.0f;
  float barycentric[3]{};
};

// A shader lane may be a covered invocation or a helper lane. Helpers execute
// interpolation and texture instructions but never create FragmentOutput, so
// standard ps_invocations remains the number of visible samples.
struct FragmentShaderLane {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t primitive_id = 0;
  std::uint32_t parameter_index = 0;
  std::uint64_t submit_ordinal = 0;
  std::uint32_t quad_id = 0;
  std::uint32_t visible_invocation_index = kInvalidFragmentInvocationIndex;
  std::uint8_t quad_lane = 0;
  std::uint8_t sample_mask = 0;
  std::uint8_t helper = 0;
  std::uint8_t reserved = 0;
  float depth = 0.0f;
  float barycentric[3]{};
};

// Spatial USC work is retained explicitly instead of being reduced to a
// counter. Each entry represents one (parameter triangle, 2x2 quad) pair;
// invocation_indices address the flat fragment_invocations payload.
struct FragmentQuad {
  std::uint32_t parameter_index = 0;
  std::uint32_t quad_id = 0;
  std::uint64_t submit_ordinal = 0;
  std::uint32_t invocation_indices[4] = {
      kInvalidFragmentInvocationIndex, kInvalidFragmentInvocationIndex,
      kInvalidFragmentInvocationIndex, kInvalidFragmentInvocationIndex};
  std::uint8_t coverage_mask = 0;
  std::uint8_t helper_mask = 0;
  std::uint8_t write_mask = 0;
  std::uint8_t reserved = 0;
};

// PDS output descriptor. The corresponding raw coefficient dwords are copied
// into usc_coefficient_banks so USC never aliases ParameterBuffer storage.
struct UscFragmentTask {
  std::uint32_t fragment_quad_index = 0;
  std::uint32_t first_coefficient_dword = 0;
  std::uint16_t coefficient_dword_count = 0;
  std::uint16_t reserved = 0;
};

// One lane-local request emitted by an SMP instruction. Raw public Rogue
// image/sampler words and normalized coordinate bits are preserved end to end;
// TextureUnit derives the texel address from this payload and the resource.
struct TextureSampleRequest {
  std::uint32_t shader_lane_index = 0;
  // Implicit-derivative SMP is a spatial quad operation.  These fields retain
  // the FragmentFrontend/PDS quad identity across the USC -> TPU FIFO instead
  // of reconstructing derivatives from request ordering or a test-case name.
  std::uint32_t quad_id = 0;
  std::uint32_t coordinates[2]{};
  std::uint32_t texture_state[4]{};
  std::uint32_t sampler_state[4]{};
  std::uint64_t request_id = 0;
  std::uint8_t coordinate_count = 0;
  std::uint8_t component_count = 0;
  std::uint8_t descriptor_set = 0;
  std::uint8_t binding = 0;
  std::uint8_t dimension = 0;
  std::uint8_t normalized = 0;
  std::uint8_t data_request = 0;
  std::uint8_t quad_lane = 0;
  ShaderStage shader_stage = ShaderStage::kFragment;
  std::uint8_t reserved[3]{};
};

struct TextureSampleResponse {
  std::uint32_t shader_lane_index = 0;
  std::uint32_t rgba[4]{};
  std::uint64_t request_id = 0;
  ShaderStage shader_stage = ShaderStage::kFragment;
};

// Colour attachments one fragment shader can write in a single pass.
inline constexpr std::size_t kMaxRenderTargets = 4;

struct FragmentOutput {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::uint32_t primitive_id = 0;
  std::uint32_t parameter_index = 0;
  std::uint64_t submit_ordinal = 0;
  float depth = 0.0f;
  // Render-target major: attachment N occupies [4 * N, 4 * N + 4).
  std::uint32_t pixel_output[4 * kMaxRenderTargets]{};
  // PIXOUT channel mask per attachment; entries past the count stay zero.
  std::uint8_t written_mask[kMaxRenderTargets]{};
  std::uint8_t render_target_count = 1;
  std::uint8_t reserved[3]{};
};

inline constexpr std::size_t kDramLineWriteBytes = 128;

// A dirty SLC line is serialized into MemoryPool and only this line-record
// array handle crosses the SLC→DRAM FIFO. The fixed 128-byte payload matches
// the inactive DXTP SLC reference line size used by the selected cache uArch.
struct DramLineWrite {
  std::uint64_t address = 0;
  std::uint32_t bytes = kDramLineWriteBytes;
  std::uint32_t reserved = 0;
  std::uint8_t data[kDramLineWriteBytes]{};
};

static_assert(std::is_trivially_copyable_v<DramLineWrite>);

const char *PipelineStageName(PipelineStage stage);
std::size_t GetComponentTypeBytes(VertexComponentType type);
void RequireStage(PipelineStage actual, PipelineStage expected,
                  const char *module_name);

void ReleaseFunctionalPayloads(MemoryPool &pool, const PipelineState &state);

inline bool HasPoolHandle(PoolHandle handle) { return handle.generation != 0; }

template <typename T>
PoolHandle StoreNewArray(MemoryPool &pool, const std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::overflow_error("MemoryPool array size overflow");
  const std::size_t bytes = values.size() * sizeof(T);
  const PoolHandle handle = pool.Allocate(bytes);
  auto &destination = pool.Write(handle);
  if (bytes != 0)
    std::memcpy(destination.data(), values.data(), bytes);
  return handle;
}

template <typename T>
std::vector<T> LoadArray(const MemoryPool &pool, PoolHandle handle) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto &source = pool.Read(handle);
  if (source.size() % sizeof(T) != 0)
    throw std::runtime_error("MemoryPool array has an invalid byte size");
  std::vector<T> values(source.size() / sizeof(T));
  if (!source.empty())
    std::memcpy(values.data(), source.data(), source.size());
  return values;
}

template <typename T>
void StoreArray(MemoryPool &pool, PoolHandle handle,
                const std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  auto &destination = pool.Write(handle);
  if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
      destination.size() != values.size() * sizeof(T)) {
    throw std::runtime_error("MemoryPool array write size mismatch");
  }
  if (!destination.empty())
    std::memcpy(destination.data(), values.data(), destination.size());
}

static_assert(std::is_trivially_copyable_v<DrawCommand>);
static_assert(std::is_trivially_copyable_v<DrawListShaderStats>);
static_assert(std::is_trivially_copyable_v<DrawListStats>);
static_assert(std::is_trivially_copyable_v<DepthState>);
static_assert(std::is_trivially_copyable_v<BlendState>);
static_assert(std::is_trivially_copyable_v<FaceCullState>);
static_assert(std::is_trivially_copyable_v<RasterState>);
static_assert(std::is_trivially_copyable_v<InputVertex>);
static_assert(std::is_trivially_copyable_v<VertexBufferResource>);
static_assert(std::is_trivially_copyable_v<VertexAttributeBinding>);
static_assert(std::is_trivially_copyable_v<VertexLane>);
static_assert(std::is_trivially_copyable_v<VertexLaneRef>);
static_assert(std::is_trivially_copyable_v<ShaderSharedRegister>);
static_assert(std::is_trivially_copyable_v<TextureMipLevel>);
static_assert(std::is_trivially_copyable_v<TextureResource>);
static_assert(std::is_trivially_copyable_v<SamplerState>);
static_assert(std::is_trivially_copyable_v<ShaderVaryingBinding>);
static_assert(std::is_trivially_copyable_v<PrimitiveKey>);
static_assert(std::is_trivially_copyable_v<RasterTriangle>);
static_assert(std::is_trivially_copyable_v<TileRecord>);
static_assert(std::is_trivially_copyable_v<TilePrimitiveRef>);
static_assert(std::is_trivially_copyable_v<EdgeEquation>);
static_assert(std::is_trivially_copyable_v<ParameterTriangle>);
static_assert(std::is_trivially_copyable_v<ParameterCoefficientSet>);
static_assert(std::is_trivially_copyable_v<FragmentCandidate>);
static_assert(std::is_trivially_copyable_v<FragmentInvocation>);
static_assert(std::is_trivially_copyable_v<FragmentShaderLane>);
static_assert(std::is_trivially_copyable_v<FragmentQuad>);
static_assert(std::is_trivially_copyable_v<UscFragmentTask>);
static_assert(std::is_trivially_copyable_v<TextureSampleRequest>);
static_assert(std::is_trivially_copyable_v<TextureSampleResponse>);
static_assert(std::is_trivially_copyable_v<FragmentOutput>);

} // namespace pvrgpu::stub
