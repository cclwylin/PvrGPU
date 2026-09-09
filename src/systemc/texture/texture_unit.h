// TextureUnit 模擬 PowerVR TPU（Texture Processing Unit，紋理處理
// 單元）。SMP request 從 USC（Unified Shading Cluster）送入；TPU 解析
// texture/sampler descriptor、計算 normalized repeat addressing，並做
// nearest 或四 tap bilinear filtering。每個真實 texel read 都以 FIFO
//（First-In, First-Out）送至 TCU（Texture Cache Unit）。TCU/SLC/DRAM
// 回應後，TPU 將 RGBA response 交回 USC
// continuation，由 USC 完成 WDF/PIXOUT。Texture allocation 只在首次
// sample 前預置到 DRAM；non-texture cases 則無 request 通過 Run。Bulk data
// 留在 MemoryPool，FIFO 僅傳 handle/control，timing 是 event-driven。
#pragma once

#include "common/functional_types.h"
#include "memory/gpu_memory_system.h"
#include "texture/texture_filter.h"
#include "shader/pco_iss.h"
#include "memory_pool.h"
#include "model_types.h"

#include <systemc>

#include <array>
#include <cstdint>

namespace pvrgpu::stub {

// Memory request IDs a sample's texel fetches are numbered from: the sample's
// own request ID times this, plus the tap index within the sample.  Sixteen is
// the most taps one sample reads -- a 3D trilinear filter takes two mip
// levels, two depth slices each and a 2x2 footprint per slice -- and the IDs
// only have to be distinct within a batch, so the stride is fixed rather than
// packed to the taps a particular sample happens to issue.
inline constexpr std::uint64_t kTextureSampleTapRequestStride = 16;

// Raw gather's binary32 coordinate datapath: round s*extent and independently
// round +/-0.5 before truncation and clamp-to-edge. Do not derive the upper
// neighbor from the lower-clamped index, or replace finite-precision addressing
// with an ideal double floor; both differ at observable half-texel boundaries.
std::array<std::uint32_t, 2> ComputeTextureGatherClampToEdge(
    float coordinate, std::uint32_t extent);

// Validates a declared tightly packed RGBA8 texture view of an actual earlier
// sequence color attachment in unified GPU memory. A one-level view aliases
// the producer bytes without rewriting them; a multi-level view derives and
// commits every lower mip. No captured or golden image bytes participate.
MemoryAccessStats MaterializeSequenceColorMipChain(
    GpuMemorySystem &memory, const DriverPcoSampledTexture &texture,
    std::uint64_t attachment_address = 0);

// `compressed` says which enum the seven-bit texformat field is read
// through.  Rogue overlays FORMAT and FORMAT_COMPRESSED on the same bits, so
// the value alone cannot say whether 0 means U8 or ASTC_4x4.  The structured
// resource the command carried supplies that, and the two are cross-checked
// afterwards exactly as every other descriptor field is -- the model is told
// which enum applies, not what the answer is.
RogueTextureImageDescriptor DecodeRogueTextureImageDescriptor(
    const std::array<std::uint32_t, 4>& words, bool compressed = false);
RogueTextureSamplerDescriptor DecodeRogueTextureSamplerDescriptor(
    const std::array<std::uint32_t, 4>& words);
// Native NNCOORDS/SNO texelFetch addressing. Returns false for shader
// coordinates, layer or sample outside the image; those requests return zero
// without a memory read. Malformed resource/request metadata fails closed.
// llvmpipe lp_build_sample_ms_offset adds a sample-major stride; our existing
// render-target storage instead places the sample inside each pixel.
bool ComputeTextureMultisampleTexelOffset(
    const TextureResource &resource, const TextureSampleRequest &request,
    std::uint32_t layer, std::uint64_t *offset);
// Non-MS NNCOORDS + REPLACE: integer-valued float coordinates/LOD emitted
// by native PCO. No sampler clamp, wrap or filtering changes the selected texel.
bool ComputeTextureTexelOffset(
    const TextureResource &resource, const TextureSampleRequest &request,
    std::uint32_t array_layer, std::uint64_t *offset);
// Cross-check the raw one-level array depth/TEXTYPE used by native TAO and
// textureSize against the resource extent; structured metadata cannot replace
// or widen a shader-visible descriptor field.
void ValidateTextureSingleLevelDimensions(
    const std::array<std::uint32_t, 4> &words,
    const TextureResource &resource);
// Explicit whole-cube, uncompressed, single-sample layout. TAO must name an
// exact shader-produced cube base, never a face or an approximate byte offset.
void ValidateTextureCubeArrayLayout(const TextureResource &resource);
std::uint32_t TextureCubeArrayBaseFace(const TextureResource &resource,
                                      std::uint64_t image_address,
                                      std::uint64_t sample_address);
// What driver-PCO sampling can serve: a format the unit decodes, address
// modes the wrap arithmetic implements, a LOD window that runs forwards.
// Level selection is not a constraint -- see texture_filter.h.
bool DriverPcoTextureDescriptorClassSupported(
    const RogueTextureImageDescriptor& image,
    const RogueTextureSamplerDescriptor& sampler,
    std::uint32_t descriptor_count);
TextureImplicitLod ComputeTextureImplicitLod(
    const std::array<std::array<float, 2>, 4>& coordinates,
    const RogueTextureImageDescriptor& image,
    const RogueTextureSamplerDescriptor& sampler);
TextureImplicitLod ComputeTextureExplicitLod(
    float level, const RogueTextureImageDescriptor &image,
    const RogueTextureSamplerDescriptor &sampler);
TextureImplicitLod ComputeTexture3DImplicitLod(
    const std::array<std::array<float, 3>, 4> &coordinates,
    std::uint32_t depth, const RogueTextureImageDescriptor &image,
    const RogueTextureSamplerDescriptor &sampler);
TextureImplicitLod ApplyTextureLodBias(
    const TextureImplicitLod &implicit, float bias,
    const RogueTextureImageDescriptor &image,
    const RogueTextureSamplerDescriptor &sampler,
    bool undefined_cube_footprint = false);
// Cube directions use a common-face projection for implicit derivatives.
// Undefined nonfinite footprints select minimum LOD, retaining raw derivative
// fields for diagnostics; actual cube texel addressing is separately bounded.
TextureImplicitLod ComputeTextureCubeImplicitLod(
    const std::array<std::array<float, 3>, 4> &directions,
    const RogueTextureImageDescriptor &image,
    const RogueTextureSamplerDescriptor &sampler);

class TextureUnit final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      sample_input{"sample_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      sample_output{"sample_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      vertex_sample_input{"vertex_sample_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      vertex_sample_output{"vertex_sample_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      geometry_sample_input{"geometry_sample_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      geometry_sample_output{"geometry_sample_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND> tessellation_control_sample_input{"tessellation_control_sample_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND> tessellation_control_sample_output{"tessellation_control_sample_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND> tessellation_evaluation_sample_input{"tessellation_evaluation_sample_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND> tessellation_evaluation_sample_output{"tessellation_evaluation_sample_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND> compute_sample_input{"compute_sample_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND> compute_sample_output{"compute_sample_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      cache_request{"cache_request"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      cache_response{"cache_response"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      upload_request{"upload_request"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      upload_response{"upload_response"};

  TextureUnit(sc_core::sc_module_name name, MemoryPool& pool,
              GpuMemorySystem *memory = nullptr);

 private:
  void Run();
  void SampleRun();
  void VertexSampleRun();
  void GeometrySampleRun();
  void TessellationControlSampleRun();
  void TessellationEvaluationSampleRun();
  void ComputeSampleRun();
  void SampleRunForStage(
      ShaderStage stage,
      sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                       sc_core::SC_ZERO_OR_MORE_BOUND> &sample_input_port,
      sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                       sc_core::SC_ZERO_OR_MORE_BOUND> &sample_output_port);

  MemoryPool& pool_;
  GpuMemorySystem *memory_;
  // All six shader-stage descriptor-set namespaces are independent. Residency is kept
  // within one PipelineState (including all of its SMP continuation rounds)
  // and reset when the next physical draw receives a new state handle.
  std::array<std::array<bool, kPcoMaximumTextureDescriptorSets>, 6>
      texture_preloaded_{};
  std::array<std::array<std::uint64_t, kPcoMaximumTextureDescriptorSets>, 6>
      preloaded_address_{};
  std::array<std::array<std::uint64_t, kPcoMaximumTextureDescriptorSets>, 6>
      preloaded_bytes_{};
  std::array<PoolHandle, 6> residency_state_{};
};

}  // namespace pvrgpu::stub
