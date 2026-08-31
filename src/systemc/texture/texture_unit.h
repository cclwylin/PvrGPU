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
#include "memory_pool.h"
#include "model_types.h"

#include <systemc>

#include <array>
#include <cstdint>

namespace pvrgpu::stub {

// Decoded subset of the public Rogue STRIDE image descriptor which the
// selected reference TPU supports.  Unsupported/reserved encodings fail in
// DecodeRogueTextureImageDescriptor rather than falling back to parallel
// software metadata.
struct RogueTextureImageDescriptor {
  std::uint64_t gpu_address = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t row_pitch_bytes = 0;
  std::uint8_t mip_count = 0;
  TextureFormat format = TextureFormat::kRgba8Unorm;
  TextureLayout layout = TextureLayout::kLinear;
};

// Decoded subset of public Rogue SAMPLER_WORD0/1.  The raw fields select
// normalized repeat addressing, image filters, mip filtering, and U4.6 LOD
// clamps.  Gate 18 uses mip-linear implicit LOD; no parallel case metadata is
// allowed to override these hardware words.
struct RogueTextureSamplerDescriptor {
  TextureFilter min_filter = TextureFilter::kNearest;
  TextureFilter mag_filter = TextureFilter::kNearest;
  TextureFilter mip_filter = TextureFilter::kNearest;
  TextureWrapMode wrap_u = TextureWrapMode::kRepeat;
  TextureWrapMode wrap_v = TextureWrapMode::kRepeat;
  std::uint16_t min_lod_u4_6 = 0;
  std::uint16_t max_lod_u4_6 = 0;
  std::uint8_t normalized_coordinates = 1;
};

struct TextureLinearAxis {
  std::uint32_t lower = 0;
  std::uint32_t upper = 0;
  std::uint16_t weight = 0;
};

// Result of the selected TPU's implicit-derivative LOD datapath.  The four
// coordinates are ordered as the architectural 2x2 quad lanes 0,1,2,3.
struct TextureImplicitLod {
  float lambda = 0.0F;
  float mip_weight = 0.0F;
  std::uint8_t level0 = 0;
  std::uint8_t level1 = 0;
  std::uint8_t mip_weight_u8 = 0;
  std::uint8_t reserved = 0;
};

RogueTextureImageDescriptor DecodeRogueTextureImageDescriptor(
    const std::array<std::uint32_t, 4>& words);
RogueTextureSamplerDescriptor DecodeRogueTextureSamplerDescriptor(
    const std::array<std::uint32_t, 4>& words);
TextureLinearAxis ComputeTextureLinearRepeat(float coordinate,
                                             std::uint32_t extent,
                                             TextureWrapMode wrap = TextureWrapMode::kRepeat,
                                             float round_threshold = 0.5F);
std::uint8_t LerpTextureUnorm8(std::uint8_t first, std::uint8_t second,
                               std::uint16_t weight);
TextureImplicitLod ComputeTextureImplicitLod(
    const std::array<std::array<float, 2>, 4>& coordinates,
    const RogueTextureImageDescriptor& image,
    const RogueTextureSamplerDescriptor& sampler);

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

  MemoryPool& pool_;
  GpuMemorySystem *memory_;
  bool texture_preloaded_ = false;
  std::uint64_t preloaded_address_ = 0;
  std::uint64_t preloaded_bytes_ = 0;
};

}  // namespace pvrgpu::stub
