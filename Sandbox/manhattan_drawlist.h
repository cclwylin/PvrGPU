// GFXBench Manhattan Frame 0 DrawList 定義與資料結構
#pragma once

#include "common/functional_types.h"
#include "common/pipeline_state.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pvrgpu::sandbox {

enum class ManhattanPass : std::uint8_t {
  kShadowDepth = 0,    // Pass 1: 純深度 Shadow Map
  kGBufferSolids = 1,  // Pass 2: G-Buffer 幾何實體 (MRT)
  kLighting = 2,       // Pass 3: 延遲光照計算
  kTransparents = 3,   // Pass 4: 半透明與後處理
};

struct ManhattanFboConfig {
  std::uint32_t width = 320;
  std::uint32_t height = 180;
  std::uint8_t color_attachment_count = 1;
  bool has_depth_attachment = true;
  bool has_stencil_attachment = false;
};

struct ManhattanDrawList {
  std::uint32_t draw_id = 0;
  ManhattanPass pass = ManhattanPass::kShadowDepth;
  std::string name;

  // FBO 與清屏狀態
  ManhattanFboConfig fbo;
  bool do_clear_color = false;
  bool do_clear_depth = false;
  std::array<float, 4> clear_color{0.0f, 0.0f, 0.0f, 1.0f};
  float clear_depth = 1.0f;

  // 幾何頂點與索引
  std::vector<float> vertex_attributes; // 例如 pos(float3) + normal(float3) + uv(float2)
  std::vector<std::uint16_t> indices;
  std::uint32_t vertex_stride_bytes = 0;
  std::uint32_t vertex_count = 0;
  std::uint32_t index_count = 0;

  // PCO Shader Binaries (真實 Mesa 公開編譯字節碼)
  std::vector<std::uint8_t> vs_pco_binary;
  std::vector<std::uint8_t> fs_pco_binary;

  // 光柵與深度狀態
  pvrgpu::stub::RasterState raster_state;
};

// Frame 0 前綴序列
struct ManhattanFrame0 {
  std::vector<ManhattanDrawList> draw_lists;
};

} // namespace pvrgpu::sandbox
