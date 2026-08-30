// Module：ClipCull。
// 縮寫：非縮寫（裁剪與剔除）。
// 功能：讀取 PCO ISS（Instruction Set Simulator）產生的 VTXOUT0..3。
// Fill.Solid 保留 non-indexed 直通；Triangle.Setup 與 AttributeFetchShader
// 依 VertexLaneRef 重組 indexed triangles，依 Mesa 公開 clipper plane-bit
// order 與 dp>=0 boundary 執行 homogeneous 六平面 Sutherland-Hodgman
// clipping、fan emission、GLES face cull 與 viewport transform。zero-area 或
// clean-path face-culled setup candidate 仍保留 primitive identity 與 counter，
// 由 rasterizable marker 阻止後續 tiling。bounded FIFO（First-In,
// First-Out）只傳 MemoryPool handle，以 event-driven delay 完成。PCO 是 Mesa
// 公開 PowerVR compiler backend 的名稱，本專案不推定其縮寫展開。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class ClipCull final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  ClipCull(sc_core::sc_module_name name, MemoryPool &pool);

private:
  void Run();

  MemoryPool &pool_;
};

} // namespace pvrgpu::stub
