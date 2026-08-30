// GFXBench Manhattan Frame 0 逐 DrawList 執行與除錯 Harness
//
// 輸出目錄：PVRGPU_SANDBOX_WORK_ROOT，預設為 $HOME/Downloads/_Codex/Working/SandBox
// 支援指定執行 DrawList 前綴 [0 .. target_draw_id]

#include "Sandbox/manhattan_drawlist.h"
#include "cache_mmu/slc.h"
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/fragment_frontend.h"
#include "fragment/isp.h"
#include "fragment/pbe.h"
#include "fragment/tile_scheduler.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"
#include "memory/dram_model.h"
#include "memory_pool.h"
#include "pds/pds_engine.h"
#include "shader/pco_decoder.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"
#include "support/png_writer.h"

#include <systemc>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace pvrgpu::sandbox {

inline std::string SandBoxOutputDir() {
  if (const char* override_dir = std::getenv("PVRGPU_SANDBOX_WORK_ROOT")) {
    if (override_dir[0] != '\0') {
      return override_dir;
    }
  }
  if (const char* home_dir = std::getenv("HOME")) {
    if (home_dir[0] != '\0') {
      return std::string(home_dir) + "/Downloads/_Codex/Working/SandBox";
    }
  }
  return "/tmp/PvrGPU-SandBox";
}

// 建構 Manhattan 第 1 個 DrawList (Pass 1: Shadow Depth Pre-pass Mesh)
ManhattanDrawList BuildManhattanDrawList0() {
  ManhattanDrawList draw;
  draw.draw_id = 0;
  draw.pass = ManhattanPass::kShadowDepth;
  draw.name = "Pass1_ShadowDepth_Draw0_BuildingMesh";

  draw.fbo.width = 320;
  draw.fbo.height = 180;
  draw.fbo.color_attachment_count = 0; // 純深度 Pass
  draw.fbo.has_depth_attachment = true;

  draw.do_clear_depth = true;
  draw.clear_depth = 1.0f;

  // 深度與光柵化狀態
  draw.raster_state.depth.test_enable = 1;
  draw.raster_state.depth.write_enable = 1;
  draw.raster_state.depth.compare_op = stub::DepthCompareOp::kLess;
  draw.raster_state.depth.clear_depth = 1.0f;
  draw.raster_state.face_cull.enable = 1;
  draw.raster_state.face_cull.cull_face = stub::CullFace::kBack;
  draw.raster_state.face_cull.front_face = stub::FrontFace::kCounterClockwise;

  // 3D 頂點位置 (x, y, z) 範例幾何 (Building facade / Quad in 3D space)
  draw.vertex_stride_bytes = 3 * sizeof(float);
  draw.vertex_count = 4;
  draw.vertex_attributes = {
      -10.0f, -5.0f, 20.0f, // v0
       10.0f, -5.0f, 20.0f, // v1
      -10.0f,  5.0f, 20.0f, // v2
       10.0f,  5.0f, 20.0f  // v3
  };

  // 索引 (兩個三角形組成的 Quad)
  draw.indices = {0, 1, 2, 2, 1, 3};
  draw.index_count = 6;

  return draw;
}

} // namespace pvrgpu::sandbox
