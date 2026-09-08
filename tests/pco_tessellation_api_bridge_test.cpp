// SPDX-License-Identifier: MIT
// Real compiler binaries travel through API ownership, all independent shader
// modules and raster/readback. The test supplies no product-side answer data.
#include "pvrgpu_systemc_api.h"
#include "pco_tessellation_compiler_fixtures.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
unsigned checks = 0;
void Check(bool value, const std::string &message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
struct Fixture {
  std::vector<std::uint8_t> vs = Tess0VertexPco(), tcs = kTess0tcs,
      tes = kTess0tes, fs = Tess0FragmentPco();
  std::array<std::uint32_t, 8> control_shared{};
  std::array<std::uint32_t, 4> evaluation_shared{};
  pvrgpu_systemc_tessellation tess{};
  pvrgpu_systemc_driver_command draw{};
  Fixture() {
    tess.control_pco = tcs.data(); tess.control_pco_size = tcs.size();
    tess.evaluation_pco = tes.data(); tess.evaluation_pco_size = tes.size();
    tess.control_shared = control_shared.data(); tess.control_shared_count = 8;
    tess.evaluation_shared = evaluation_shared.data(); tess.evaluation_shared_count = 4;
    tess.control_abi = {6,3,0,0,8,8,0,0,8,0};
    tess.evaluation_abi = {2,5,4,0,4,4,0,0,4,0};
    tess.input_vertices = tess.output_vertices = tess.vertices_per_instance = 1;
    tess.input_stride_dwords = tess.output_vertex_stride_dwords = 4;
    tess.per_vertex_offset_dwords = 6; tess.patch_stride_dwords = 10;
    draw.version = PVRGPU_SYSTEMC_API_VERSION;
    draw.command = "draw_pco_triangles";
    draw.case_name = "native.tessellation.ownership";
    draw.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    draw.frame = 1;
    draw.framebuffer_width = draw.width = draw.framebuffer_height = draw.height = 16;
    draw.primitive_mode = 14;
    draw.vertex_count = draw.instance_count = 1;
    draw.vertex_pco = vs.data(); draw.vertex_pco_size = vs.size();
    draw.fragment_pco = fs.data(); draw.fragment_pco_size = fs.size();
    draw.vertex_pco_abi = {0,0,4,0,0,0,0,0,0,0};
    draw.fragment_pco_abi = {1,0,0,4,0,0,0,0,0,0};
    draw.position_output_count = draw.varying_output_start = 4;
    draw.fragment_position_count = draw.fragment_varying_start = 4;
    draw.fragment_output_mask[0] = 15;
    draw.viewport_scale_bits[0] = draw.viewport_scale_bits[1] = 0x41000000;
    draw.viewport_scale_bits[2] = 0x3f000000;
    std::memcpy(draw.viewport_translate_bits, draw.viewport_scale_bits, sizeof(draw.viewport_scale_bits));
    draw.half_pixel_center = draw.depth_clip_near = draw.depth_clip_far = 1;
    draw.sample_mask = UINT32_MAX; draw.color_mask = 15;
    draw.blend_source_rgb_factor = draw.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.color_attachment_source_command_index = draw.depth_attachment_source_command_index = PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.clear_color_bits[0] = draw.clear_color_bits[2] = draw.clear_color_bits[3] = 0x3f800000;
    draw.tessellation = &tess;
  }
};
}

int main(int argc, char **argv) {
  try {
    static_assert(PVRGPU_SYSTEMC_API_VERSION == 25);
    const bool incomplete_patch = argc > 2 && std::string(argv[2]) == "incomplete";
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() / ("pvrgpu-tess-api25-" + std::to_string(nonce));
    unsigned mode_index = 0;
    // A SystemC session owns one immutable memory mode; exercise modes in
    // separate CTest processes rather than changing hardware during a run.
    for (const char *mode : {argc > 1 ? argv[1] : "cache"}) {
      Fixture f;
      auto sequence = f.draw;
      sequence.command = "draw_pco_sequence";
      sequence.vertex_pco = sequence.fragment_pco = nullptr;
      sequence.vertex_pco_size = sequence.fragment_pco_size = 0;
      sequence.vertex_pco_abi = sequence.fragment_pco_abi = {};
      sequence.tessellation = nullptr;
      sequence.vertex_count = sequence.instance_count = sequence.primitive_mode = 0;
      sequence.pco_sequence_commands = &f.draw; sequence.pco_sequence_command_count = 1;
      const auto dir = root / mode;
      const auto jsonl = (dir / "model.jsonl").string(), outdir = (dir / "out").string();
      const auto stderr_path = (dir / "model.stderr.log").string();
      pvrgpu_systemc_submit_info info{};
      info.version = PVRGPU_SYSTEMC_API_VERSION; info.command = &sequence;
      info.memory_mode = mode; info.jsonl_path = jsonl.c_str(); info.outdir = outdir.c_str();
      info.stderr_path = stderr_path.c_str();
      info.submission_generation = ++mode_index;
      std::array<char, 1024> error{};
      const auto reject = [&](const char *reason) {
        error.fill(0);
        const int result = pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size());
        Check(result == 2 && std::string(error.data()).find(reason) != std::string::npos,
              std::string("reject invalid Tessellation ABI: ") + error.data());
      };
      f.tess.input_vertices = 33; reject("patch vertex extent"); f.tess.input_vertices = 1;
      f.tess.domain = 3; reject("domain/spacing"); f.tess.domain = 0;
      f.tess.control_abi.vertex_inputs = 2; reject("native stage"); f.tess.control_abi.vertex_inputs = 3;
      f.control_shared[0] = 1; reject("unrelocated"); f.control_shared[0] = 0;
      f.tess.patch_stride_dwords = 9; reject("storage layout"); f.tess.patch_stride_dwords = 10;
      f.draw.vertex_count = 4097; f.tess.vertices_per_instance = 4097;
      reject("patch count exceeds");
      f.draw.vertex_count = UINT32_MAX; f.tess.vertices_per_instance = 1; f.tess.input_vertices = 32;
      reject("input occurrence count exceeds");
      f.draw.vertex_count = f.tess.input_vertices = 1;
      Check(!std::filesystem::exists(dir), "invalid envelopes create no model output");
      if (incomplete_patch) f.tess.input_vertices = 2;
      for (unsigned stage = 0; stage != 2; ++stage) {
        const auto &bytes = stage ? f.tes : f.tcs;
        const int result = pvrgpu_systemc_can_execute_pco_binary(stage + 3,
            bytes.data(), bytes.size(), error.data(), error.size());
        Check(result == 0, std::string("real independent stage decode: ") + error.data());
      }
      std::filesystem::create_directories(outdir);
      error.fill(0);
      const int accepted = pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size());
      Check(accepted == 0, std::string("accept native four-stage pipeline: ") + error.data());
      for (auto *bytes : {&f.vs, &f.tcs, &f.tes, &f.fs}) std::fill(bytes->begin(), bytes->end(), 0);
      f.control_shared.fill(UINT32_MAX); f.evaluation_shared.fill(UINT32_MAX);
      f.tess = {}; f.draw = {};
      pvrgpu_systemc_graphics_stats stats{};
      stats.version = PVRGPU_SYSTEMC_API_VERSION; stats.submission_generation = info.submission_generation;
      const int executed = pvrgpu_systemc_flush_graphics_stats(&stats, error.data(), error.size());
      Check(executed == 0, std::string("execute immutable Tessellation payload: ") + error.data());
      Check(stats.physical_submissions == 1 && stats.ia_primitives == (incomplete_patch ? 0U : 1U) &&
                (incomplete_patch ? stats.primitives_generated == 0 : stats.primitives_generated > 1) &&
                stats.gs_primitives == 0 && stats.gs_invocations == 0,
            "primitive query counts generated TES geometry, not input patches or GS work");
      std::array<std::uint8_t, 16 * 16 * 4> pixels{};
      pvrgpu_systemc_readback_info readback{};
      readback.version = PVRGPU_SYSTEMC_API_VERSION; readback.width = readback.height = 16;
      readback.bytes_per_pixel = 4; readback.pixels = pixels.data(); readback.pixels_size = pixels.size();
      Check(pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) == 0 && readback.pixels_written,
            std::string("native TES raster readback: ") + error.data());
      unsigned colored = 0;
      for (unsigned pixel = 0; pixel < 256; ++pixel) {
        const auto *rgba = pixels.data() + pixel * 4;
        const bool clear = rgba[0] == 255 && rgba[1] == 0 && rgba[2] == 255 && rgba[3] == 255;
        const bool shader = rgba[0] == 64 && rgba[1] == 128 && rgba[2] == 191 && rgba[3] == 255;
        Check(clear || shader, "readback contains only actual clear or FS color");
        colored += shader;
      }
      Check(incomplete_patch ? colored == 0 : colored > 0 && colored < 256,
            "only complete input patches produce TES coverage");
      std::ifstream stream(jsonl);
      const std::string json{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
      for (const char *evidence : {incomplete_patch ? "\"hs_invocations\":0" : "\"hs_invocations\":1",
           incomplete_patch ? "\"tcs_invocations\":0" : "\"tcs_invocations\":1",
           incomplete_patch ? "\"tcs_output_write_bytes\":0" : "\"tcs_output_write_bytes\":24", "\"vertex_attribute_bytes\":0",
           "\"pool_bytes_in_flight\":0", "\"pool_leaks\":0"})
        Check(json.find(evidence) != std::string::npos, std::string("native/pool evidence: ") + evidence);
      std::cout << "native Tessellation API25 " << mode << " PASS, primitives=" << stats.primitives_generated << '\n';
    }
    // Only this test's unique temporary directory is removed on success.
    std::filesystem::remove_all(root);
    std::cout << "native Tessellation API25 " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pco-tessellation-api-bridge-test: " << error.what() << '\n';
    return 1;
  }
}
