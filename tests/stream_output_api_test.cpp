// SPDX-License-Identifier: MIT
// Native VS execution, owned API payloads and generation-qualified DRAM output.
#include "pvrgpu_systemc_api.h"
#include "shader/pco_iss.h"

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
#include <vector>

namespace {
unsigned checks = 0;
void Check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) throw std::runtime_error(label);
}
struct Fixture {
  std::vector<std::uint8_t> vs = pvrgpu::stub::ConditionalsVertexPcoBinary();
  std::vector<std::uint8_t> fs = pvrgpu::stub::ConditionalsFragmentPcoBinary();
  std::array<float, 24> vertices = {-2,0,0,1, 0,0,0,1, 2,0,0,1,
                                   0,2,0,1, 0,-2,0,1, 0,0,0,1};
  std::array<std::uint32_t,16> vs_shared = {
    0x3f800000,0,0,0, 0,0x3f800000,0,0,
    0,0,0x3f800000,0, 0,0,0,0x3f800000};
  std::array<std::uint32_t,4> fs_shared = {0x3f800000,0,0,0x3f800000};
  std::vector<std::uint8_t> backing = std::vector<std::uint8_t>(256, 0xa5);
  std::array<pvrgpu_systemc_stream_output_binding,2> bindings = {{{0,4,0,0,0},{0,2,1,0,0}}};
  std::array<pvrgpu_systemc_stream_output_target,2> targets{};
  pvrgpu_systemc_stream_output so{};
  pvrgpu_systemc_driver_command draw{};
  Fixture() {
    targets[0] = {0,101,201,backing.data(),backing.size(),16,96,0,4};
    targets[1] = {1,101,202,backing.data(),backing.size(),128,48,0,2};
    so = {bindings.data(),1,targets.data(),1};
    draw.version = PVRGPU_SYSTEMC_API_VERSION;
    draw.command = "draw_pco_triangles"; draw.case_name = "native.stream_output";
    draw.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    draw.frame = 1; draw.width = draw.height = 16;
    draw.framebuffer_width = draw.framebuffer_height = 16;
    draw.primitive_mode = 0; draw.vertex_count = 6; draw.instance_count = 1;
    draw.vertex_stride = 16; draw.raw_vertex_data = reinterpret_cast<const std::uint8_t *>(vertices.data());
    draw.raw_vertex_data_size = sizeof(vertices);
    draw.vertex_attribute_count = 1; draw.vertex_attribute_components[0] = 4;
    draw.vertex_pco = vs.data(); draw.vertex_pco_size = vs.size();
    draw.fragment_pco = fs.data(); draw.fragment_pco_size = fs.size();
    draw.vertex_shared = vs_shared.data(); draw.vertex_shared_count = vs_shared.size();
    draw.fragment_shared = fs_shared.data(); draw.fragment_shared_count = fs_shared.size();
    draw.vertex_pco_abi = {10,4,4,0,16,0,16,0,0,0};
    draw.fragment_pco_abi = {4,0,0,4,4,0,4,0,0,0};
    draw.position_output_count = draw.varying_output_start = 4;
    draw.fragment_position_count = draw.fragment_varying_start = 4;
    draw.fragment_output_mask[0] = 15;
    draw.viewport_scale_bits[0] = draw.viewport_scale_bits[1] = 0x41000000;
    draw.viewport_scale_bits[2] = 0x3f000000;
    std::copy_n(draw.viewport_scale_bits,3,draw.viewport_translate_bits);
    draw.half_pixel_center = draw.depth_clip_near = draw.depth_clip_far = 1;
    draw.sample_mask = UINT32_MAX; draw.color_mask = 15;
    draw.blend_source_rgb_factor = draw.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.color_attachment_source_command_index = draw.depth_attachment_source_command_index = PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.stream_output = &so;
  }
};
}

int main(int argc, char **argv) {
  try {
    const char *mode = argc > 1 ? argv[1] : "cache";
    const auto root = std::filesystem::temp_directory_path() /
        ("pvrgpu-so-api26-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const auto jsonl = (root / "model.jsonl").string();
    const auto outdir = (root / "out").string();
    std::filesystem::create_directories(outdir);
    const auto stderr_log = (root / "model.stderr.log").string();
    pvrgpu_systemc_driver_command sequence{};
    sequence.version = PVRGPU_SYSTEMC_API_VERSION; sequence.command = "draw_pco_sequence";
    sequence.frame = 1;
    sequence.case_name = "native.stream_output"; sequence.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    sequence.width = sequence.height = sequence.framebuffer_width = sequence.framebuffer_height = 16;
    sequence.pco_sequence_command_count = 1;
    pvrgpu_systemc_submit_info info{};
    info.version = PVRGPU_SYSTEMC_API_VERSION; info.command = &sequence;
    info.jsonl_path = jsonl.c_str(); info.stderr_path = stderr_log.c_str(); info.memory_mode = mode;
    info.outdir = outdir.c_str();
    std::array<char,1024> error{};
    Fixture f;
    sequence.pco_sequence_commands = &f.draw;
    const auto reject = [&](const char *expected) {
      error.fill(0);
      Check(pvrgpu_systemc_submit_driver_command(&info,error.data(),error.size()) == 2 &&
            std::string(error.data()).find(expected) != std::string::npos,
            std::string("boundary ") + expected + ": " + error.data());
    };
    f.so.binding_count = 65; reject("binding/target"); f.so.binding_count = 1;
    f.bindings[0].output_dword = 4; reject("binding output"); f.bindings[0].output_dword = 0;
    f.bindings[0].stream = 1; reject("binding output"); f.bindings[0].stream = 0;
    f.targets[0].buffer_offset = 252; reject("target identity"); f.targets[0].buffer_offset = 16;
    f.targets[0].stride_dwords = 3; reject("binding exceeds"); f.targets[0].stride_dwords = 4;
    f.targets[0].internal_offset = 100; reject("target identity"); f.targets[0].internal_offset = 0;
    f.targets[0].target_token = 0; reject("target identity"); f.targets[0].target_token = 201;
    f.so.target_count = 2;
    f.targets[1].target_token = 201; reject("token is duplicated"); f.targets[1].target_token = 202;
    f.targets[1].bytes_size = 255; reject("snapshots disagree"); f.targets[1].bytes_size = 256;
    f.so.target_count = 1;
    std::array<pvrgpu_systemc_driver_command,2> two = {f.draw,f.draw};
    sequence.pco_sequence_commands = two.data(); sequence.pco_sequence_command_count = 2;
    reject("completion between");
    sequence.pco_sequence_command_count = 1;
    sequence.pco_sequence_commands = &f.draw;
    pvrgpu_systemc_varying_binding varying{4,1,4,0};
    f.draw.varying_binding_count = 1; reject("explicit varying binding list");
    f.draw.varying_bindings = &varying;
    f.draw.varying_binding_count = 65; reject("explicit varying binding list");
    f.draw.varying_binding_count = 1;
    reject("explicit varying binding output/coefficient");
    f.draw.vertex_pco_abi.vertex_outputs = 5; f.draw.varying_output_count = 1;
    f.draw.fragment_pco_abi.coefficients = 8; f.draw.fragment_varying_count = 4;
    varying.flat = 2; reject("explicit varying binding output/coefficient"); varying.flat = 0;
    varying.coefficient_dword = 8; reject("explicit varying binding output/coefficient"); varying.coefficient_dword = 4;
    f.draw.varying_binding_count = 0; reject("do not cover fragment coefficients");

    // All points lie at/outside clip boundaries; feedback must preserve the
    // shader's homogeneous positions before clipping or viewport conversion.
    for (unsigned scenario = 0; scenario != 5; ++scenario) {
      Fixture current;
      current.draw.varying_bindings = &varying; // Explicit empty FS mapping.
      auto expected = current.backing;
      unsigned captured = 6, vertices_per_primitive = 1;
      if (scenario == 1) { // Triangle overflow commits exactly one whole primitive.
        current.draw.primitive_mode = 4; current.targets[0].buffer_size = 80;
        captured = 3; vertices_per_primitive = 3;
      } else if (scenario == 2) { // Append consumes remaining range, preserves prefix.
        current.targets[0].internal_offset = 32; captured = 4;
      } else if (scenario == 3) { // Two bindings share one physical BO, independent cursors.
        current.so.binding_count = current.so.target_count = 2;
      } else if (scenario == 4) { // Missing target overflows all enabled targets atomically.
        current.so.binding_count = 2; captured = 0;
      }
      const auto initial_offset = current.targets[0].internal_offset;
      for (unsigned v = 0; v != captured; ++v) {
        std::memcpy(expected.data() + 16 + initial_offset + 16*v, current.vertices.data()+4*v,16);
        if (scenario == 3)
          std::memcpy(expected.data() + 128 + 8*v, current.vertices.data()+4*v,8);
      }
      sequence.pco_sequence_commands = &current.draw;
      info.submission_generation = scenario + 1;
      error.fill(0);
      Check(pvrgpu_systemc_submit_driver_command(&info,error.data(),error.size()) == 0,
            std::string("submit native SO: ") + error.data());
      // The producer may release all borrowed storage immediately after submit.
      std::fill(current.vs.begin(),current.vs.end(),0);
      current.vertices.fill(99); current.backing.assign(256,0x3c);
      current.bindings = {}; current.targets = {}; current.so = {}; current.draw = {};
      std::vector<std::uint8_t> result(256,0xcc);
      pvrgpu_systemc_stream_output_readback readback{};
      readback.version = PVRGPU_SYSTEMC_API_VERSION;
      readback.submission_generation = info.submission_generation;
      readback.resource_token = 101; readback.target_token = 201;
      readback.bytes = result.data(); readback.bytes_size = result.size();
      Check(pvrgpu_systemc_flush_stream_output(&readback,error.data(),error.size()) == 0 && readback.data_written,
            std::string("native readback: ") + error.data());
      Check(result == expected,"all feedback bytes and untouched guard regions match");
      Check(readback.internal_offset == initial_offset + captured*16,"cursor reflects committed vertices");
      pvrgpu_systemc_graphics_stats stats{};
      stats.version = PVRGPU_SYSTEMC_API_VERSION; stats.submission_generation = info.submission_generation;
      Check(pvrgpu_systemc_flush_graphics_stats(&stats,error.data(),error.size()) == 0,"completed SO query");
      Check(stats.stream_output_primitives_written == captured/vertices_per_primitive &&
            stats.stream_output_primitives_storage_needed == 6/vertices_per_primitive,
            "written/needed count whole preclip primitives");
      if (scenario == 3) {
        readback.target_token = 202; std::fill(result.begin(),result.end(),0);
        Check(pvrgpu_systemc_flush_stream_output(&readback,error.data(),error.size()) == 0 &&
              result == expected && readback.internal_offset == 48,
              "aliased target reads the same final resource and its own cursor");
      }
      readback.target_token = 999;
      Check(pvrgpu_systemc_flush_stream_output(&readback,error.data(),error.size()) == 2 && !readback.data_written,
            "wrong target cannot consume a readback");
      readback.submission_generation = 999;
      Check(pvrgpu_systemc_flush_stream_output(&readback,error.data(),error.size()) == 2,
            "wrong generation cannot consume a readback");
    }
    std::ifstream input(jsonl);
    const std::string report{std::istreambuf_iterator<char>(input),{}};
    Check(report.find("\"pool_leaks\":0") != std::string::npos &&
          report.find("\"type\":\"error\"") == std::string::npos,"model completed with balanced ownership");
    std::cout << "stream output API " << mode << ' ' << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "stream output API FAIL: " << e.what() << '\n';
    return 1;
  }
}
