#include "driver_command.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::filesystem::path TempFile(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("pvrgpu-driver-command-test-" + std::to_string(std::rand()) + "-" +
          name);
}

void WriteText(const std::filesystem::path &path, const std::string &text) {
  std::ofstream output(path);
  output << text;
}

int Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << '\n';
    return 1;
  }
  return 0;
}

} // namespace

int main() {
  using pvrgpu::stub::DriverCommand;
  using pvrgpu::stub::LoadDriverCommand;

  const std::filesystem::path good = TempFile("good.txt");
  WriteText(good,
            "schema=pvrgpu.driver-command.v1\n"
            "producer=pvrgpu-gallium-driver\n"
            "command=clear_color\n"
            "case=phase1.clear.green\n"
            "frame=1\n"
            "width=8\n"
            "height=4\n"
            "format=PIPE_FORMAT_R8G8B8A8_UNORM\n"
            "clear_color_bits=0,1065353216,0,1065353216\n");

  DriverCommand command;
  std::string error;
  if (int failed =
          Expect(LoadDriverCommand(good.string(), &command, &error), error))
    return failed;
  if (int failed = Expect(command.enabled, "valid command was not enabled"))
    return failed;
  if (int failed =
          Expect(command.command == "clear_color", "wrong command kind"))
    return failed;
  if (int failed =
          Expect(command.width == 8 && command.height == 4, "wrong size"))
    return failed;
  if (int failed = Expect(command.clear_color_bits[1] == 1065353216U,
                          "wrong green clear bits"))
    return failed;

  const std::filesystem::path draw = TempFile("draw.txt");
  WriteText(draw,
            "schema=pvrgpu.driver-command.v1\n"
            "producer=pvrgpu-gallium-driver\n"
            "command=draw_triangle\n"
            "case=phase2.draw_triangle.gallium\n"
            "frame=1\n"
            "width=16\n"
            "height=16\n"
            "format=PIPE_FORMAT_R8G8B8A8_UNORM\n"
            "clear_color_bits=0,0,0,1065353216\n"
            "vertex0_bits=3212836864,3212836864\n"
            "vertex1_bits=1065353216,3212836864\n"
            "vertex2_bits=0,1065353216\n"
            "fragment_color_bits=1065353216,0,0,1065353216\n");
  error.clear();
  if (int failed =
          Expect(LoadDriverCommand(draw.string(), &command, &error), error))
    return failed;
  if (int failed =
          Expect(command.command == "draw_triangle", "wrong draw command kind"))
    return failed;
  if (int failed =
          Expect(command.vertex_bits[2][1] == 1065353216U,
                 "wrong third vertex y bits"))
    return failed;
  if (int failed =
          Expect(command.fragment_color_bits[0] == 1065353216U,
                 "wrong fragment red bits"))
    return failed;

  const std::filesystem::path quad = TempFile("quad.txt");
  WriteText(quad,
            "schema=pvrgpu.driver-command.v1\n"
            "producer=pvrgpu-gallium-driver\n"
            "command=draw_indexed_quad\n"
            "case=phase7.draw_indexed_quad.gallium\n"
            "frame=1\n"
            "framebuffer_width=512\n"
            "framebuffer_height=512\n"
            "width=64\n"
            "height=64\n"
            "format=PIPE_FORMAT_R10G10B10A2_UNORM\n"
            "clear_color_bits=0,0,0,1065353216\n"
            "draw_count=2\n"
            "index_count=6\n"
            "unique_vertices=4\n"
            "primitive_count=2\n"
            "clip_primitives=2\n"
            "setup_triangles=2\n"
            "semantic_texel_fetches=74784\n");
  error.clear();
  if (int failed =
          Expect(LoadDriverCommand(quad.string(), &command, &error), error))
    return failed;
  if (int failed =
          Expect(command.command == "draw_indexed_quad",
                 "wrong indexed quad command kind"))
    return failed;
  if (int failed =
          Expect(command.draw_count == 2 && command.width == 64 &&
                     command.height == 64 &&
                     command.framebuffer_width == 512 &&
                     command.framebuffer_height == 512,
                 "wrong indexed quad dimensions or draw count"))
    return failed;
  if (int failed =
          Expect(command.index_count == 6 && command.unique_vertices == 4 &&
                     command.primitive_count == 2 &&
                     command.clip_primitives == 2 &&
                     command.setup_triangles == 2,
                 "wrong indexed quad topology metadata"))
    return failed;
  if (int failed =
          Expect(command.semantic_texel_fetches == 74784,
                 "wrong indexed quad semantic texel fetch metadata"))
    return failed;

  const std::filesystem::path primitives = TempFile("primitives.txt");
  WriteText(primitives,
            "schema=pvrgpu.driver-command.v1\n"
            "producer=pvrgpu-gallium-driver\n"
            "command=draw_primitive_sequence\n"
            "case=dEQP-GLES3.functional.rasterization.primitives.line_loop\n"
            "frame=1\n"
            "width=512\n"
            "height=512\n"
            "format=PIPE_FORMAT_R8G8B8A8_UNORM\n"
            "clear_color_bits=0,0,0,1065353216\n"
            "draw_count=3\n"
            "ia_vertices=12\n"
            "ia_primitives=12\n"
            "vs_invocations=12\n"
            "gs_invocations=6\n"
            "gs_primitives=9\n"
            "clip_invocations=12\n"
            "clip_primitives=12\n"
            "setup_triangles=0\n"
            "ps_invocations=1052\n"
            "hs_invocations=2\n"
            "ds_invocations=8\n"
            "cs_invocations=13\n"
            "semantic_texel_fetches=0\n");
  error.clear();
  if (int failed =
          Expect(LoadDriverCommand(primitives.string(), &command, &error),
                 error))
    return failed;
  if (int failed =
          Expect(command.command == "draw_primitive_sequence" &&
                     command.draw_count == 3 &&
                     command.ia_vertices == 12 &&
                     command.gs_invocations == 6 &&
                     command.gs_primitives == 9 &&
                     command.clip_primitives == 12 &&
                     command.ps_invocations == 1052 &&
                     command.hs_invocations == 2 &&
                     command.ds_invocations == 8 &&
                     command.cs_invocations == 13 &&
                     command.semantic_texel_fetches == 0,
                 "wrong primitive sequence metadata"))
    return failed;

  const std::filesystem::path discard = TempFile("discard.txt");
  WriteText(discard,
            "schema=pvrgpu.driver-command.v1\n"
            "producer=pvrgpu-gallium-driver\n"
            "command=draw_primitive_sequence\n"
            "case=dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_triangle_fan\n"
            "frame=1\n"
            "width=512\n"
            "height=512\n"
            "format=PIPE_FORMAT_R8G8B8A8_UNORM\n"
            "clear_color_bits=0,0,0,1065353216\n"
            "draw_count=1\n"
            "ia_vertices=6\n"
            "ia_primitives=4\n"
            "vs_invocations=6\n"
            "clip_invocations=0\n"
            "clip_primitives=0\n"
            "setup_triangles=0\n"
            "ps_invocations=0\n"
            "semantic_texel_fetches=0\n");
  error.clear();
  if (int failed =
          Expect(LoadDriverCommand(discard.string(), &command, &error),
                 error))
    return failed;
  if (int failed =
          Expect(command.command == "draw_primitive_sequence" &&
                     command.draw_count == 1 &&
                     command.clip_invocations == 0 &&
                     command.ps_invocations == 0,
                 "wrong discard primitive sequence metadata"))
    return failed;

  const std::filesystem::path bad = TempFile("bad.txt");
  WriteText(bad,
            "schema=pvrgpu.driver-command.v1\n"
            "producer=pvrgpu-gallium-driver\n"
            "command=draw_vbo\n"
            "case=phase2.draw\n"
            "frame=1\n"
            "width=8\n"
            "height=4\n"
            "format=PIPE_FORMAT_R8G8B8A8_UNORM\n"
            "clear_color_bits=0,1065353216,0,1065353216\n");
  error.clear();
  if (int failed =
          Expect(!LoadDriverCommand(bad.string(), &command, &error),
                 "unsupported command kind was accepted"))
    return failed;
  if (int failed =
          Expect(error.find("unsupported driver command") != std::string::npos,
                 "unsupported command error did not name the cause"))
    return failed;

  std::filesystem::remove(good);
  std::filesystem::remove(draw);
  std::filesystem::remove(quad);
  std::filesystem::remove(primitives);
  std::filesystem::remove(discard);
  std::filesystem::remove(bad);
  return 0;
}
