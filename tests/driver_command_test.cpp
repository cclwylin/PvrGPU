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

bool ReplaceOnce(std::string *text, const std::string &before,
                 const std::string &after) {
  if (!text)
    return false;
  const std::size_t offset = text->find(before);
  if (offset == std::string::npos)
    return false;
  text->replace(offset, before.size(), after);
  return true;
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

  const std::filesystem::path textured = TempFile("textured.txt");
  const std::filesystem::path texture_sidecar = TempFile("texture.rgba8");
  std::string textured_text =
      "schema=pvrgpu.driver-command.v1\n"
      "producer=pvrgpu-gallium-driver\n"
      "command=draw_textured_triangles\n"
      "case=glmark2.effect2d\n"
      "frame=1\n"
      "framebuffer_width=80\n"
      "framebuffer_height=60\n"
      "width=80\n"
      "height=60\n"
      "format=PIPE_FORMAT_R8G8B8A8_UNORM\n"
      "clear_color_bits=0,0,0,1065353216\n";
  for (std::uint32_t vertex = 0; vertex < 6; ++vertex) {
    textured_text += "vertex" + std::to_string(vertex) +
                     "_bits=3212836864,3212836864\n";
    textured_text += "texcoord" + std::to_string(vertex) +
                     "_bits=0,1065353216\n";
  }
  textured_text +=
      "texture_width=800\n"
      "texture_height=600\n"
      "texture_rgba8_path=" +
      texture_sidecar.string() + "\n";
  WriteText(textured, textured_text);
  error.clear();
  if (int failed =
          Expect(LoadDriverCommand(textured.string(), &command, &error), error))
    return failed;
  if (int failed = Expect(
          command.command == "draw_textured_triangles" &&
              command.framebuffer_width == 80 &&
              command.framebuffer_height == 60 &&
              command.vertex_bits[5][0] == UINT32_C(3212836864) &&
              command.texcoord_bits[5][1] == UINT32_C(1065353216) &&
              command.texture_width == 800 && command.texture_height == 600 &&
              command.texture_rgba8_path == texture_sidecar.string(),
          "wrong textured triangle command payload"))
    return failed;

  const std::filesystem::path pco = TempFile("pco.txt");
  const std::string pco_text =
      "schema=pvrgpu.driver-command.v1\n"
      "producer=pvrgpu-gallium-driver\n"
      "command=draw_pco_triangles\n"
      "case=glmark2.conditionals\n"
      "frame=1\n"
      "framebuffer_width=80\n"
      "framebuffer_height=60\n"
      "width=80\n"
      "height=60\n"
      "format=PIPE_FORMAT_R8G8B8A8_UNORM\n"
      "clear_color_bits=0,0,0,1065353216\n"
      "raw_vertex_data_size=73728\n"
      "vertex_stride=12\n"
      "vertex_count=6144\n"
      "first_vertex=0\n"
      "instance_count=1\n"
      "primitive_mode=4\n"
      "indexed=0\n"
      "vertex_pco_size=520\n"
      "fragment_pco_size=520\n"
      "vertex_shared_count=16\n"
      "vertex_shared_words=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
      "fragment_shared_count=4\n"
      "fragment_shared_words=0,0,0,0\n"
      "vertex_pco_abi=10,4,4,0,16,0,16,0\n"
      "fragment_pco_abi=4,0,0,0,4,0,4,0\n"
      "position_linkage=0,4,0,0\n"
      "viewport_scale_bits=1109393408,1106247680,1056964608\n"
      "viewport_translate_bits=1109393408,1106247680,1056964608\n"
      "raster_state=0,2,0,0,0,0,0,1,0,0,1,1,0\n"
      "sample_mask=4294967295\n"
      "color_state=15,0,1\n"
      "depth_state=1,1,3,1065353216,1\n";
  WriteText(pco, pco_text);
  error.clear();
  if (int failed =
          Expect(LoadDriverCommand(pco.string(), &command, &error), error))
    return failed;
  if (int failed = Expect(
          command.command == "draw_pco_triangles" &&
              command.vertex_count == 6144 && command.vertex_stride == 12 &&
              command.declared_raw_vertex_data_size == 73728 &&
              command.declared_vertex_pco_size == 520 &&
              command.declared_fragment_pco_size == 520 &&
              command.vertex_shared.size() == 16 &&
              command.fragment_shared.size() == 4 &&
              command.raw_vertex_data.empty() && command.vertex_pco.empty() &&
              command.fragment_pco.empty(),
          "wrong PCO audit-metadata command payload"))
    return failed;

  const std::filesystem::path pco_counters =
      TempFile("pco-counters.txt");
  WriteText(pco_counters,
            pco_text +
                "draw_count=180\n"
                "ia_vertices=3370\n"
                "ia_primitives=3010\n"
                "vs_invocations=3370\n"
                "gs_invocations=0\n"
                "gs_primitives=0\n"
                "clip_invocations=3010\n"
                "clip_primitives=3004\n"
                "hs_invocations=0\n"
                "ds_invocations=0\n"
                "cs_invocations=0\n"
                "ps_invocations=1553\n"
                "setup_triangles=3004\n"
                "semantic_texel_fetches=0\n");
  error.clear();
  if (int failed = Expect(
          LoadDriverCommand(pco_counters.string(), &command, &error), error))
    return failed;
  if (int failed = Expect(
          command.draw_count == 180 && command.ia_vertices == 3370 &&
              command.ia_primitives == 3010 &&
              command.vs_invocations == 3370 &&
              command.clip_invocations == 3010 &&
              command.clip_primitives == 3004 &&
              command.ps_invocations == 1553 &&
              command.setup_triangles == 3004,
          "wrong optional PCO sequence counter metadata"))
    return failed;

  const std::filesystem::path pco_texture =
      TempFile("pco-texture.txt");
  std::string pco_texture_text = pco_text;
  if (!ReplaceOnce(&pco_texture_text, "raw_vertex_data_size=73728",
                   "raw_vertex_data_size=1152") ||
      !ReplaceOnce(&pco_texture_text, "vertex_stride=12",
                   "vertex_stride=32") ||
      !ReplaceOnce(&pco_texture_text, "vertex_count=6144",
                   "vertex_count=36") ||
      !ReplaceOnce(&pco_texture_text, "vertex_pco_size=520",
                   "vertex_pco_size=752") ||
      !ReplaceOnce(&pco_texture_text, "fragment_pco_size=520",
                   "fragment_pco_size=304") ||
      !ReplaceOnce(
          &pco_texture_text,
          "vertex_pco_abi=10,4,4,0,16,0,16,0",
          "vertex_pco_abi=11,12,7,0,16,0,16,0") ||
      !ReplaceOnce(
          &pco_texture_text,
          "fragment_pco_abi=4,0,0,0,4,0,4,0",
          "fragment_pco_abi=8,0,0,16,4,0,0,0") ||
      !ReplaceOnce(&pco_texture_text, "position_linkage=0,4,0,0",
                   "position_linkage=0,4,0,4\n"
                   "varying_linkage=4,3,4,12")) {
    return 1;
  }
  pco_texture_text +=
      "sampled_texture_count=1\n"
      "sampled_texture_bytes_size=1048576\n"
      "sampled_texture_width=512\n"
      "sampled_texture_height=512\n"
      "sampled_texture_row_pitch=2048\n"
      "sampled_texture_format=PIPE_FORMAT_R8G8B8X8_UNORM\n"
      "sampled_texture_mip_count=1\n";
  WriteText(pco_texture, pco_texture_text);
  error.clear();
  if (int failed = Expect(
          LoadDriverCommand(pco_texture.string(), &command, &error), error))
    return failed;
  if (int failed = Expect(
          command.sampled_texture_count == 1 &&
              command.declared_sampled_texture_bytes_size == 1048576 &&
              command.sampled_texture_width == 512 &&
              command.sampled_texture_height == 512 &&
              command.sampled_texture_row_pitch == 2048 &&
              command.sampled_texture_format ==
                  "PIPE_FORMAT_R8G8B8X8_UNORM" &&
              command.sampled_texture_mip_count == 1 &&
              command.varying_output_start == 4 &&
              command.varying_output_count == 3 &&
              command.fragment_varying_start == 4 &&
              command.fragment_varying_count == 12,
          "wrong PCO sampled-texture audit metadata"))
    return failed;

  const std::filesystem::path bad_pco = TempFile("bad-pco.txt");
  std::string bad_pco_text = pco_text;
  const std::size_t raw_size = bad_pco_text.find("raw_vertex_data_size=73728");
  if (raw_size == std::string::npos)
    return 1;
  bad_pco_text.replace(raw_size, std::string("raw_vertex_data_size=73728").size(),
                       "raw_vertex_data_size=73727");
  WriteText(bad_pco, bad_pco_text);
  error.clear();
  if (int failed = Expect(
          !LoadDriverCommand(bad_pco.string(), &command, &error) &&
              error.find("strict 80x60 conditionals profile") !=
                  std::string::npos,
          "malformed PCO VBO byte count was accepted"))
    return failed;

  bad_pco_text = pco_text;
  const std::size_t vertex_pco_size =
      bad_pco_text.find("vertex_pco_size=520");
  if (vertex_pco_size == std::string::npos)
    return 1;
  bad_pco_text.replace(vertex_pco_size,
                       std::string("vertex_pco_size=520").size(),
                       "vertex_pco_size=519");
  WriteText(bad_pco, bad_pco_text);
  error.clear();
  if (int failed = Expect(
          !LoadDriverCommand(bad_pco.string(), &command, &error) &&
              error.find("strict 80x60 conditionals profile") !=
                  std::string::npos,
          "wrong 519-byte conditionals VS profile was accepted"))
    return failed;

  bad_pco_text = pco_text;
  const std::size_t fragment_pco_size =
      bad_pco_text.find("fragment_pco_size=520");
  if (fragment_pco_size == std::string::npos)
    return 1;
  bad_pco_text.replace(fragment_pco_size,
                       std::string("fragment_pco_size=520").size(),
                       "fragment_pco_size=519");
  WriteText(bad_pco, bad_pco_text);
  error.clear();
  if (int failed = Expect(
          !LoadDriverCommand(bad_pco.string(), &command, &error) &&
              error.find("strict 80x60 conditionals profile") !=
                  std::string::npos,
          "wrong 519-byte conditionals FS profile was accepted"))
    return failed;

  const std::filesystem::path bad_pco_linkage =
      TempFile("bad-pco-linkage.txt");
  std::string bad_pco_linkage_text = pco_text;
  const std::size_t position_linkage =
      bad_pco_linkage_text.find("position_linkage=0,4,0,0");
  if (position_linkage == std::string::npos)
    return 1;
  bad_pco_linkage_text.replace(
      position_linkage, std::string("position_linkage=0,4,0,0").size(),
      "position_linkage=0,4,0,4");
  WriteText(bad_pco_linkage, bad_pco_linkage_text);
  error.clear();
  if (int failed = Expect(
          !LoadDriverCommand(bad_pco_linkage.string(), &command, &error) &&
              error.find("strict conditionals profile") != std::string::npos,
          "interpolated fragment-position linkage was accepted"))
    return failed;

  const std::filesystem::path bad_pco_abi = TempFile("bad-pco-abi.txt");
  std::string bad_pco_abi_text = pco_text;
  const std::size_t fragment_abi =
      bad_pco_abi_text.find("fragment_pco_abi=4,0,0,0,4,0,4,0");
  if (fragment_abi == std::string::npos)
    return 1;
  bad_pco_abi_text.replace(
      fragment_abi,
      std::string("fragment_pco_abi=4,0,0,0,4,0,4,0").size(),
      "fragment_pco_abi=5,0,0,0,4,0,4,0");
  WriteText(bad_pco_abi, bad_pco_abi_text);
  error.clear();
  if (int failed = Expect(
          !LoadDriverCommand(bad_pco_abi.string(), &command, &error) &&
              error.find("strict conditionals profile") != std::string::npos,
          "wrong fragment temporary ABI count was accepted"))
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
  std::filesystem::remove(textured);
  std::filesystem::remove(pco);
  std::filesystem::remove(pco_counters);
  std::filesystem::remove(bad_pco);
  std::filesystem::remove(bad_pco_linkage);
  std::filesystem::remove(bad_pco_abi);
  std::filesystem::remove(primitives);
  std::filesystem::remove(discard);
  std::filesystem::remove(bad);
  return 0;
}
