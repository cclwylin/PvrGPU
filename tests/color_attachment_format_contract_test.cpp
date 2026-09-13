// API31 color metadata ownership/alias/readback contracts, no model execution.
#include "model_types.h"
#include "model_runner.h"
#include "pvrgpu_color_formats.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned checks = 0;
void Check(bool value, const char *why) {
  ++checks;
  if (!value) throw std::runtime_error(why);
}
constexpr std::array<const char *, 3> formats = {
    "PIPE_FORMAT_R8G8B8A8_UNORM", "PIPE_FORMAT_R10G10B10A2_UNORM",
    "PIPE_FORMAT_B10G10R10A2_UNORM"};
pvrgpu::stub::DriverCommand Command(const std::vector<std::string> &names) {
  pvrgpu::stub::DriverCommand command;
  command.format = names[0];
  command.render_target_count = static_cast<std::uint32_t>(names.size());
  command.color_attachment_formats = names;
  command.framebuffer_width = 4; command.framebuffer_height = 3;
  return command;
}
}

int main() {
  using namespace pvrgpu::stub;
  try {
    for (unsigned targets = 1; targets <= 4; ++targets) {
      unsigned combinations = 1;
      for (unsigned target = 0; target < targets; ++target) combinations *= 3;
      for (unsigned choice = 0; choice < combinations; ++choice) {
        unsigned remaining = choice;
        std::vector<std::string> names;
        std::array<const char *, 4> borrowed{};
        for (unsigned target = 0; target < targets; ++target) {
          borrowed[target] = formats[remaining % 3]; names.emplace_back(borrowed[target]);
          remaining /= 3;
        }
        Check(!pvrgpu_color_formats_error(names[0].c_str(), targets, targets, borrowed.data()), "public valid vector");
        auto a = Command(names), b = a;
        Check(DriverColorAttachmentFormatsAreValid(a), "owned valid vector");
        Check(DriverColorAttachmentFormatsMatch(a, b), "same vector");
        Check(EffectiveDriverColorAttachmentFormats(a) == names, "effective vector");
        b.color_attachment_source_command_index = 0;
        std::vector<std::uint64_t> colors, depths;
        Check(ResolveSequenceAttachmentAddresses({a,b}, &colors, &depths) && colors[0] == colors[1], "exact alias");
        for (unsigned target = 0; target < targets; ++target) {
          auto bad = b;
          bad.color_attachment_formats[target] = names[target] == formats[0] ? formats[1] : formats[0];
          if (!target) bad.format = bad.color_attachment_formats[0];
          Check(!DriverColorAttachmentFormatsMatch(a,bad), "same-size format mismatch");
          Check(!ResolveSequenceAttachmentAddresses({a,bad}, &colors, &depths), "alias refuses same-size mismatch");
        }
        auto legacy = a; legacy.color_attachment_formats.clear();
        bool homogeneous = true;
        for (const auto &name : names) homogeneous &= name == names[0];
        Check(DriverColorAttachmentFormatsMatch(a,legacy) == homogeneous, "legacy effective equivalence");
        ModelFramebuffer output;
        output.width = 4; output.height = 3; output.pixels.resize(48);
        output.extra.resize(targets - 1, std::vector<std::uint8_t>(48));
        output.color_formats = names; output.color_formats_explicit = true;
        output.bytes_per_pixel_per_target.assign(targets, 4);
        for (unsigned target = 0; target < targets; ++target) {
          Check(output.ColorFormatMatches(target,names[target].c_str()), "explicit readback exact identity");
          Check(!output.ColorFormatMatches(target,nullptr), "explicit readback missing identity");
          Check(!output.ColorFormatMatches(target,""), "empty requested identity");
          Check(!output.ColorFormatMatches(target,"PIPE_FORMAT_R32_UINT"), "same-size integer mismatch");
        }
        Check(!output.ColorFormatMatches(targets,names[0].c_str()), "readback target bounds");
        output.color_formats_explicit = false;
        Check(output.ColorFormatMatches(0,nullptr), "legacy missing identity accepted");
        Check(output.ColorFormatMatches(0,names[0].c_str()), "legacy supplied identity checked");
        output.color_formats_explicit = true;
        output.bytes_per_pixel_per_target[0] = 16;
        Check(!output.ColorFormatMatches(0,names[0].c_str()), "explicit fixed4B transport");
      }
    }
    auto a = Command({formats[0],formats[1],formats[2],formats[0]});
    auto canonical = Command({formats[0], "PIPE_FORMAT_R8_SNORM",
                              "PIPE_FORMAT_R16G16_FLOAT",
                              "PIPE_FORMAT_R11G11B10_FLOAT"});
    Check(DriverColorAttachmentFormatsAreValid(canonical),
          "canonical native precision vector");
    Check(DriverColorAttachmentBytesPerPixel(
              canonical.color_attachment_formats[1]) == 16 &&
              DriverColorAttachmentBytesPerPixel(
                  canonical.color_attachment_formats[2]) == 16 &&
              DriverColorAttachmentBytesPerPixel(
                  canonical.color_attachment_formats[3]) == 16,
          "canonical native formats use RGBA32F transport");
    auto exact_integer = Command({"PIPE_FORMAT_R8_UINT",
                                  "PIPE_FORMAT_R16G16_SINT",
                                  "PIPE_FORMAT_R10G10B10A2_UINT",
                                  "PIPE_FORMAT_R32G32B32A32_UINT"});
    Check(DriverColorAttachmentFormatsAreValid(exact_integer),
          "native integer codec vector");
    Check(DriverColorAttachmentBytesPerPixel(
              "PIPE_FORMAT_R32G32B32A32_UNORM") == 32,
          "RGBA32_UNORM exact transport width");
    Check(DriverColorAttachmentMaximumBytesPerPixel(
              Command({formats[0], "PIPE_FORMAT_R32G32B32A32_UNORM"})) == 32,
          "mixed MRT extent ignored a wide later target");
    ModelFramebuffer wide;
    wide.width = wide.height = wide.sample_count = wide.layer_count = 1;
    wide.bytes_per_pixel = 32;
    wide.pixels.resize(32);
    wide.color_formats = {"PIPE_FORMAT_R32G32B32A32_UNORM"};
    wide.color_formats_explicit = true;
    wide.bytes_per_pixel_per_target = {32};
    Check(wide.valid() &&
              wide.ColorFormatMatches(0,
                  "PIPE_FORMAT_R32G32B32A32_UNORM"),
          "RGBA32_UNORM 32-byte model framebuffer contract");
    for (const auto *invalid : {"", "UNKNOWN", "PIPE_FORMAT_R9G9B9E5_FLOAT"}) {
      auto bad = a; bad.color_attachment_formats[1] = invalid;
      Check(!DriverColorAttachmentFormatsAreValid(bad), "unsupported mix");
      ModelFramebuffer output; output.extra.resize(3); output.color_formats = bad.color_attachment_formats; output.color_formats_explicit = true;
      Check(!output.ColorFormatMatches(0,formats[0]), "invalid cached explicit codec");
    }
    for (unsigned count : {0U,1U,2U,3U,5U,UINT32_MAX}) {
      auto bad = a; bad.render_target_count = count;
      Check(!DriverColorAttachmentFormatsAreValid(bad), "mismatched count");
    }
    auto bad = a; bad.format = formats[1];
    Check(!DriverColorAttachmentFormatsAreValid(bad), "target0 common mismatch");
    std::array<const char *,4> pointers{};
    Check(!pvrgpu_color_formats_error(formats[0],4,0,pointers.data()), "public legacy null tail");
    for (unsigned target=0; target<4; ++target) {
      pointers[target] = "";
      Check(pvrgpu_color_formats_error(formats[0],4,0,pointers.data()), "empty string not canonical null");
      pointers[target] = nullptr;
    }
    pointers[0] = formats[0];
    Check(pvrgpu_color_formats_error(formats[0],4,4,pointers.data()), "missing active pointer");
    std::cout << "color attachment format contracts: PASS " << checks << " checks\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
  return 0;
}
