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
        output.color_formats_explicit = true; output.bytes_per_pixel = 16;
        Check(!output.ColorFormatMatches(0,names[0].c_str()), "explicit fixed4B transport");
      }
    }
    auto a = Command({formats[0],formats[1],formats[2],formats[0]});
    for (const auto *invalid : {"", "UNKNOWN", "PIPE_FORMAT_R8G8B8A8_SRGB", "PIPE_FORMAT_R32_UINT", "PIPE_FORMAT_R32G32B32A32_FLOAT"}) {
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
