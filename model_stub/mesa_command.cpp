#include "mesa_command.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pvrgpu::stub {
namespace {

constexpr const char *kMesaPocSchema = "pvrgpu.mesa-poc-command.v1";
constexpr const char *kMesaPocDriver = "llvmpipe-gallium-trace-poc";
constexpr const char *kFillSolidRdcSha256 =
    "f3c62f8b703cb01500b814056b47778c85a8f8df41d0e978f18b59152b0c5f89";
constexpr const char *kFillSolidDepthNeverRdcSha256 =
    "4a14ac852e943f2df2457b1d3ffad2ddb5814883282b6a72e5168e94c9d7e3cc";
constexpr const char *kFillSolidDepthNotEqualRdcSha256 =
    "d751bbf6677c39174c90835ad43af49e967967534bbc8de9b87cc7e8a30c0212";
constexpr const char *kFillSolidBlendedRdcSha256 =
    "becc1f553574c1fb0ff3f932ecc6551dcac7209c2a7f02e5e0a2bc42557d0358";
constexpr const char *kTriangleSetupRdcSha256 =
    "9b62113a4a7c548c901bdd0369af5e6353638b6b34a93ff208e4e316c11fa027";
constexpr const char *kTriangleSetupAllCulledRdcSha256 =
    "fae1067d6824b9f7e85a16b480bb92922ba6dfa87026abe4e9e154f6cfd4f503";
constexpr const char *kTriangleSetupHalfCulledRdcSha256 =
    "fd7d9bb51c669c778119c79e40fa0f70e58e15af84354a1b4c5af9d416a6a354";
constexpr const char *kAttributeFetchRdcSha256 =
    "c28a1758f75b6530ed7504c9dc99806e4d60b276e2e4737661734f3d44bc0426";
constexpr const char *kAttributeFetchTwoRdcSha256 =
    "fbe613edc90c847de622816fa59bce21581a4a2065e5e07475ea59345bdf5f50";
constexpr const char *kAttributeFetchFourRdcSha256 =
    "cc04a911f7c40fc79a76bdb839682bb11c4c4790ae8f70286cc83f4db3aae211";
constexpr const char *kAttributeFetchEightRdcSha256 =
    "0d95895a87699cacdadacce46575f32f9077fe467bc6b6a876b2b5fa5faac65b";
constexpr const char *kVaryingsOneRdcSha256 =
    "bc66794e9238766912509aae23aca3e4c4dc13fbde6cba5a7fba09c27af10024";
constexpr const char *kVaryingsTwoRdcSha256 =
    "f8621ee760590779d87948942cd54bf2c515bd58e650c9ba08d578d838376f4a";
constexpr const char *kVaryingsFourRdcSha256 =
    "488bb35b54edc87de83bf879ae03d00df3979a8fcb9d8e8a8a5e80768b0394e9";
constexpr const char *kVaryingsEightRdcSha256 =
    "6b68979fe5bda3794cc2a162d35a296bd18e602720e7e773468bf9500b443e16";
constexpr const char *kFillTexNearestRdcSha256 =
    "a5c931d93c23b1c5f60db2f55bb52da307162cae4a8e7617cb0f13389f6ba628";
constexpr const char *kFillTexBilinearRdcSha256 =
    "e95c079af23f836a66b215df7fc053afec0ed8a059863a43f62127ecee146f9b";
constexpr const char *kFillTexTrilinear01RdcSha256 =
    "4535bd3680ffd9b72ff5de51713aba1740d74151d934cb58c9efc8a9f5cd00ed";
constexpr const char *kFillTexTrilinear04RdcSha256 =
    "7607bee8da39f3af40e2a5826b21d319acee0b44dcf2067db7580de0f85a5a1d";
constexpr const char *kFillTexTrilinear05RdcSha256 =
    "c24649c349619ad5911bced14fca152c14506469f0abefb9fb8b7f877b537562";
constexpr const char *kFillSolidVertexShaderSha256 =
    "e544b9470afe4be75e419bb5b23b81d495beb88fff90bde4056f597311a78ca6";
constexpr const char *kFillSolidFragmentShaderSha256 =
    "f3bbac76eb55a5f460d8bb8878466772762d080ca2d64e4e6c956c4b0525ee8a";
constexpr const char *kTriangleSetupVertexShaderSha256 =
    "39a0f8b26aac73a6c2869f7fecc3ca57edcfa5a7e32f2db653efd83bfbbe4b11";
constexpr const char *kAttributeFetchVertexShaderSha256 =
    "ded668556ddee688da3d0b641d0b6ed443b36350dfa8155f4aaac1e5a0db2d30";
constexpr const char *kAttributeFetchTwoVertexShaderSha256 =
    "079be6f870090f41644831145328a84f02c42c6f031d8e82a93a54bb4b544c6d";
constexpr const char *kAttributeFetchFourVertexShaderSha256 =
    "d5811ed233fa3185955c62056f71dfac12b2493a9788ebe0414b189c4d2bdc41";
constexpr const char *kAttributeFetchEightVertexShaderSha256 =
    "9bddbd076f29f358120f7f8f0853b6260bc322b5b5cfdbba4d77b78f5964e8f5";
constexpr const char *kAttributeFetchFragmentShaderSha256 =
    "864ee2432f98ee6f27913fdf7890f7a041d4702b894931b925373ec12226f6c2";
constexpr const char *kVaryingsOneVertexShaderSha256 =
    "7f8685dfbd01a7059e30ced4da534f6b16407ca483475387af509128541bfce4";
constexpr const char *kVaryingsOneFragmentShaderSha256 =
    "ee077b186eb280018d1115eeed197976932cc690dbc45458330cd8b45d0f478f";
constexpr const char *kVaryingsTwoVertexShaderSha256 =
    "974078e19024d0c32348367e0edf13267e5a4d4676f7f0844ed310c2fe266e5e";
constexpr const char *kVaryingsTwoFragmentShaderSha256 =
    "79f8efcbc2a55c551615d2cfe19f71ce63a1af5f727a9f827ba74e2300c51757";
constexpr const char *kVaryingsFourVertexShaderSha256 =
    "4071b07bce3d9fbf1a20b9546e82318ee7a02c3d276bce391b1f96c018d4c8f8";
constexpr const char *kVaryingsFourFragmentShaderSha256 =
    "5bed2b0ede26a212ddaebb371be2234e51dce2b32f9fb4301252a640915f7649";
constexpr const char *kVaryingsEightVertexShaderSha256 =
    "c6c805bf4318142e19b009d3a648089ffcb6f30d69e58aec6693de2ed20a0271";
constexpr const char *kVaryingsEightFragmentShaderSha256 =
    "5e382c8949869949d8a70ff2e0c2ce6171e3252124059aaf79bd5f8ceb4a097d";
constexpr const char *kFillTexVertexShaderSha256 =
    "f6a20b8f3bdd1d44fdf42b651768b294ad4f5172be6d278b6a211a80f09a0934";
constexpr const char *kFillTexFragmentShaderSha256 =
    "9a80dac94df40be1611bbba9e8c0867dcdf6515de31c4ea2c16eb8fe2ace05e6";

const std::set<std::string> &RequiredFields() {
  static const std::set<std::string> fields = {
      "schema",
      "mesa_driver",
      "manifest_index",
      "case",
      "width",
      "height",
      "rdc_sha256",
      "api_trace_sha256",
      "gallium_trace_sha256",
      "vertex_shader_sha256",
      "fragment_shader_sha256",
      "api_calls",
      "api_draw_calls",
      "primitive",
      "indexed",
      "draw_count",
      "index_size",
      "index_buffer_hex",
      "gallium_draw_calls",
      "gallium_target_draws",
      "gallium_framebuffer_matches",
      "gallium_resource_creates",
      "gallium_state_calls",
      "gallium_constant_bytes",
      "framebuffer_format",
      "clear_color_bits",
      "depth_enabled",
      "depth_write",
      "depth_func",
      "blend_enabled",
      "blend_rgb_func",
      "blend_rgb_src_factor",
      "blend_rgb_dst_factor",
      "blend_alpha_func",
      "blend_alpha_src_factor",
      "blend_alpha_dst_factor",
      "front_ccw",
      "cull_face",
      "vertex_elements",
      "vertex_format",
      "vertex_stride",
      "vertex_offset",
      "vertex_buffer_map",
      "vertex_buffer_hex",
      "vertex_buffer_1_hex",
      "vertex_constant_hex",
      "fragment_constant_hex",
      "texture_format",
      "texture_width",
      "texture_height",
      "texture_levels",
      "texture_mip_sizes",
      "texture_mip_hex",
      "sampler_min_img_filter",
      "sampler_min_mip_filter",
      "sampler_mag_img_filter",
  };
  return fields;
}

bool IsDecimal(const std::string &text) {
  return !text.empty() &&
         std::all_of(text.begin(), text.end(),
                     [](unsigned char c) { return c >= '0' && c <= '9'; });
}

bool ParseU64(const std::string &text, std::uint64_t *value) {
  if (!value || !IsDecimal(text))
    return false;
  try {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(text, &consumed, 10);
    if (consumed != text.size())
      return false;
    *value = static_cast<std::uint64_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseU32(const std::string &text, std::uint32_t *value) {
  std::uint64_t parsed = 0;
  if (!ParseU64(text, &parsed) ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

bool ParseBool01(const std::string &text, bool *value) {
  if (!value || (text != "0" && text != "1"))
    return false;
  *value = text == "1";
  return true;
}

bool IsSha256(const std::string &text) {
  return text.size() == 64 &&
         std::all_of(text.begin(), text.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

unsigned HexNibble(char c) {
  if (c >= '0' && c <= '9')
    return static_cast<unsigned>(c - '0');
  if (c >= 'a' && c <= 'f')
    return static_cast<unsigned>(c - 'a' + 10);
  if (c >= 'A' && c <= 'F')
    return static_cast<unsigned>(c - 'A' + 10);
  return 16;
}

bool DecodeHex(const std::string &text, std::vector<std::uint8_t> *bytes) {
  if (!bytes || text.empty() || text.size() % 2 != 0)
    return false;
  bytes->clear();
  bytes->reserve(text.size() / 2);
  for (std::size_t index = 0; index < text.size(); index += 2) {
    const unsigned high = HexNibble(text[index]);
    const unsigned low = HexNibble(text[index + 1]);
    if (high > 15 || low > 15)
      return false;
    bytes->push_back(static_cast<std::uint8_t>((high << 4U) | low));
  }
  return true;
}

std::uint32_t LoadLittleEndianU32(const std::uint8_t *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

float FloatFromBits(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool ParseClearBits(const std::string &text,
                    std::array<std::uint32_t, 4> *values) {
  if (!values)
    return false;
  std::istringstream input(text);
  std::string part;
  for (std::size_t index = 0; index < values->size(); ++index) {
    if (!std::getline(input, part, ',') || !ParseU32(part, &(*values)[index]))
      return false;
  }
  return !std::getline(input, part, ',');
}

bool ParseU32List(const std::string &text,
                  std::vector<std::uint32_t> *values) {
  if (!values || text.empty())
    return false;
  values->clear();
  std::istringstream input(text);
  std::string part;
  while (std::getline(input, part, ',')) {
    std::uint32_t value = 0;
    if (!ParseU32(part, &value))
      return false;
    values->push_back(value);
  }
  return !values->empty() && text.back() != ',';
}

bool IsPinnedGlbenchTexture(const std::vector<std::uint8_t> &bytes) {
  std::size_t offset = 0;
  std::uint32_t size = 512;
  for (std::uint32_t level = 0; level < 10; ++level) {
    for (std::uint32_t row = 0; row < size; ++row) {
      for (std::uint32_t column = 0; column < size; ++column) {
        if (offset + 4 > bytes.size())
          return false;
        const std::uint8_t pattern =
            static_cast<std::uint8_t>((row ^ column) << level);
        std::array<std::uint8_t, 4> expected = {
            level % 3U != 0U ? pattern : std::uint8_t{0},
            level % 3U != 1U ? pattern : std::uint8_t{0},
            level % 3U != 2U ? pattern : std::uint8_t{0},
            255U,
        };
        if (level == 9 && size == 1)
          expected = {255U, 255U, 255U, 255U};
        if (!std::equal(expected.begin(), expected.end(), bytes.begin() + offset))
          return false;
        offset += expected.size();
      }
    }
    size = std::max(1U, size >> 1U);
  }
  return offset == bytes.size();
}

bool ReadFields(const std::string &path,
                std::map<std::string, std::string> *fields,
                std::string *error) {
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open Mesa POC command capsule: " + path;
    return false;
  }
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line.back() == '\r') {
      *error = "invalid blank/CR line in Mesa POC command capsule";
      return false;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == line.size()) {
      *error = "invalid key=value at Mesa POC command line " +
               std::to_string(line_number);
      return false;
    }
    const std::string key = line.substr(0, separator);
    const std::string value = line.substr(separator + 1);
    if (!RequiredFields().count(key)) {
      *error = "unknown Mesa POC command field: " + key;
      return false;
    }
    if (!fields->emplace(key, value).second) {
      *error = "duplicate Mesa POC command field: " + key;
      return false;
    }
  }
  if (!input.eof()) {
    *error = "failed while reading Mesa POC command capsule";
    return false;
  }
  if (fields->size() != RequiredFields().size()) {
    for (const std::string &field : RequiredFields()) {
      if (!fields->count(field)) {
        *error = "missing Mesa POC command field: " + field;
        return false;
      }
    }
  }
  return true;
}

} // namespace

bool LoadMesaPocCommand(const std::string &path, MesaPocCommand *command,
                        std::string *error) {
  if (!command || !error)
    return false;
  error->clear();
  std::map<std::string, std::string> fields;
  if (!ReadFields(path, &fields, error))
    return false;

  MesaPocCommand parsed;
  parsed.schema = fields["schema"];
  parsed.mesa_driver = fields["mesa_driver"];
  parsed.test_case = fields["case"];
  parsed.rdc_sha256 = fields["rdc_sha256"];
  parsed.api_trace_sha256 = fields["api_trace_sha256"];
  parsed.gallium_trace_sha256 = fields["gallium_trace_sha256"];
  parsed.vertex_shader_sha256 = fields["vertex_shader_sha256"];
  parsed.fragment_shader_sha256 = fields["fragment_shader_sha256"];
  parsed.primitive = fields["primitive"];
  parsed.framebuffer_format = fields["framebuffer_format"];
  parsed.blend_rgb_func = fields["blend_rgb_func"];
  parsed.blend_rgb_src_factor = fields["blend_rgb_src_factor"];
  parsed.blend_rgb_dst_factor = fields["blend_rgb_dst_factor"];
  parsed.blend_alpha_func = fields["blend_alpha_func"];
  parsed.blend_alpha_src_factor = fields["blend_alpha_src_factor"];
  parsed.blend_alpha_dst_factor = fields["blend_alpha_dst_factor"];
  parsed.vertex_format = fields["vertex_format"];
  parsed.texture_format = fields["texture_format"];

  if (!ParseU32(fields["manifest_index"], &parsed.manifest_index) ||
      !ParseU32(fields["width"], &parsed.width) ||
      !ParseU32(fields["height"], &parsed.height) ||
      !ParseU64(fields["api_calls"], &parsed.api_calls) ||
      !ParseU64(fields["api_draw_calls"], &parsed.api_draw_calls) ||
      !ParseBool01(fields["indexed"], &parsed.indexed) ||
      !ParseU32(fields["draw_count"], &parsed.draw_count) ||
      !ParseU32(fields["index_size"], &parsed.index_size) ||
      !ParseU64(fields["gallium_draw_calls"], &parsed.gallium_draw_calls) ||
      !ParseU64(fields["gallium_target_draws"],
                &parsed.gallium_target_draws) ||
      !ParseU64(fields["gallium_framebuffer_matches"],
                &parsed.gallium_framebuffer_matches) ||
      !ParseU64(fields["gallium_resource_creates"],
                &parsed.gallium_resource_creates) ||
      !ParseU64(fields["gallium_state_calls"],
                &parsed.gallium_state_calls) ||
      !ParseU64(fields["gallium_constant_bytes"],
                &parsed.gallium_constant_bytes) ||
      !ParseClearBits(fields["clear_color_bits"], &parsed.clear_color_bits) ||
      !ParseBool01(fields["depth_enabled"], &parsed.depth_enabled) ||
      !ParseBool01(fields["depth_write"], &parsed.depth_write) ||
      !ParseU32(fields["depth_func"], &parsed.depth_func) ||
      !ParseBool01(fields["blend_enabled"], &parsed.blend_enabled) ||
      !ParseBool01(fields["front_ccw"], &parsed.front_ccw) ||
      !ParseU32(fields["cull_face"], &parsed.cull_face) ||
      !ParseU32(fields["vertex_elements"], &parsed.vertex_elements) ||
      !ParseU32(fields["vertex_stride"], &parsed.vertex_stride) ||
      !ParseU32(fields["vertex_offset"], &parsed.vertex_offset) ||
      !ParseU32List(fields["vertex_buffer_map"],
                    &parsed.vertex_buffer_map) ||
      !ParseU32(fields["texture_width"], &parsed.texture_width) ||
      !ParseU32(fields["texture_height"], &parsed.texture_height) ||
      !ParseU32(fields["texture_levels"], &parsed.texture_levels) ||
      !ParseU32(fields["sampler_min_img_filter"],
                &parsed.sampler_min_img_filter) ||
      !ParseU32(fields["sampler_min_mip_filter"],
                &parsed.sampler_min_mip_filter) ||
      !ParseU32(fields["sampler_mag_img_filter"],
                &parsed.sampler_mag_img_filter)) {
    *error = "Mesa POC command capsule contains an invalid integer/state field";
    return false;
  }

  const auto decode_float2 = [](const std::string &text,
                                std::vector<std::uint8_t> *bytes,
                                std::vector<float> *values) {
    if (!bytes || !values || text.size() > 32U * 1024U * 1024U ||
        !DecodeHex(text, bytes) ||
        bytes->size() % (2U * sizeof(float)) != 0 ||
        bytes->size() > 16U * 1024U * 1024U) {
      return false;
    }
    values->clear();
    values->reserve(bytes->size() / sizeof(float));
    for (std::size_t offset = 0; offset < bytes->size(); offset += 4) {
      const float value =
          FloatFromBits(LoadLittleEndianU32(bytes->data() + offset));
      if (!std::isfinite(value))
        return false;
      values->push_back(value);
    }
    return true;
  };
  std::vector<std::uint8_t> vertex_bytes;
  std::vector<std::uint8_t> vertex_bytes_1;
  if (!decode_float2(fields["vertex_buffer_hex"], &vertex_bytes,
                     &parsed.vertex_float2)) {
    *error = "Mesa POC primary VBO is not bounded packed float2 data";
    return false;
  }
  const bool has_second_vertex_buffer = fields["vertex_buffer_1_hex"] != "NONE";
  if (has_second_vertex_buffer &&
      !decode_float2(fields["vertex_buffer_1_hex"], &vertex_bytes_1,
                     &parsed.vertex_float2_1)) {
    *error = "Mesa POC secondary VBO is not bounded packed float2 data";
    return false;
  }
  if (parsed.vertex_buffer_map.size() != parsed.vertex_elements ||
      std::any_of(parsed.vertex_buffer_map.begin(),
                  parsed.vertex_buffer_map.end(),
                  [&](std::uint32_t index) {
                    return index > (has_second_vertex_buffer ? 1U : 0U);
                  }) ||
      std::find(parsed.vertex_buffer_map.begin(), parsed.vertex_buffer_map.end(),
                0U) == parsed.vertex_buffer_map.end() ||
      (has_second_vertex_buffer &&
       std::find(parsed.vertex_buffer_map.begin(),
                 parsed.vertex_buffer_map.end(), 1U) ==
           parsed.vertex_buffer_map.end())) {
    *error = "Mesa POC vertex buffer map is inconsistent with its payloads";
    return false;
  }
  if (fields["vertex_constant_hex"] != "NONE" &&
      (fields["vertex_constant_hex"].size() > 32 ||
       !DecodeHex(fields["vertex_constant_hex"], &parsed.vertex_constants))) {
    *error = "Mesa POC vertex constant is not a bounded byte payload";
    return false;
  }
  if (fields["fragment_constant_hex"] != "NONE" &&
      (!DecodeHex(fields["fragment_constant_hex"],
                  &parsed.fragment_constants) ||
       parsed.fragment_constants.size() != 16)) {
    *error = "Mesa POC fragment constant must be NONE or one vec4";
    return false;
  }

  if (parsed.texture_format == "NONE") {
    if (parsed.texture_width != 0 || parsed.texture_height != 0 ||
        parsed.texture_levels != 0 || fields["texture_mip_sizes"] != "NONE" ||
        fields["texture_mip_hex"] != "NONE" ||
        parsed.sampler_min_img_filter != 0 ||
        parsed.sampler_min_mip_filter != 0 ||
        parsed.sampler_mag_img_filter != 0) {
      *error = "Mesa POC non-texture command contains texture state";
      return false;
    }
  } else {
    if (parsed.texture_format != "PIPE_FORMAT_R8G8B8A8_UNORM" ||
        parsed.texture_width == 0 || parsed.texture_height == 0 ||
        parsed.texture_levels == 0 || parsed.texture_levels > 10 ||
        !ParseU32List(fields["texture_mip_sizes"],
                      &parsed.texture_mip_sizes) ||
        parsed.texture_mip_sizes.size() != parsed.texture_levels ||
        fields["texture_mip_hex"].size() > 32U * 1024U * 1024U ||
        !DecodeHex(fields["texture_mip_hex"], &parsed.texture_bytes) ||
        parsed.texture_bytes.size() > 16U * 1024U * 1024U) {
      *error = "Mesa POC texture payload is malformed or out of bounds";
      return false;
    }
    std::uint64_t expected_total = 0;
    for (std::uint32_t level = 0; level < parsed.texture_levels; ++level) {
      const std::uint64_t expected_size =
          static_cast<std::uint64_t>(
              std::max(1U, parsed.texture_width >> level)) *
          std::max(1U, parsed.texture_height >> level) * 4U;
      if (parsed.texture_mip_sizes[level] != expected_size) {
        *error = "Mesa POC texture mip size does not match RGBA8 dimensions";
        return false;
      }
      expected_total += expected_size;
    }
    if (expected_total != parsed.texture_bytes.size()) {
      *error = "Mesa POC texture payload size does not match its mip table";
      return false;
    }
  }

  if (parsed.indexed) {
    std::vector<std::uint8_t> index_bytes;
    const std::uint64_t expected_index_bytes =
        static_cast<std::uint64_t>(parsed.draw_count) * parsed.index_size;
    if (parsed.index_size != sizeof(std::uint16_t) ||
        expected_index_bytes > 16U * 1024U * 1024U ||
        !DecodeHex(fields["index_buffer_hex"], &index_bytes) ||
        index_bytes.size() != expected_index_bytes) {
      *error = "Mesa POC indexed draw has invalid uint16 index bytes";
      return false;
    }
    parsed.indices.reserve(parsed.draw_count);
    const std::size_t vertex_count = parsed.vertex_float2.size() / 2;
    for (std::size_t offset = 0; offset < index_bytes.size(); offset += 2) {
      const std::uint16_t index = static_cast<std::uint16_t>(
          index_bytes[offset] |
          (static_cast<std::uint16_t>(index_bytes[offset + 1]) << 8U));
      if (index >= vertex_count) {
        *error = "Mesa POC index references outside the captured VBO";
        return false;
      }
      parsed.indices.push_back(index);
    }
  } else if (parsed.index_size != 0 || fields["index_buffer_hex"] != "NONE") {
    *error = "Mesa POC non-indexed draw unexpectedly contains index bytes";
    return false;
  }

  const bool hashes_valid =
      IsSha256(parsed.rdc_sha256) && IsSha256(parsed.api_trace_sha256) &&
      IsSha256(parsed.gallium_trace_sha256) &&
      IsSha256(parsed.vertex_shader_sha256) &&
      IsSha256(parsed.fragment_shader_sha256);
  const std::array<std::uint32_t, 4> black_clear_bits = {
      0, 0, 0, UINT32_C(0x3f800000)};
  const std::array<std::uint32_t, 4> green_clear_bits = {
      0, UINT32_C(0x3f800000), 0, UINT32_C(0x3f800000)};
  const std::array<std::uint32_t, 4> red_fragment_bits = {
      UINT32_C(0x3f800000), 0, 0, UINT32_C(0x3f800000)};
  const std::array<std::uint32_t, 4> orange_fragment_bits = {
      UINT32_C(0x3f800000), UINT32_C(0x3f000000), 0,
      UINT32_C(0x3f800000)};
  const std::array<std::uint32_t, 4> cyan_fragment_bits = {
      0, UINT32_C(0x3f000000), UINT32_C(0x3f000000),
      UINT32_C(0x3f800000)};
  const auto fragment_matches = [&](const auto &expected) {
    if (parsed.fragment_constants.size() != expected.size() * sizeof(std::uint32_t))
      return false;
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (LoadLittleEndianU32(parsed.fragment_constants.data() + index * 4) !=
          expected[index]) {
        return false;
      }
    }
    return true;
  };
  const bool unblended =
      !parsed.blend_enabled &&
      parsed.blend_rgb_src_factor == "PIPE_BLENDFACTOR_UNKNOWN" &&
      parsed.blend_rgb_dst_factor == "PIPE_BLENDFACTOR_UNKNOWN" &&
      parsed.blend_alpha_src_factor == "PIPE_BLENDFACTOR_UNKNOWN" &&
      parsed.blend_alpha_dst_factor == "PIPE_BLENDFACTOR_UNKNOWN";
  const bool all_primary_bindings =
      std::all_of(parsed.vertex_buffer_map.begin(),
                  parsed.vertex_buffer_map.end(),
                  [](std::uint32_t index) { return index == 0; });
  const bool no_texture_resources =
      parsed.vertex_float2_1.empty() && parsed.vertex_constants.empty() &&
      parsed.texture_format == "NONE" && parsed.texture_width == 0 &&
      parsed.texture_height == 0 && parsed.texture_levels == 0 &&
      parsed.texture_mip_sizes.empty() && parsed.texture_bytes.empty();
  const bool common_raster_contract =
      parsed.schema == kMesaPocSchema && parsed.mesa_driver == kMesaPocDriver &&
      parsed.width == 512 && parsed.height == 512 && hashes_valid &&
      parsed.api_draw_calls == 1 && parsed.gallium_draw_calls >= 1 &&
      parsed.gallium_target_draws >= 1 &&
      parsed.gallium_target_draws <= parsed.gallium_draw_calls &&
      parsed.gallium_framebuffer_matches >= 1 &&
      parsed.gallium_resource_creates >= 1 && parsed.gallium_state_calls >= 1 &&
      parsed.framebuffer_format == "PIPE_FORMAT_R8G8B8A8_UNORM" &&
      parsed.blend_rgb_func == "PIPE_BLEND_ADD" &&
      parsed.blend_alpha_func == "PIPE_BLEND_ADD" &&
      parsed.vertex_format == "PIPE_FORMAT_R32G32_FLOAT" &&
      parsed.vertex_stride == 8 && parsed.vertex_offset == 0;
  const bool fill_shape =
      parsed.vertex_shader_sha256 == kFillSolidVertexShaderSha256 &&
      parsed.fragment_shader_sha256 == kFillSolidFragmentShaderSha256 &&
      parsed.api_calls == 20 && parsed.primitive == "triangle_strip" &&
      !parsed.indexed && parsed.index_size == 0 && parsed.draw_count == 4 &&
      vertex_bytes.size() == 8U * sizeof(float) &&
      all_primary_bindings && no_texture_resources &&
      fragment_matches(red_fragment_bits) &&
      parsed.gallium_constant_bytes >= 16 &&
      parsed.clear_color_bits == black_clear_bits && !parsed.front_ccw &&
      parsed.cull_face == 0;
  const bool triangle_geometry =
      parsed.vertex_shader_sha256 == kTriangleSetupVertexShaderSha256 &&
      parsed.fragment_shader_sha256 == kFillSolidFragmentShaderSha256 &&
      parsed.api_calls == 24 && parsed.primitive == "triangles" &&
      parsed.indexed && parsed.index_size == sizeof(std::uint16_t) &&
      parsed.draw_count == 98304 && parsed.indices.size() == 98304 &&
      vertex_bytes.size() == 16641U * 2U * sizeof(float) &&
      all_primary_bindings && no_texture_resources &&
      parsed.gallium_constant_bytes >= 16 &&
      parsed.clear_color_bits == green_clear_bits && !parsed.depth_enabled &&
      !parsed.depth_write && parsed.depth_func == 0 && unblended &&
      !parsed.front_ccw && parsed.vertex_elements == 1;
  const bool attribute_geometry =
      parsed.fragment_shader_sha256 == kAttributeFetchFragmentShaderSha256 &&
      parsed.api_calls == 24 && parsed.primitive == "triangles" &&
      parsed.indexed && parsed.index_size == sizeof(std::uint16_t) &&
      parsed.draw_count == 24576 && parsed.indices.size() == 24576 &&
      vertex_bytes.size() == 4225U * 2U * sizeof(float) &&
      all_primary_bindings && no_texture_resources &&
      parsed.fragment_constants.empty() && parsed.gallium_constant_bytes == 0 &&
      parsed.clear_color_bits == green_clear_bits && !parsed.depth_enabled &&
      !parsed.depth_write && parsed.depth_func == 0 && unblended &&
      !parsed.front_ccw && parsed.cull_face == 2;
  const bool varying_geometry =
      parsed.api_calls == 24 && parsed.primitive == "triangles" &&
      parsed.indexed && parsed.index_size == sizeof(std::uint16_t) &&
      parsed.draw_count == 96 && parsed.indices.size() == 96 &&
      vertex_bytes.size() == 25U * 2U * sizeof(float) &&
      all_primary_bindings && no_texture_resources &&
      parsed.fragment_constants.empty() && parsed.gallium_constant_bytes == 0 &&
      parsed.clear_color_bits == green_clear_bits && !parsed.depth_enabled &&
      !parsed.depth_write && parsed.depth_func == 0 && unblended &&
      !parsed.front_ccw && parsed.cull_face == 0 &&
      parsed.vertex_elements == 1;
  const auto vertex_constant_matches = [&](std::uint32_t expected) {
    return parsed.vertex_constants.size() == sizeof(expected) &&
           LoadLittleEndianU32(parsed.vertex_constants.data()) == expected;
  };
  const std::vector<std::uint32_t> expected_texture_mips = {
      1048576U, 262144U, 65536U, 16384U, 4096U,
      1024U,    256U,    64U,    16U,    4U,
  };
  const bool texture_geometry =
      parsed.vertex_shader_sha256 == kFillTexVertexShaderSha256 &&
      parsed.fragment_shader_sha256 == kFillTexFragmentShaderSha256 &&
      parsed.primitive == "triangle_strip" &&
      !parsed.indexed && parsed.index_size == 0 && parsed.draw_count == 4 &&
      vertex_bytes.size() == 8U * sizeof(float) &&
      vertex_bytes_1.size() == 8U * sizeof(float) &&
      parsed.vertex_buffer_map == std::vector<std::uint32_t>{0, 1} &&
      parsed.vertex_constants.size() == sizeof(std::uint32_t) &&
      parsed.fragment_constants.empty() && parsed.gallium_constant_bytes >= 4 &&
      parsed.texture_format == "PIPE_FORMAT_R8G8B8A8_UNORM" &&
      parsed.texture_width == 512 && parsed.texture_height == 512 &&
      parsed.texture_levels == 10 &&
      parsed.texture_mip_sizes == expected_texture_mips &&
      IsPinnedGlbenchTexture(parsed.texture_bytes) &&
      parsed.clear_color_bits == black_clear_bits && !parsed.depth_enabled &&
      !parsed.depth_write && parsed.depth_func == 0 && unblended &&
      !parsed.front_ccw && parsed.cull_face == 0 &&
      parsed.vertex_elements == 2;
  const bool sequential_case_contract =
      (fill_shape && parsed.manifest_index == 1 &&
       parsed.test_case == "fill_solid" &&
       parsed.rdc_sha256 == kFillSolidRdcSha256 && !parsed.depth_enabled &&
       !parsed.depth_write && parsed.depth_func == 0 && unblended &&
       parsed.vertex_elements == 1) ||
      (fill_shape && parsed.manifest_index == 2 &&
       parsed.test_case == "fill_solid_depth_never" &&
       parsed.rdc_sha256 == kFillSolidDepthNeverRdcSha256 &&
       parsed.depth_enabled && parsed.depth_write && parsed.depth_func == 0 &&
       unblended && parsed.vertex_elements == 1) ||
      (fill_shape && parsed.manifest_index == 3 &&
       parsed.test_case == "fill_solid_depth_neq" &&
       parsed.rdc_sha256 == kFillSolidDepthNotEqualRdcSha256 &&
       parsed.depth_enabled && parsed.depth_write && parsed.depth_func == 5 &&
       unblended && parsed.vertex_elements == 1) ||
      (fill_shape && parsed.manifest_index == 4 &&
       parsed.test_case == "fill_solid_blended" &&
       parsed.rdc_sha256 == kFillSolidBlendedRdcSha256 &&
       !parsed.depth_enabled && !parsed.depth_write && parsed.depth_func == 0 &&
       parsed.blend_enabled &&
       parsed.blend_rgb_src_factor == "PIPE_BLENDFACTOR_SRC_ALPHA" &&
       parsed.blend_rgb_dst_factor == "PIPE_BLENDFACTOR_INV_SRC_ALPHA" &&
       parsed.blend_alpha_src_factor == "PIPE_BLENDFACTOR_SRC_ALPHA" &&
       parsed.blend_alpha_dst_factor == "PIPE_BLENDFACTOR_INV_SRC_ALPHA" &&
       parsed.vertex_elements == 1) ||
      (triangle_geometry && fragment_matches(orange_fragment_bits) &&
       parsed.manifest_index == 5 &&
       parsed.test_case == "triangle_setup" &&
       parsed.rdc_sha256 == kTriangleSetupRdcSha256 &&
       parsed.cull_face == 0) ||
      (triangle_geometry && fragment_matches(orange_fragment_bits) &&
       parsed.manifest_index == 6 &&
       parsed.test_case == "triangle_setup_all_culled" &&
       parsed.rdc_sha256 == kTriangleSetupAllCulledRdcSha256 &&
       parsed.cull_face == 2) ||
      (triangle_geometry && fragment_matches(cyan_fragment_bits) &&
       parsed.manifest_index == 7 &&
       parsed.test_case == "triangle_setup_half_culled" &&
       parsed.rdc_sha256 == kTriangleSetupHalfCulledRdcSha256 &&
       parsed.cull_face == 2) ||
      (attribute_geometry && parsed.manifest_index == 8 &&
       parsed.test_case == "attribute_fetch_shader" &&
       parsed.rdc_sha256 == kAttributeFetchRdcSha256 &&
       parsed.vertex_shader_sha256 == kAttributeFetchVertexShaderSha256 &&
       parsed.vertex_elements == 1) ||
      (attribute_geometry && parsed.manifest_index == 9 &&
       parsed.test_case == "attribute_fetch_shader_2_attr" &&
       parsed.rdc_sha256 == kAttributeFetchTwoRdcSha256 &&
       parsed.vertex_shader_sha256 == kAttributeFetchTwoVertexShaderSha256 &&
       parsed.vertex_elements == 2) ||
      (attribute_geometry && parsed.manifest_index == 10 &&
       parsed.test_case == "attribute_fetch_shader_4_attr" &&
       parsed.rdc_sha256 == kAttributeFetchFourRdcSha256 &&
       parsed.vertex_shader_sha256 == kAttributeFetchFourVertexShaderSha256 &&
       parsed.vertex_elements == 4) ||
      (attribute_geometry && parsed.manifest_index == 11 &&
       parsed.test_case == "attribute_fetch_shader_8_attr" &&
       parsed.rdc_sha256 == kAttributeFetchEightRdcSha256 &&
       parsed.vertex_shader_sha256 == kAttributeFetchEightVertexShaderSha256 &&
       parsed.vertex_elements == 8) ||
      (varying_geometry && parsed.manifest_index == 12 &&
       parsed.test_case == "varyings_shader_1" &&
       parsed.rdc_sha256 == kVaryingsOneRdcSha256 &&
       parsed.vertex_shader_sha256 == kVaryingsOneVertexShaderSha256 &&
       parsed.fragment_shader_sha256 == kVaryingsOneFragmentShaderSha256) ||
      (varying_geometry && parsed.manifest_index == 13 &&
       parsed.test_case == "varyings_shader_2" &&
       parsed.rdc_sha256 == kVaryingsTwoRdcSha256 &&
       parsed.vertex_shader_sha256 == kVaryingsTwoVertexShaderSha256 &&
       parsed.fragment_shader_sha256 == kVaryingsTwoFragmentShaderSha256) ||
      (varying_geometry && parsed.manifest_index == 14 &&
       parsed.test_case == "varyings_shader_4" &&
       parsed.rdc_sha256 == kVaryingsFourRdcSha256 &&
       parsed.vertex_shader_sha256 == kVaryingsFourVertexShaderSha256 &&
       parsed.fragment_shader_sha256 == kVaryingsFourFragmentShaderSha256) ||
      (varying_geometry && parsed.manifest_index == 15 &&
       parsed.test_case == "varyings_shader_8" &&
       parsed.rdc_sha256 == kVaryingsEightRdcSha256 &&
       parsed.vertex_shader_sha256 == kVaryingsEightVertexShaderSha256 &&
       parsed.fragment_shader_sha256 == kVaryingsEightFragmentShaderSha256) ||
      (texture_geometry && parsed.manifest_index == 16 &&
       parsed.test_case == "fill_tex_nearest" &&
       parsed.rdc_sha256 == kFillTexNearestRdcSha256 &&
       parsed.api_calls == 37 &&
       vertex_constant_matches(UINT32_C(0x3f800000)) &&
       parsed.sampler_min_img_filter == 0 &&
       parsed.sampler_min_mip_filter == 2 &&
       parsed.sampler_mag_img_filter == 0) ||
      (texture_geometry && parsed.manifest_index == 17 &&
       parsed.test_case == "fill_tex_bilinear" &&
       parsed.rdc_sha256 == kFillTexBilinearRdcSha256 &&
       parsed.api_calls == 39 &&
       vertex_constant_matches(UINT32_C(0x3f800000)) &&
       parsed.sampler_min_img_filter == 1 &&
       parsed.sampler_min_mip_filter == 2 &&
       parsed.sampler_mag_img_filter == 1) ||
      (texture_geometry && parsed.manifest_index == 18 &&
       parsed.test_case == "fill_tex_trilinear_linear_01" &&
       parsed.rdc_sha256 == kFillTexTrilinear01RdcSha256 &&
       parsed.api_calls == 42 &&
       vertex_constant_matches(UINT32_C(0x3f6ed917)) &&
       parsed.sampler_min_img_filter == 1 &&
       parsed.sampler_min_mip_filter == 1 &&
       parsed.sampler_mag_img_filter == 1) ||
      (texture_geometry && parsed.manifest_index == 19 &&
       parsed.test_case == "fill_tex_trilinear_linear_04" &&
       parsed.rdc_sha256 == kFillTexTrilinear04RdcSha256 &&
       parsed.api_calls == 41 &&
       vertex_constant_matches(UINT32_C(0x3f420c4a)) &&
       parsed.sampler_min_img_filter == 1 &&
       parsed.sampler_min_mip_filter == 1 &&
       parsed.sampler_mag_img_filter == 1) ||
      (texture_geometry && parsed.manifest_index == 20 &&
       parsed.test_case == "fill_tex_trilinear_linear_05" &&
       parsed.rdc_sha256 == kFillTexTrilinear05RdcSha256 &&
       parsed.api_calls == 40 &&
       vertex_constant_matches(UINT32_C(0x3f350481)) &&
       parsed.sampler_min_img_filter == 1 &&
       parsed.sampler_min_mip_filter == 1 &&
       parsed.sampler_mag_img_filter == 1);
  if (!common_raster_contract || !sequential_case_contract) {
    *error = "Mesa POC command is outside the enabled sequential contracts";
    return false;
  }

  parsed.enabled = true;
  *command = std::move(parsed);
  return true;
}

} // namespace pvrgpu::stub
