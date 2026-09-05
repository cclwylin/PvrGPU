#include "driver_command.h"
#include "shader/pco_iss.h"

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
#include <vector>

namespace pvrgpu::stub {
namespace {

constexpr const char *kDriverCommandSchema = "pvrgpu.driver-command.v1";
constexpr const char *kDriverCommandProducer = "pvrgpu-gallium-driver";
constexpr const char *kClearColorCommand = "clear_color";
constexpr const char *kDrawTriangleCommand = "draw_triangle";
constexpr const char *kDrawIndexedQuadCommand = "draw_indexed_quad";
constexpr const char *kDrawTexturedTrianglesCommand =
    "draw_textured_triangles";
constexpr const char *kDrawPcoTrianglesCommand = "draw_pco_triangles";
constexpr const char *kRgba8Format = "PIPE_FORMAT_R8G8B8A8_UNORM";
constexpr const char *kRgbx8Format = "PIPE_FORMAT_R8G8B8X8_UNORM";
constexpr const char *kBgrx8Format = "PIPE_FORMAT_B8G8R8X8_UNORM";
constexpr const char *kR5G6B5Format = "PIPE_FORMAT_R5G6B5_UNORM";
constexpr const char *kB5G6R5Format = "PIPE_FORMAT_B5G6R5_UNORM";
constexpr const char *kR10G10B10A2Format = "PIPE_FORMAT_R10G10B10A2_UNORM";
constexpr const char *kB10G10R10A2Format = "PIPE_FORMAT_B10G10R10A2_UNORM";

const std::set<std::string> &KnownFields() {
  static const std::set<std::string> fields = {
      "schema", "producer", "command", "case", "frame",
      "raw_index_data_size", "index_size", "first_index", "base_vertex",
      "render_target_count", "vertex_attribute_count",
      "framebuffer_width", "framebuffer_height", "width",
      "height", "format", "clear_color_bits", "fragment_color_bits",
      "vertex0_bits", "vertex1_bits", "vertex2_bits",
      "vertex3_bits", "vertex4_bits", "vertex5_bits",
      "texcoord0_bits", "texcoord1_bits", "texcoord2_bits",
      "texcoord3_bits", "texcoord4_bits", "texcoord5_bits",
      "texture_width", "texture_height", "texture_rgba8_path",
      "raw_vertex_data_size", "vertex_stride", "vertex_count",
      "first_vertex", "instance_count", "primitive_mode", "indexed",
      "vertex_pco_size", "fragment_pco_size", "vertex_shared_count",
      "vertex_shared_words", "fragment_shared_count",
      "fragment_shared_words", "vertex_pco_abi", "fragment_pco_abi",
      "position_linkage", "varying_linkage", "viewport_scale_bits",
      "viewport_translate_bits", "raster_state", "scissor_rect",
      "primitive_width", "point_size_output", "sample_mask",
      "color_state", "depth_state",
      "sampled_texture_count", "sampled_texture_bytes_size",
      "sampled_texture_width", "sampled_texture_height",
      "sampled_texture_row_pitch", "sampled_texture_format",
      "sampled_texture_mip_count",
      "draw_count", "index_count", "unique_vertices", "primitive_count",
      "clip_primitives", "setup_triangles", "semantic_texel_fetches",
      "ia_vertices", "ia_primitives", "vs_invocations", "clip_invocations",
      "gs_invocations", "gs_primitives", "ps_invocations",
      "hs_invocations", "ds_invocations", "cs_invocations",
      "framebuffer_rgba8_path",
  };
  return fields;
}

// A line width or point size the model can widen to: finite, at least one
// device pixel, and bounded so the expanded quad stays inside the surface
// arithmetic clip/cull performs.
bool DriverPrimitiveWidthIsValid(std::uint32_t bits) {
  // All-zero bits mean the command states no width, which is the GLES default
  // of one device pixel.
  if (bits == 0)
    return true;
  float width = 0.0F;
  static_assert(sizeof(width) == sizeof(bits));
  std::memcpy(&width, &bits, sizeof(width));
  return std::isfinite(width) && width >= 1.0F && width <= 1024.0F;
}

const std::set<std::string> &BaseFields() {
  static const std::set<std::string> fields = {
      "schema", "producer", "command", "case",
      "frame",  "width",    "height",  "format",
  };
  return fields;
}

const std::set<std::string> &ClearColorFields() {
  static const std::set<std::string> fields = {
      "schema",           "producer", "command",
      "case",             "frame",    "width",
      "height",           "format",   "clear_color_bits",
  };
  return fields;
}

const std::set<std::string> &DrawTriangleFields() {
  static const std::set<std::string> fields = {
      "schema",              "producer",       "command",
      "case",                "frame",          "width",
      "height",              "format",         "clear_color_bits",
      "fragment_color_bits", "vertex0_bits",   "vertex1_bits",
      "vertex2_bits",
  };
  return fields;
}

const std::set<std::string> &DrawIndexedQuadFields() {
  static const std::set<std::string> fields = {
      "schema",         "producer",       "command",
      "case",               "frame",              "framebuffer_width",
      "framebuffer_height", "width",              "height",
      "format",             "clear_color_bits",   "draw_count",
      "index_count",        "unique_vertices",    "primitive_count",
      "clip_primitives",    "setup_triangles",    "semantic_texel_fetches",
  };
  return fields;
}

const std::set<std::string> &DrawTexturedTrianglesFields() {
  static const std::set<std::string> fields = {
      "schema",              "producer",           "command",
      "case",                "frame",              "framebuffer_width",
      "framebuffer_height",  "width",              "height",
      "format",              "clear_color_bits",   "vertex0_bits",
      "vertex1_bits",        "vertex2_bits",       "vertex3_bits",
      "vertex4_bits",        "vertex5_bits",       "texcoord0_bits",
      "texcoord1_bits",      "texcoord2_bits",     "texcoord3_bits",
      "texcoord4_bits",      "texcoord5_bits",     "texture_width",
      "texture_height",      "texture_rgba8_path",
  };
  return fields;
}

const std::set<std::string> &DrawPcoTrianglesFields() {
  static const std::set<std::string> fields = {
      "schema", "producer", "command", "case", "frame",
      "framebuffer_width", "framebuffer_height", "width", "height",
      "format", "clear_color_bits", "raw_vertex_data_size",
      "vertex_stride", "vertex_count", "first_vertex", "instance_count",
      "primitive_mode", "indexed", "vertex_pco_size", "fragment_pco_size",
      "vertex_shared_count", "vertex_shared_words",
      "fragment_shared_count", "fragment_shared_words", "vertex_pco_abi",
      "fragment_pco_abi", "position_linkage", "viewport_scale_bits",
      "viewport_translate_bits", "raster_state", "scissor_rect",
      "primitive_width", "point_size_output", "sample_mask",
      "color_state", "depth_state",
  };
  return fields;
}

bool IsDecimal(const std::string &text) {
  return !text.empty() &&
         std::all_of(text.begin(), text.end(),
                     [](unsigned char c) { return c >= '0' && c <= '9'; });
}

bool IsSupportedDriverCommandFormat(const std::string &format) {
  return format == kRgba8Format || format == kRgbx8Format ||
         format == kBgrx8Format || format == kR5G6B5Format ||
         format == kB5G6R5Format || format == kR10G10B10A2Format ||
         format == kB10G10R10A2Format;
}

bool PcoSingleDrawResolutionSupported(std::uint32_t framebuffer_width,
                                      std::uint32_t framebuffer_height,
                                      std::uint32_t width,
                                      std::uint32_t height) {
  return width == framebuffer_width && height == framebuffer_height &&
         ((framebuffer_width == 80 && framebuffer_height == 60) ||
          (framebuffer_width == 800 && framebuffer_height == 600) ||
          (framebuffer_width == 512 && framebuffer_height == 512));
}

std::array<std::uint32_t, 3> PcoViewportBits(
    std::uint32_t framebuffer_width, std::uint32_t framebuffer_height) {
  const std::array<float, 3> values = {
      static_cast<float>(framebuffer_width) * 0.5F,
      static_cast<float>(framebuffer_height) * 0.5F,
      0.5F,
  };
  std::array<std::uint32_t, 3> bits{};
  static_assert(sizeof(values) == sizeof(bits));
  std::memcpy(bits.data(), values.data(), sizeof(bits));
  return bits;
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

bool ParseI32(const std::string &text, std::int32_t *value) {
  if (text.empty())
    return false;
  const bool negative = text[0] == '-';
  std::uint64_t magnitude = 0;
  if (!ParseU64(negative ? text.substr(1) : text, &magnitude))
    return false;
  // Two's complement allows one more magnitude on the negative side.
  const std::uint64_t limit = negative ? 2147483648ULL : 2147483647ULL;
  if (magnitude > limit)
    return false;
  *value = negative
               ? static_cast<std::int32_t>(-static_cast<std::int64_t>(magnitude))
               : static_cast<std::int32_t>(magnitude);
  return true;
}

// An absent optional field keeps the caller's default rather than failing.
bool ParseOptionalU64(const std::map<std::string, std::string> &fields,
                      const char *name, std::uint64_t *value) {
  const auto entry = fields.find(name);
  return entry == fields.end() || ParseU64(entry->second, value);
}

bool ParseOptionalU32(const std::map<std::string, std::string> &fields,
                      const char *name, std::uint32_t *value) {
  const auto entry = fields.find(name);
  return entry == fields.end() || ParseU32(entry->second, value);
}

bool ParseOptionalI32(const std::map<std::string, std::string> &fields,
                      const char *name, std::int32_t *value) {
  const auto entry = fields.find(name);
  return entry == fields.end() || ParseI32(entry->second, value);
}

template <std::size_t N>
bool ParseU32List(const std::string &text,
                  std::array<std::uint32_t, N> *values) {
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

bool ParseU32Vector(const std::string &text, std::size_t count,
                    std::vector<std::uint32_t> *values) {
  if (!values)
    return false;
  values->clear();
  if (count == 0)
    return text.empty();
  if (text.empty())
    return false;
  std::istringstream input(text);
  std::string part;
  values->reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::uint32_t value = 0;
    if (!std::getline(input, part, ',') || !ParseU32(part, &value))
      return false;
    values->push_back(value);
  }
  return !std::getline(input, part, ',');
}

bool RequireExactFields(const std::map<std::string, std::string> &fields,
                        const std::set<std::string> &required,
                        const std::string &command,
                        std::string *error) {
  std::vector<std::string> missing;
  for (const std::string &field : required) {
    if (!fields.count(field))
      missing.push_back(field);
  }
  if (!missing.empty()) {
    std::ostringstream message;
    message << "driver command is missing required field(s): ";
    for (std::size_t index = 0; index < missing.size(); ++index) {
      if (index != 0)
        message << ", ";
      message << missing[index];
    }
    *error = message.str();
    return false;
  }

  for (const auto &entry : fields) {
    const bool optional_pco_counter =
        command == kDrawPcoTrianglesCommand &&
        (entry.first == "draw_count" || entry.first == "ia_vertices" ||
         entry.first == "ia_primitives" ||
         entry.first == "vs_invocations" ||
         entry.first == "gs_invocations" ||
         entry.first == "gs_primitives" ||
         entry.first == "clip_invocations" ||
         entry.first == "clip_primitives" ||
         entry.first == "hs_invocations" ||
         entry.first == "ds_invocations" ||
         entry.first == "cs_invocations" ||
         entry.first == "ps_invocations" ||
         entry.first == "setup_triangles" ||
         entry.first == "semantic_texel_fetches");
    const bool optional_pco_texture =
        command == kDrawPcoTrianglesCommand &&
        (entry.first == "sampled_texture_count" ||
         entry.first == "sampled_texture_bytes_size" ||
         entry.first == "sampled_texture_width" ||
         entry.first == "sampled_texture_height" ||
         entry.first == "sampled_texture_row_pitch" ||
         entry.first == "sampled_texture_format" ||
         entry.first == "sampled_texture_mip_count");
    const bool optional_pco_linkage =
        command == kDrawPcoTrianglesCommand &&
        entry.first == "varying_linkage";
    // Index payload: present only on an indexed PCO triangles draw.
    const bool optional_pco_render_targets =
        command == kDrawPcoTrianglesCommand &&
        (entry.first == "render_target_count" ||
         entry.first == "vertex_attribute_count");
    const bool optional_pco_index =
        command == kDrawPcoTrianglesCommand &&
        (entry.first == "raw_index_data_size" ||
         entry.first == "index_size" || entry.first == "index_count" ||
         entry.first == "first_index" || entry.first == "base_vertex");
    if (!required.count(entry.first) && !optional_pco_counter &&
        !optional_pco_texture && !optional_pco_linkage &&
        !optional_pco_index && !optional_pco_render_targets) {
      *error = "field is not valid for " + command +
               " driver command: " + entry.first;
      return false;
    }
  }
  return true;
}

bool ParseOptionalU32(const std::map<std::string, std::string> &fields,
                      const std::string &field,
                      std::uint32_t *value,
                      std::string *error) {
  const auto entry = fields.find(field);
  if (entry == fields.end())
    return true;
  if (ParseU32(entry->second, value))
    return true;
  *error = "driver command " + field + " must be a uint32 value";
  return false;
}

bool RequireBaseFields(const std::map<std::string, std::string> &fields,
                       std::string *error) {
  std::vector<std::string> missing;
  for (const std::string &field : BaseFields()) {
    if (!fields.count(field))
      missing.push_back(field);
  }
  if (missing.empty())
    return true;

  std::ostringstream message;
  message << "driver command is missing required field(s): ";
  for (std::size_t index = 0; index < missing.size(); ++index) {
    if (index != 0)
      message << ", ";
    message << missing[index];
  }
  *error = message.str();
  return false;
}

bool ReadFields(const std::string &path,
                std::map<std::string, std::string> *fields,
                std::string *error) {
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open driver command: " + path;
    return false;
  }
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line.back() == '\r') {
      *error = "invalid blank/CR line in driver command";
      return false;
    }
    const std::size_t separator = line.find('=');
    const std::string key =
        separator == std::string::npos ? std::string{} :
                                         line.substr(0, separator);
    const bool empty_shared_words =
        separator != std::string::npos && separator + 1 == line.size() &&
        (key == "vertex_shared_words" ||
         key == "fragment_shared_words");
    if (separator == std::string::npos || separator == 0 ||
        (separator + 1 == line.size() && !empty_shared_words)) {
      *error = "invalid key=value at driver command line " +
               std::to_string(line_number);
      return false;
    }
    const std::string value = line.substr(separator + 1);
    if (!KnownFields().count(key)) {
      *error = "unknown driver command field: " + key;
      return false;
    }
    if (!fields->emplace(key, value).second) {
      *error = "duplicate driver command field: " + key;
      return false;
    }
  }
  if (!input.eof()) {
    *error = "failed while reading driver command";
    return false;
  }
  return true;
}

} // namespace

bool LoadDriverCommand(const std::string &path, DriverCommand *command,
                       std::string *error) {
  if (!command || !error)
    return false;
  std::map<std::string, std::string> fields;
  if (!ReadFields(path, &fields, error))
    return false;
  if (!RequireBaseFields(fields, error))
    return false;

  DriverCommand parsed;
  parsed.enabled = true;
  parsed.schema = fields["schema"];
  parsed.producer = fields["producer"];
  parsed.command = fields["command"];
  parsed.test_case = fields["case"];
  parsed.format = fields["format"];

  if (parsed.schema != kDriverCommandSchema) {
    *error = "driver command schema mismatch: " + parsed.schema;
    return false;
  }
  if (parsed.producer != kDriverCommandProducer) {
    *error = "unsupported driver command producer: " + parsed.producer;
    return false;
  }
  if (parsed.command != kClearColorCommand &&
      parsed.command != kDrawTriangleCommand &&
      parsed.command != kDrawIndexedQuadCommand &&
      parsed.command != kDrawTexturedTrianglesCommand &&
      parsed.command != kDrawPcoTrianglesCommand) {
    *error = "unsupported driver command: " + parsed.command;
    return false;
  }
  if (parsed.test_case.empty()) {
    *error = "driver command case must not be empty";
    return false;
  }
  if (!IsSupportedDriverCommandFormat(parsed.format)) {
    *error = "unsupported driver command format: " + parsed.format;
    return false;
  }
  if (!ParseU32(fields["frame"], &parsed.frame) || parsed.frame != 1) {
    *error = "driver command frame must be exactly 1";
    return false;
  }
  if (!ParseU32(fields["width"], &parsed.width) || parsed.width == 0) {
    *error = "driver command width must be positive";
    return false;
  }
  if (!ParseU32(fields["height"], &parsed.height) || parsed.height == 0) {
    *error = "driver command height must be positive";
    return false;
  }
  if (parsed.command == kDrawIndexedQuadCommand ||
      parsed.command == kDrawTexturedTrianglesCommand ||
      parsed.command == kDrawPcoTrianglesCommand) {
    if (!ParseU32(fields["framebuffer_width"], &parsed.framebuffer_width) ||
        parsed.framebuffer_width == 0) {
      *error = parsed.command + " framebuffer_width must be positive";
      return false;
    }
    if (!ParseU32(fields["framebuffer_height"], &parsed.framebuffer_height) ||
        parsed.framebuffer_height == 0) {
      *error = parsed.command + " framebuffer_height must be positive";
      return false;
    }
    if (parsed.width > parsed.framebuffer_width ||
        parsed.height > parsed.framebuffer_height) {
      *error = parsed.command +
               " viewport width/height must fit framebuffer";
      return false;
    }
  } else {
    parsed.framebuffer_width = parsed.width;
    parsed.framebuffer_height = parsed.height;
  }
  const std::set<std::string> &required =
      parsed.command == kClearColorCommand
          ? ClearColorFields()
          : parsed.command == kDrawTriangleCommand
                ? DrawTriangleFields()
                : parsed.command == kDrawIndexedQuadCommand
                      ? DrawIndexedQuadFields()
                      : parsed.command == kDrawTexturedTrianglesCommand
                            ? DrawTexturedTrianglesFields()
                            : DrawPcoTrianglesFields();
  if (!RequireExactFields(fields, required, parsed.command, error))
    return false;

  if (!ParseU32List(fields["clear_color_bits"], &parsed.clear_color_bits)) {
    *error = "driver command clear_color_bits must contain four uint32 values";
    return false;
  }
  if (parsed.command == kDrawTriangleCommand) {
    if (!ParseU32List(fields["fragment_color_bits"],
                      &parsed.fragment_color_bits)) {
      *error =
          "driver command fragment_color_bits must contain four uint32 values";
      return false;
    }
    if (!ParseU32List(fields["vertex0_bits"], &parsed.vertex_bits[0]) ||
        !ParseU32List(fields["vertex1_bits"], &parsed.vertex_bits[1]) ||
        !ParseU32List(fields["vertex2_bits"], &parsed.vertex_bits[2])) {
      *error =
          "driver command vertexN_bits must contain two uint32 values";
      return false;
    }
  }
  if (parsed.command == kDrawIndexedQuadCommand) {
    if (!ParseU32(fields["draw_count"], &parsed.draw_count) ||
        parsed.draw_count == 0) {
      *error = "driver command draw_count must be positive";
      return false;
    }
    if (!ParseU32(fields["index_count"], &parsed.index_count) ||
        parsed.index_count != 6) {
      *error = "draw_indexed_quad index_count must be exactly 6";
      return false;
    }
    if (!ParseU32(fields["unique_vertices"], &parsed.unique_vertices) ||
        parsed.unique_vertices != 4) {
      *error = "draw_indexed_quad unique_vertices must be exactly 4";
      return false;
    }
    if (!ParseU32(fields["primitive_count"], &parsed.primitive_count) ||
        parsed.primitive_count != 2) {
      *error = "draw_indexed_quad primitive_count must be exactly 2";
      return false;
    }
    if (!ParseU32(fields["clip_primitives"], &parsed.clip_primitives) ||
        parsed.clip_primitives > parsed.primitive_count) {
      *error =
          "draw_indexed_quad clip_primitives must be within primitive_count";
      return false;
    }
    if (!ParseU32(fields["setup_triangles"], &parsed.setup_triangles) ||
        parsed.setup_triangles > parsed.primitive_count) {
      *error =
          "draw_indexed_quad setup_triangles must be within primitive_count";
      return false;
    }
    if (!ParseU64(fields["semantic_texel_fetches"],
                  &parsed.semantic_texel_fetches)) {
      *error = "driver command semantic_texel_fetches must be a uint64 value";
      return false;
    }
  }
  if (parsed.command == kDrawTexturedTrianglesCommand) {
    for (std::size_t vertex = 0; vertex < parsed.vertex_bits.size(); ++vertex) {
      const std::string suffix = std::to_string(vertex) + "_bits";
      if (!ParseU32List(fields["vertex" + suffix],
                        &parsed.vertex_bits[vertex]) ||
          !ParseU32List(fields["texcoord" + suffix],
                        &parsed.texcoord_bits[vertex])) {
        *error =
            "draw_textured_triangles vertexN_bits and texcoordN_bits must "
            "contain two uint32 values";
        return false;
      }
    }
    if (!ParseU32(fields["texture_width"], &parsed.texture_width) ||
        parsed.texture_width == 0 || parsed.texture_width > 16384U ||
        !ParseU32(fields["texture_height"], &parsed.texture_height) ||
        parsed.texture_height == 0 || parsed.texture_height > 16384U) {
      *error =
          "draw_textured_triangles texture dimensions must be within 1..16384";
      return false;
    }
    if (fields["texture_rgba8_path"].empty()) {
      *error = "draw_textured_triangles texture_rgba8_path is empty";
      return false;
    }
    parsed.texture_rgba8_path = fields["texture_rgba8_path"];
  }
  if (parsed.command == kDrawPcoTrianglesCommand) {
    std::uint32_t vertex_shared_count = 0;
    std::uint32_t fragment_shared_count = 0;
    std::array<std::uint32_t, 8> vertex_abi{};
    std::array<std::uint32_t, 8> fragment_abi{};
    std::array<std::uint32_t, 4> position_linkage{};
    std::array<std::uint32_t, 4> varying_linkage{};
    std::array<std::uint32_t, 13> raster_state{};
    std::array<std::uint32_t, 4> scissor_rect{};
    std::array<std::uint32_t, 2> primitive_width{};
    std::array<std::uint32_t, 2> point_size_output{};
    std::array<std::uint32_t, 3> color_state{};
    std::array<std::uint32_t, 5> depth_state{};
    if (!ParseU64(fields["raw_vertex_data_size"],
                  &parsed.declared_raw_vertex_data_size) ||
        !ParseU32(fields["vertex_stride"], &parsed.vertex_stride) ||
        parsed.vertex_stride == 0 ||
        !ParseU32(fields["vertex_count"], &parsed.vertex_count) ||
        parsed.vertex_count == 0 ||
        !ParseU32(fields["first_vertex"], &parsed.first_vertex) ||
        !ParseU32(fields["instance_count"], &parsed.instance_count) ||
        parsed.instance_count != 1 ||
        !ParseU32(fields["primitive_mode"], &parsed.primitive_mode) ||
        (parsed.primitive_mode != 4 && parsed.primitive_mode != 5 &&
         parsed.primitive_mode != 6) ||
        !ParseU32(fields["indexed"], &parsed.indexed) || parsed.indexed > 1 ||
        !ParseOptionalU32(fields, "vertex_attribute_count",
                          &parsed.vertex_attribute_count) ||
        parsed.vertex_attribute_count > 16 ||
        !ParseOptionalU32(fields, "render_target_count",
                          &parsed.render_target_count) ||
        parsed.render_target_count == 0 || parsed.render_target_count > 4 ||
        !ParseOptionalU64(fields, "raw_index_data_size",
                          &parsed.declared_raw_index_data_size) ||
        !ParseOptionalU32(fields, "index_size", &parsed.index_size) ||
        !ParseOptionalU32(fields, "index_count", &parsed.index_count) ||
        !ParseOptionalU32(fields, "first_index", &parsed.first_index) ||
        !ParseOptionalI32(fields, "base_vertex", &parsed.base_vertex) ||
        !ParseU64(fields["vertex_pco_size"],
                  &parsed.declared_vertex_pco_size) ||
        parsed.declared_vertex_pco_size == 0 ||
        parsed.declared_vertex_pco_size > kDriverPcoMaximumBinaryBytes ||
        !ParseU64(fields["fragment_pco_size"],
                  &parsed.declared_fragment_pco_size) ||
        parsed.declared_fragment_pco_size == 0 ||
        parsed.declared_fragment_pco_size > kDriverPcoMaximumBinaryBytes ||
        !ParseU32(fields["vertex_shared_count"], &vertex_shared_count) ||
        vertex_shared_count > kPcoMaximumSharedCount ||
        !ParseU32Vector(fields["vertex_shared_words"], vertex_shared_count,
                        &parsed.vertex_shared) ||
        !ParseU32(fields["fragment_shared_count"], &fragment_shared_count) ||
        fragment_shared_count > kPcoMaximumSharedCount ||
        !ParseU32Vector(fields["fragment_shared_words"],
                        fragment_shared_count, &parsed.fragment_shared) ||
        !ParseU32List(fields["vertex_pco_abi"], &vertex_abi) ||
        !ParseU32List(fields["fragment_pco_abi"], &fragment_abi) ||
        !ParseU32List(fields["position_linkage"], &position_linkage) ||
        !ParseU32List(fields["viewport_scale_bits"],
                      &parsed.viewport_scale_bits) ||
        !ParseU32List(fields["viewport_translate_bits"],
                      &parsed.viewport_translate_bits) ||
        !ParseU32List(fields["raster_state"], &raster_state) ||
        !ParseU32List(fields["scissor_rect"], &scissor_rect) ||
        !ParseU32List(fields["primitive_width"], &primitive_width) ||
        !ParseU32List(fields["point_size_output"], &point_size_output) ||
        !ParseU32(fields["sample_mask"], &parsed.sample_mask) ||
        !ParseU32List(fields["color_state"], &color_state) ||
        !ParseU32List(fields["depth_state"], &depth_state)) {
      *error = "draw_pco_triangles metadata is malformed or outside the "
               "bounded native PCO profile";
      return false;
    }
    const auto varying = fields.find("varying_linkage");
    if (varying != fields.end() &&
        !ParseU32List(varying->second, &varying_linkage)) {
      *error = "draw_pco_triangles varying_linkage must contain four uint32 "
               "values";
      return false;
    }
    parsed.vertex_pco_abi = {vertex_abi[0], vertex_abi[1], vertex_abi[2],
                             vertex_abi[3], vertex_abi[4], vertex_abi[5],
                             vertex_abi[6], vertex_abi[7]};
    parsed.fragment_pco_abi = {
        fragment_abi[0], fragment_abi[1], fragment_abi[2], fragment_abi[3],
        fragment_abi[4], fragment_abi[5], fragment_abi[6], fragment_abi[7]};
    parsed.position_output_start = position_linkage[0];
    parsed.position_output_count = position_linkage[1];
    parsed.fragment_position_start = position_linkage[2];
    parsed.fragment_position_count = position_linkage[3];
    parsed.varying_output_start = varying_linkage[0];
    parsed.varying_output_count = varying_linkage[1];
    parsed.fragment_varying_start = varying_linkage[2];
    parsed.fragment_varying_count = varying_linkage[3];
    parsed.front_ccw = raster_state[0];
    parsed.cull_face = raster_state[1];
    parsed.fill_front = raster_state[2];
    parsed.fill_back = raster_state[3];
    parsed.scissor = raster_state[4];
    parsed.scissor_x = scissor_rect[0];
    parsed.scissor_y = scissor_rect[1];
    parsed.scissor_width = scissor_rect[2];
    parsed.scissor_height = scissor_rect[3];
    parsed.line_width_bits = primitive_width[0];
    parsed.point_size_bits = primitive_width[1];
    parsed.point_size_output_start = point_size_output[0];
    parsed.point_size_output_count = point_size_output[1];
    if (parsed.point_size_output_count > 1 ||
        (parsed.point_size_output_count == 0 &&
         parsed.point_size_output_start != 0)) {
      *error = "driver command point size output span is invalid";
      return false;
    }
    if (!DriverPrimitiveWidthIsValid(parsed.line_width_bits) ||
        !DriverPrimitiveWidthIsValid(parsed.point_size_bits)) {
      *error = "driver command line width or point size is not a supported "
               "positive value";
      return false;
    }
    if (parsed.scissor != 0) {
      // An enabled scissor must name a non-empty rectangle inside the render
      // target; the model bounds rasterization by it directly.
      if (parsed.scissor_width == 0 || parsed.scissor_height == 0 ||
          static_cast<std::uint64_t>(parsed.scissor_x) + parsed.scissor_width >
              parsed.framebuffer_width ||
          static_cast<std::uint64_t>(parsed.scissor_y) + parsed.scissor_height >
              parsed.framebuffer_height) {
        *error = "driver command scissor rectangle is not inside the render "
                 "target";
        return false;
      }
    } else if (parsed.scissor_x != 0 || parsed.scissor_y != 0 ||
               parsed.scissor_width != 0 || parsed.scissor_height != 0) {
      *error = "driver command carries a scissor rectangle with scissor "
               "disabled";
      return false;
    }
    parsed.rasterizer_discard = raster_state[5];
    parsed.multisample = raster_state[6];
    parsed.half_pixel_center = raster_state[7];
    parsed.bottom_edge_rule = raster_state[8];
    parsed.clip_halfz = raster_state[9];
    parsed.depth_clip_near = raster_state[10];
    parsed.depth_clip_far = raster_state[11];
    parsed.depth_clamp = raster_state[12];
    parsed.color_mask = color_state[0];
    parsed.blend_enable = color_state[1];
    parsed.dither = color_state[2];
    parsed.depth_enable = depth_state[0];
    parsed.depth_write = depth_state[1];
    parsed.depth_func = depth_state[2];
    parsed.depth_clear_bits = depth_state[3];
    parsed.depth_format = depth_state[4];
    if (!PcoSingleDrawResolutionSupported(
            parsed.framebuffer_width, parsed.framebuffer_height,
            parsed.width, parsed.height)) {
      *error = "draw_pco_triangles resolution must use a framebuffer-sized "
               "80x60 or 800x600 viewport";
      return false;
    }
    const std::array<std::uint32_t, 3> expected_viewport =
        PcoViewportBits(parsed.framebuffer_width,
                        parsed.framebuffer_height);
    const std::uint64_t end_vertex =
        static_cast<std::uint64_t>(parsed.first_vertex) +
        parsed.vertex_count;
    const std::uint64_t expected_vertex_bytes =
        end_vertex * parsed.vertex_stride;
    const bool conditionals_geometry =
        parsed.vertex_stride == 12 && parsed.vertex_count == 6144 &&
        parsed.first_vertex == 0 && parsed.primitive_mode == 4;
    const auto abi_bounded = [](const DriverPcoStageAbi &abi,
                                bool allow_zero_temps) {
      return (allow_zero_temps || abi.temps != 0) &&
             abi.temps <= kPcoTemporaryCount &&
             abi.vertex_inputs <= kPcoVertexInputCount &&
             abi.vertex_outputs <= kPcoVertexOutputCount &&
             abi.coefficients <= kPcoMaximumVaryingCoefficientCount &&
             abi.shareds <= kPcoMaximumSharedCount &&
             abi.push_constant_start <= abi.shareds &&
             abi.push_constant_count <=
                 abi.shareds - abi.push_constant_start &&
             abi.entry_offset == 0;
    };
    if (end_vertex > std::numeric_limits<std::uint32_t>::max() ||
        parsed.vertex_stride >
            std::numeric_limits<std::uint64_t>::max() / end_vertex ||
        expected_vertex_bytes != parsed.declared_raw_vertex_data_size ||
        vertex_shared_count != parsed.vertex_pco_abi.shareds ||
        fragment_shared_count != parsed.fragment_pco_abi.shareds ||
        /* A pass-through VS forwarding position and colour uses no temps. */
        !abi_bounded(parsed.vertex_pco_abi, true) ||
        !abi_bounded(parsed.fragment_pco_abi, true) ||
        parsed.position_output_start != 0 ||
        parsed.position_output_count != 4 ||
        parsed.vertex_pco_abi.vertex_outputs !=
            parsed.position_output_count + parsed.varying_output_count ||
        (parsed.varying_output_count != 0 &&
         parsed.varying_output_start != parsed.position_output_count) ||
        parsed.varying_output_count > kDriverPcoMaximumVaryingComponents ||
        parsed.fragment_position_start != 0 ||
        parsed.fragment_pco_abi.coefficients !=
            parsed.fragment_position_count +
                parsed.fragment_varying_count ||
        (parsed.fragment_varying_count != 0 &&
         parsed.fragment_varying_start !=
             parsed.fragment_position_count) ||
        parsed.fragment_varying_count !=
            parsed.varying_output_count * 4U ||
        parsed.front_ccw > 1 || parsed.cull_face > 3 ||
        parsed.fill_front != 0 || parsed.fill_back != 0 ||
        parsed.rasterizer_discard != 0 ||
        parsed.multisample > 1 || parsed.half_pixel_center != 1 ||
        parsed.bottom_edge_rule != 0 || parsed.clip_halfz != 0 ||
        parsed.depth_clip_near != 1 || parsed.depth_clip_far != 1 ||
        parsed.depth_clamp != 0 || parsed.sample_mask != UINT32_MAX ||
        parsed.color_mask > 0x0f || parsed.blend_enable > 1 ||
        parsed.dither != 1 || parsed.depth_enable > 1 ||
        parsed.depth_write > 1 || parsed.depth_func > 7 ||
        parsed.depth_format == 0 ||
        parsed.viewport_scale_bits != expected_viewport ||
        parsed.viewport_translate_bits != expected_viewport) {
      *error = conditionals_geometry
                   ? "draw_pco_triangles metadata is malformed or outside "
                     "the strict supported-resolution conditionals profile "
                     "(80x60 or 800x600)"
                   : "draw_pco_triangles ABI/raster metadata is outside "
                     "model bounds";
      return false;
    }

    if (conditionals_geometry) {
      if (parsed.declared_vertex_pco_size != 520 ||
          parsed.declared_fragment_pco_size != 520 ||
          !DriverPcoStageAbiMatches(parsed.vertex_pco_abi,
                                    kConditionalsVertexPcoAbi) ||
          !DriverPcoStageAbiMatches(parsed.fragment_pco_abi,
                                    kConditionalsFragmentPcoAbi) ||
          parsed.varying_output_count != 0 ||
          parsed.fragment_position_count != 0 ||
          parsed.fragment_varying_count != 0 ||
          parsed.viewport_scale_bits != expected_viewport ||
          parsed.viewport_translate_bits != expected_viewport ||
          parsed.cull_face != 2 || parsed.depth_enable != 1 ||
          parsed.depth_write != 1 || parsed.depth_func != 3 ||
          parsed.depth_clear_bits != UINT32_C(0x3f800000)) {
        *error = "draw_pco_triangles ABI/raster metadata does not match the "
                 "strict supported-resolution conditionals profile "
                 "(80x60 or 800x600)";
        return false;
      }
    }

    const auto sampled_count = fields.find("sampled_texture_count");
    if (sampled_count != fields.end()) {
      if (!ParseU32(sampled_count->second,
                    &parsed.sampled_texture_count) ||
          parsed.sampled_texture_count > 1) {
        *error = "draw_pco_triangles sampled_texture_count must be 0 or 1";
        return false;
      }
      const std::array<const char *, 6> texture_fields = {
          "sampled_texture_bytes_size", "sampled_texture_width",
          "sampled_texture_height", "sampled_texture_row_pitch",
          "sampled_texture_format", "sampled_texture_mip_count"};
      for (const char *field : texture_fields) {
        if ((parsed.sampled_texture_count == 1) !=
            (fields.find(field) != fields.end())) {
          *error = "draw_pco_triangles sampled texture fields are incomplete";
          return false;
        }
      }
      if (parsed.sampled_texture_count == 1 &&
          (!ParseU64(fields.at("sampled_texture_bytes_size"),
                     &parsed.declared_sampled_texture_bytes_size) ||
           !ParseU32(fields.at("sampled_texture_width"),
                     &parsed.sampled_texture_width) ||
           !ParseU32(fields.at("sampled_texture_height"),
                     &parsed.sampled_texture_height) ||
           !ParseU32(fields.at("sampled_texture_row_pitch"),
                     &parsed.sampled_texture_row_pitch) ||
           !ParseU32(fields.at("sampled_texture_mip_count"),
                     &parsed.sampled_texture_mip_count) ||
           fields.at("sampled_texture_format").empty())) {
        *error = "draw_pco_triangles sampled texture metadata is malformed";
        return false;
      }
      if (parsed.sampled_texture_count == 1)
        parsed.sampled_texture_format = fields.at("sampled_texture_format");
      if (parsed.sampled_texture_count == 1) {
        const std::uint64_t tight_pitch =
            static_cast<std::uint64_t>(parsed.sampled_texture_width) * 4U;
        const std::uint64_t expected_bytes =
            static_cast<std::uint64_t>(parsed.sampled_texture_row_pitch) *
            parsed.sampled_texture_height;
        if (parsed.sampled_texture_width == 0 ||
            parsed.sampled_texture_height == 0 ||
            parsed.sampled_texture_row_pitch < tight_pitch ||
            parsed.sampled_texture_mip_count == 0 ||
            expected_bytes == 0 ||
            expected_bytes !=
                parsed.declared_sampled_texture_bytes_size) {
          *error = "draw_pco_triangles sampled texture layout is invalid";
          return false;
        }
      }
    } else {
      const std::array<const char *, 6> texture_fields = {
          "sampled_texture_bytes_size", "sampled_texture_width",
          "sampled_texture_height", "sampled_texture_row_pitch",
          "sampled_texture_format", "sampled_texture_mip_count"};
      for (const char *field : texture_fields) {
        if (fields.find(field) != fields.end()) {
          *error = "draw_pco_triangles sampled texture fields require "
                   "sampled_texture_count";
          return false;
        }
      }
    }
    if (!ParseOptionalU32(fields, "draw_count", &parsed.draw_count, error) ||
        !ParseOptionalU32(fields, "ia_vertices", &parsed.ia_vertices,
                          error) ||
        !ParseOptionalU32(fields, "ia_primitives", &parsed.ia_primitives,
                          error) ||
        !ParseOptionalU32(fields, "vs_invocations", &parsed.vs_invocations,
                          error) ||
        !ParseOptionalU32(fields, "gs_invocations", &parsed.gs_invocations,
                          error) ||
        !ParseOptionalU32(fields, "gs_primitives", &parsed.gs_primitives,
                          error) ||
        !ParseOptionalU32(fields, "clip_invocations",
                          &parsed.clip_invocations, error) ||
        !ParseOptionalU32(fields, "clip_primitives", &parsed.clip_primitives,
                          error) ||
        !ParseOptionalU32(fields, "hs_invocations", &parsed.hs_invocations,
                          error) ||
        !ParseOptionalU32(fields, "ds_invocations", &parsed.ds_invocations,
                          error) ||
        !ParseOptionalU32(fields, "cs_invocations", &parsed.cs_invocations,
                          error) ||
        !ParseOptionalU32(fields, "setup_triangles", &parsed.setup_triangles,
                          error)) {
      return false;
    }
    const auto parse_optional_u64 =
        [&](const char *field, std::uint64_t *value) {
          const auto entry = fields.find(field);
          if (entry == fields.end())
            return true;
          if (ParseU64(entry->second, value))
            return true;
          *error = std::string("driver command ") + field +
                   " must be a uint64 value";
          return false;
        };
    if (!parse_optional_u64("ps_invocations", &parsed.ps_invocations) ||
        !parse_optional_u64("semantic_texel_fetches",
                            &parsed.semantic_texel_fetches)) {
      return false;
    }
  }
  *command = parsed;
  return true;
}

} // namespace pvrgpu::stub
