#include "driver_command.h"

#include <algorithm>
#include <array>
#include <cstdint>
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
constexpr const char *kDrawPrimitiveSequenceCommand = "draw_primitive_sequence";
constexpr const char *kRgba8Format = "PIPE_FORMAT_R8G8B8A8_UNORM";
constexpr const char *kRgbx8Format = "PIPE_FORMAT_R8G8B8X8_UNORM";
constexpr const char *kBgrx8Format = "PIPE_FORMAT_B8G8R8X8_UNORM";
constexpr const char *kR10G10B10A2Format = "PIPE_FORMAT_R10G10B10A2_UNORM";
constexpr const char *kB10G10R10A2Format = "PIPE_FORMAT_B10G10R10A2_UNORM";

const std::set<std::string> &KnownFields() {
  static const std::set<std::string> fields = {
      "schema", "producer", "command", "case", "frame",
      "framebuffer_width", "framebuffer_height", "width",
      "height", "format", "clear_color_bits", "fragment_color_bits",
      "vertex0_bits", "vertex1_bits", "vertex2_bits",
      "draw_count", "index_count", "unique_vertices", "primitive_count",
      "clip_primitives", "setup_triangles", "semantic_texel_fetches",
      "ia_vertices", "ia_primitives", "vs_invocations", "clip_invocations",
      "ps_invocations",
  };
  return fields;
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

const std::set<std::string> &DrawPrimitiveSequenceFields() {
  static const std::set<std::string> fields = {
      "schema",           "producer",        "command",
      "case",             "frame",           "width",
      "height",           "format",          "clear_color_bits",
      "draw_count",       "ia_vertices",     "ia_primitives",
      "vs_invocations",   "clip_invocations", "clip_primitives",
      "setup_triangles",  "ps_invocations",
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
         format == kBgrx8Format || format == kR10G10B10A2Format ||
         format == kB10G10R10A2Format;
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
    if (!required.count(entry.first)) {
      *error = "field is not valid for " + command +
               " driver command: " + entry.first;
      return false;
    }
  }
  return true;
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
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == line.size()) {
      *error = "invalid key=value at driver command line " +
               std::to_string(line_number);
      return false;
    }
    const std::string key = line.substr(0, separator);
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
      parsed.command != kDrawPrimitiveSequenceCommand) {
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
  if (parsed.command == kDrawIndexedQuadCommand) {
    if (!ParseU32(fields["framebuffer_width"], &parsed.framebuffer_width) ||
        parsed.framebuffer_width == 0) {
      *error = "draw_indexed_quad framebuffer_width must be positive";
      return false;
    }
    if (!ParseU32(fields["framebuffer_height"], &parsed.framebuffer_height) ||
        parsed.framebuffer_height == 0) {
      *error = "draw_indexed_quad framebuffer_height must be positive";
      return false;
    }
    if (parsed.width > parsed.framebuffer_width ||
        parsed.height > parsed.framebuffer_height) {
      *error =
          "draw_indexed_quad viewport width/height must fit framebuffer";
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
                      : DrawPrimitiveSequenceFields();
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
  if (parsed.command == kDrawPrimitiveSequenceCommand) {
    if (!ParseU32(fields["draw_count"], &parsed.draw_count) ||
        parsed.draw_count == 0) {
      *error = "draw primitive sequence draw_count must be positive";
      return false;
    }
    if (!ParseU32(fields["ia_vertices"], &parsed.ia_vertices) ||
        parsed.ia_vertices == 0 ||
        !ParseU32(fields["ia_primitives"], &parsed.ia_primitives) ||
        parsed.ia_primitives == 0 ||
        !ParseU32(fields["vs_invocations"], &parsed.vs_invocations) ||
        parsed.vs_invocations == 0 ||
        !ParseU32(fields["clip_invocations"], &parsed.clip_invocations) ||
        parsed.clip_invocations > parsed.ia_primitives ||
        !ParseU32(fields["clip_primitives"], &parsed.clip_primitives) ||
        parsed.clip_primitives < parsed.clip_invocations ||
        !ParseU32(fields["setup_triangles"], &parsed.setup_triangles) ||
        parsed.setup_triangles > parsed.clip_primitives ||
        !ParseU64(fields["ps_invocations"], &parsed.ps_invocations)) {
      *error =
          "draw primitive sequence contains invalid semantic counter metadata";
      return false;
    }
  }

  *command = parsed;
  return true;
}

} // namespace pvrgpu::stub
