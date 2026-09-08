#include "rdc_runner/native_report.h"

#include <cctype>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pvrgpu::rdc {
namespace {

// Deliberately parse JSON structure: searching for a counter name can match
// an opcode, a nested DrawList, or text inside an error instead of counters.
struct Json {
  enum Kind { Object, Array, String, Number, Boolean, Null } kind = Null;
  std::map<std::string, Json> members;
  std::vector<Json> elements;
  std::string text;
};

class JsonParser {
 public:
  explicit JsonParser(const std::string &input) : input_(input) {}
  Json Parse() {
    Json value = Value(0);
    Space();
    if (position_ != input_.size()) Fail("trailing JSON data");
    return value;
  }

 private:
  [[noreturn]] void Fail(const char *message) const {
    throw std::runtime_error(std::string(message) + " at byte " +
                             std::to_string(position_));
  }
  void Space() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\t' ||
            input_[position_] == '\r' || input_[position_] == '\n')) ++position_;
  }
  bool Take(char character) {
    Space();
    if (position_ == input_.size() || input_[position_] != character) return false;
    ++position_;
    return true;
  }
  unsigned Hex4() {
    unsigned result = 0;
    for (unsigned index = 0; index < 4; ++index) {
      if (position_ == input_.size()) Fail("truncated unicode escape");
      const char c = input_[position_++];
      const unsigned digit = c >= '0' && c <= '9' ? c - '0' :
          c >= 'a' && c <= 'f' ? c - 'a' + 10 :
          c >= 'A' && c <= 'F' ? c - 'A' + 10 : 16;
      if (digit == 16) Fail("invalid unicode escape");
      result = result * 16 + digit;
    }
    return result;
  }
  std::string StringValue() {
    if (!Take('"')) Fail("expected JSON string");
    std::string result;
    while (position_ < input_.size()) {
      const unsigned char c = input_[position_++];
      if (c == '"') return result;
      if (c < 32) Fail("control byte in JSON string");
      if (c != '\\') { result += static_cast<char>(c); continue; }
      if (position_ == input_.size()) Fail("truncated JSON escape");
      switch (input_[position_++]) {
      case '"': result += '"'; break;
      case '\\': result += '\\'; break;
      case '/': result += '/'; break;
      case 'b': result += '\b'; break;
      case 'f': result += '\f'; break;
      case 'n': result += '\n'; break;
      case 'r': result += '\r'; break;
      case 't': result += '\t'; break;
      case 'u': {
        unsigned code = Hex4();
        if (code >= 0xd800 && code <= 0xdbff) {
          if (input_.compare(position_, 2, "\\u") != 0) Fail("missing low surrogate");
          position_ += 2;
          const unsigned low = Hex4();
          if (low < 0xdc00 || low > 0xdfff) Fail("invalid low surrogate");
          code = 0x10000 + (code - 0xd800) * 1024 + low - 0xdc00;
        } else if (code >= 0xdc00 && code <= 0xdfff) Fail("unpaired low surrogate");
        if (code <= 0x7f) result += static_cast<char>(code);
        else {
          if (code > 0xffff) result += static_cast<char>(0xf0 | (code >> 18));
          if (code > 0x7ff) result += static_cast<char>((code > 0xffff ? 0x80 : 0xe0) | ((code >> 12) & 63));
          result += static_cast<char>((code > 0x7ff ? 0x80 : 0xc0) | ((code >> 6) & 63));
          result += static_cast<char>(0x80 | (code & 63));
        }
        break;
      }
      default: Fail("invalid JSON escape");
      }
    }
    Fail("unterminated JSON string");
  }
  Json Value(unsigned depth) {
    if (depth > 64) Fail("JSON nesting limit exceeded");
    Space();
    if (position_ == input_.size()) Fail("missing JSON value");
    Json value;
    if (Take('{')) {
      value.kind = Json::Object;
      if (Take('}')) return value;
      do {
        std::string key = StringValue();
        if (!Take(':')) Fail("missing JSON colon");
        if (!value.members.emplace(std::move(key), Value(depth + 1)).second)
          Fail("duplicate JSON member");
        if (Take('}')) return value;
      } while (Take(','));
      Fail("unterminated JSON object");
    }
    if (Take('[')) {
      value.kind = Json::Array;
      if (Take(']')) return value;
      do {
        value.elements.push_back(Value(depth + 1));
        if (Take(']')) return value;
      } while (Take(','));
      Fail("unterminated JSON array");
    }
    if (input_[position_] == '"') {
      value.kind = Json::String; value.text = StringValue(); return value;
    }
    for (const char *literal : {"true", "false", "null"}) {
      const std::string token(literal);
      if (input_.compare(position_, token.size(), token) == 0) {
        position_ += token.size(); value.text = token;
        value.kind = token == "null" ? Json::Null : Json::Boolean;
        return value;
      }
    }
    const std::size_t start = position_;
    if (input_[position_] == '-') ++position_;
    auto digits = [&] {
      const auto before = position_;
      while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
      if (position_ == before) Fail("missing JSON number digits");
    };
    if (position_ < input_.size() && input_[position_] == '0') ++position_;
    else digits();
    if (position_ < input_.size() && input_[position_] == '.') { ++position_; digits(); }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
      digits();
    }
    value.kind = Json::Number; value.text = input_.substr(start, position_ - start);
    return value;
  }
  const std::string &input_;
  std::size_t position_ = 0;
};

const Json &Member(const Json &object, const std::string &key) {
  if (object.kind != Json::Object) throw std::runtime_error("expected JSON object");
  const auto found = object.members.find(key);
  if (found == object.members.end()) throw std::runtime_error("missing " + key);
  return found->second;
}

std::uint64_t Unsigned(const std::string &text, const std::string &name) {
  if (text.empty()) throw std::runtime_error("empty integer " + name);
  std::uint64_t value = 0;
  for (char c : text) {
    if (c < '0' || c > '9' || value > (UINT64_MAX - (c - '0')) / 10)
      throw std::runtime_error("invalid uint64 " + name);
    value = value * 10 + c - '0';
  }
  return value;
}

std::uint64_t Uint(const Json &object, const std::string &key) {
  const Json &value = Member(object, key);
  if (value.kind != Json::Number) throw std::runtime_error("expected integer " + key);
  return Unsigned(value.text, key);
}

std::string String(const Json &object, const std::string &key) {
  const Json &value = Member(object, key);
  if (value.kind != Json::String) throw std::runtime_error("expected string " + key);
  if (value.text.find('\0') != std::string::npos)
    throw std::runtime_error("NUL in protocol string " + key);
  return value.text;
}

bool Boolean(const Json &object, const std::string &key) {
  const Json &value = Member(object, key);
  if (value.kind != Json::Boolean) throw std::runtime_error("expected boolean " + key);
  return value.text == "true";
}

void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void Add(std::uint64_t &destination, std::uint64_t amount) {
  Require(amount <= UINT64_MAX - destination, "native counter sum overflow");
  destination += amount;
}

std::string Quote(const std::string &input) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string output = "\"";
  for (unsigned char c : input) {
    if (c == '"' || c == '\\') { output += '\\'; output += c; }
    else if (c < 32) { output += "\\u00"; output += hex[c >> 4]; output += hex[c & 15]; }
    else output += c;
  }
  return output + '"';
}

template <typename Callback> void ForLines(const std::string &text, Callback callback) {
  std::size_t offset = 0, number = 0;
  while (offset < text.size()) {
    const auto end = text.find('\n', offset);
    const auto line = text.substr(offset, end == std::string::npos ? end : end - offset);
    ++number;
    try { callback(line); }
    catch (const std::exception &failure) {
      throw std::runtime_error("line " + std::to_string(number) + ": " + failure.what());
    }
    if (end == std::string::npos) break;
    offset = end + 1;
  }
}

std::map<std::string, std::string> EventFields(const std::string &line, bool strict = false) {
  std::map<std::string, std::string> fields;
  std::vector<std::string> duplicates;
  std::size_t begin = 0;
  while (begin < line.size()) {
    const auto end = line.find_first_of(" \t\r", begin);
    const std::string token = line.substr(begin, end == std::string::npos ? end : end - begin);
    const auto equals = token.find('=');
    if (equals != std::string::npos) {
      const std::string key = token.substr(0, equals);
      const bool inserted = fields.emplace(key, token.substr(equals + 1)).second;
      if (!inserted) duplicates.push_back(key);
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  // Decide strictness only after collecting the event, so reordered fields
  // cannot hide a duplicate numeric completion field before event= appears.
  const auto event = fields.find("event");
  for (const auto &key : duplicates)
    Require(!strict && key != "schema" && key != "producer" && key != "event" &&
            (event == fields.end() || event->second != "compute_api_done"),
            "duplicate driver event field");
  return fields;
}

}  // namespace

bool ValidateInitialCopyAudit(const std::string &driver_events, std::string *error) {
  if (!error) return false;
  try {
    Require(driver_events.empty() || driver_events.back() == '\n', "truncated initial-copy driver audit");
    bool flushed = false;
    ForLines(driver_events, [&](const std::string &line) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) return;
      const auto fields = EventFields(line, true);
      const auto get = [&](const char *key) -> const std::string & {
        const auto found = fields.find(key);
        Require(found != fields.end(), std::string("missing initial-copy driver event ") + key);
        return found->second;
      };
      Require(get("schema") == "pvrgpu.driver-counter.v1" &&
              get("producer") == "pvrgpu-gallium-driver", "invalid initial-copy driver provenance");
      const auto &event = get("event");
      Require(event.find("error") == std::string::npos && event.find("fail") == std::string::npos &&
              event.find("unsupported") == std::string::npos && event.find("declined") == std::string::npos,
              "initial-copy driver failure: " + line);
      // Accept only known state/CPU-transfer diagnostics, never an unknown
      // event whose failure happens to use a different spelling. These events
      // are emitted by the actual driver implementation, not capture metadata.
      static constexpr std::string_view allowed[] = {
        "flush", "flush_resource", "memory_barrier", "framebuffer_boundary",
        "fence_reference", "fence_finish",
        "resource_create", "resource_destroy", "buffer_copy_region", "resource_copy_region",
        "buffer_map", "buffer_unmap", "texture_map", "texture_unmap", "texture_unmap_hash",
        "buffer_subdata", "texture_subdata", "clear_buffer", "clear_texture", "blit",
        "create_blend_state", "create_sampler_state", "create_rasterizer_state",
        "create_depth_stencil_alpha_state", "bind_blend_state", "bind_rasterizer_state",
        "bind_depth_stencil_alpha_state", "create_shader", "bind_shader", "dump_nir",
        "create_vertex_elements", "vertex_element", "bind_vertex_elements",
        "bind_sampler_states", "create_sampler_view", "destroy_sampler_view", "set_blend_color",
        "set_stencil_ref", "set_sample_mask", "set_min_samples", "set_inlinable_constants",
        "set_tess_state", "set_patch_vertices", "set_constant_buffer", "set_sampler_views",
        "set_scissor", "set_viewport", "set_vertex_buffers", "vertex_buffer",
        "set_framebuffer_state", "set_shader_buffers", "set_shader_images", "compute_state_bind",
        "compute_state_create", "set_active_query_state", "render_condition", "set_stream_output_targets",
        "stream_output_binding", "create_stream_output_target", "stream_output_target_destroy"
      };
      Require(std::find(std::begin(allowed), std::end(allowed), event) != std::end(allowed),
              "unrecognized initial-copy driver event: " + line);
      if (event == "fence_reference" || event == "fence_finish") {
        // This synchronous driver represents a failed/unknown fence with a
        // non-NULL handle. A NULL reference is only lifecycle bookkeeping,
        // never a substitute for the independent successful flush evidence.
        const auto &present = get(event == "fence_reference" ? "has_ptr" : "has_context");
        Require((present == "0" || present == "1") && get("has_fence") == "0",
                "initial-copy invalid or failed fence: " + line);
        if (event == "fence_finish") {
          Require(get("complete") == "1", "initial-copy fence did not complete: " + line);
          (void)Unsigned(get("timeout"), "initial-copy fence timeout");
        }
      }
      for (const auto &[key, value] : fields) {
        if (key == "status" || key == "result")
          Require(value == "0" || value == "ok" || value == "success" || value == "completed",
                  "initial-copy driver status failure: " + line);
        if (key == "error" || key == "failed" || key == "rejected" || key == "recording_failure")
          Require(value == "0" || value == "false", "initial-copy driver failure flag: " + line);
        if (key == "success" || key == "supported")
          Require(value == "1" || value == "true", "initial-copy driver incomplete transfer: " + line);
      }
      // A copy path needing shaders cannot silently succeed while the native
      // APIs are isolated. State binds and raw resource transfers are allowed.
      Require(event.rfind("draw", 0) != 0 && event.rfind("systemc_api_", 0) != 0 &&
              event.rfind("compute_api_", 0) != 0 && event != "launch_grid",
              "initial-copy requires non-isolated execution: " + line);
      if (event == "flush") flushed = true;
    });
    Require(flushed, "missing initial-copy driver flush evidence");
    return true;
  } catch (const std::exception &failure) { *error = failure.what(); return false; }
}

bool ParseNativeReport(const std::string &model_jsonl, const std::string &driver_events,
                       NativeReport *report, std::string *error) {
  if (!report || !error) return false;
  try {
    NativeReport result;
    for (auto field : kNativeCounterFields) result.counters[std::string(field)] = 0;
    unsigned state = 0;
    ForLines(model_jsonl, [&](const std::string &line) {
      const auto first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos || line[first] != '{') return;
      const Json message = JsonParser(line).Parse();
      const std::string type = String(message, "type");
      Require(type != "error", "native model reported an error");
      Require(String(message, "backend") == "pvrgpu", "wrong native backend");
      Require(String(message, "schema") == "pvrgpu.counter.v1", "wrong native schema");
      Require(String(message, "protocol") == "pvrgpu-jsonl" && Uint(message, "version") == 1,
              "unsupported native protocol");
      if (type == "hello") {
        Require(state == 0, "hello before previous native report completed");
        state = 1;
      } else if (type == "counter") {
        Require(state == 1, "counter without matching hello");
        Require(String(message, "provenance") == "modeled" &&
                String(message, "source") == "pvrgpu-systemc", "counter is not native modeled evidence");
        (void)Uint(message, "frame");
        const Json &counters = Member(message, "counters");
        for (auto field : kNativeCounterFields)
          Add(result.counters[std::string(field)], Uint(counters, std::string(field)));
        state = 2;
      } else if (type == "done") {
        Require(state == 2, "done without matching counter");
        Require(Uint(message, "frames") == 1, "native flush must report one logical frame");
        Require(Uint(message, "pool_leaks") == 0, "native MemoryPool leak");
        const auto bytes = message.members.find("pool_bytes_in_flight");
        if (bytes != message.members.end()) Require(Uint(message, "pool_bytes_in_flight") == 0,
                                                    "native MemoryPool bytes remain in flight");
        ++result.graphics_reports; state = 0;
      } else throw std::runtime_error("unsupported native message type " + type);
    });
    Require(state == 0, "incomplete native hello/counter/done report");
    bool graphics_pending = false, compute_pending = false;
    std::uint64_t compute_invocations = 0, compute_texels = 0;
    ForLines(driver_events, [&](const std::string &line) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) return;
      const auto fields = EventFields(line);
      const auto get = [&](const std::string &name) -> const std::string & {
        const auto found = fields.find(name);
        Require(found != fields.end(), "missing driver event " + name);
        return found->second;
      };
      Require(get("schema") == "pvrgpu.driver-counter.v1" &&
              get("producer") == "pvrgpu-gallium-driver", "invalid driver event provenance");
      const auto &event = get("event");
      Require(event.find("error") == std::string::npos &&
              event.find("fail") == std::string::npos &&
              event.find("unsupported") == std::string::npos &&
              // A rejected multisample upload returns void to Mesa without
              // necessarily setting a GL error. Unlike benign readback-map
              // declines, it means the requested transfer did not occur.
              event != "texture_subdata_declined",
              "driver failure: " + event);
      if (event == "systemc_api_submit") {
        Require(!graphics_pending && !compute_pending, "overlapping native API submits");
        graphics_pending = true;
      } else if (event == "systemc_api_done") {
        Require(graphics_pending, "graphics completion without submit");
        graphics_pending = false; ++result.graphics_submissions;
      } else if (event == "compute_api_submit") {
        Require(!compute_pending && !graphics_pending, "overlapping native API submits");
        compute_pending = true;
      } else if (event == "compute_api_done") {
        Require(compute_pending, "compute completion without submit");
        const auto value = [&](const char *key) { return Unsigned(get(key), key); };
        // All completion fields must be real integers, even those which are
        // retained only in the raw event log instead of the 17 API counters.
        for (const char *key : {"workgroups", "invocations", "alu_instructions",
             "memory_instructions", "atomic_instructions", "load_instructions",
             "store_instructions", "dram_read_bytes", "dram_write_bytes",
             "direct_read_bytes", "direct_write_bytes", "readback_bytes",
             "pool_allocations", "pool_releases", "texture_requests", "texel_fetches"})
          (void)value(key);
        Require(value("pool_allocations") == value("pool_releases"), "compute MemoryPool leak");
        Add(compute_invocations, value("invocations"));
        Add(compute_texels, value("texel_fetches"));
        compute_pending = false; ++result.compute_dispatches;
      } else if (event == "systemc_api_disabled") {
        throw std::runtime_error("native API disabled inside measured replay");
      }
    });
    Require(!graphics_pending && !compute_pending, "native submit has no completion");
    Require((result.graphics_submissions != 0) == (result.graphics_reports != 0),
            "native graphics submits and completed model reports disagree");
    Require(result.graphics_reports <= result.graphics_submissions,
            "more graphics reports than accepted native API submissions");
    Require(result.graphics_reports || result.compute_dispatches,
            "replay has no completed native graphics or compute evidence");
    // The graphics report does not own these independently executed compute
    // dispatches. Refuse ambiguous double accounting instead of guessing.
    if (result.compute_dispatches)
      Require(result.counters["cs_invocations"] == 0, "compute invocations already present in graphics report");
    Add(result.counters["cs_invocations"], compute_invocations);
    Add(result.counters["texel_fetches"], compute_texels);
    *report = std::move(result);
    return true;
  } catch (const std::exception &failure) { *error = failure.what(); return false; }
}

bool ParseFinalOutputReceipt(const std::string &json, FinalOutputReceipt *receipt,
                             std::string *error) {
  if (!receipt || !error) return false;
  try {
    const Json message = JsonParser(json).Parse();
    Require(String(message, "schema") == "pvrgpu.rdc-final-output.v2" &&
            String(message, "backend") == "pvrgpu" && String(message, "status") == "PASS",
            "missing successful native final-output receipt");
    for (const char *key : {"initial_native_isolated", "initial_contents_restored", "replay_completed", "replay_context_finished"})
      Require(Boolean(message, key), std::string("incomplete final-output evidence: ") + key);
    Require(String(message, "api_error_capture") == "synchronous-gl-debug-callback" &&
            Boolean(message, "debug_callback_verified"), "missing continuous replay API error observation");
    Require(Uint(message, "api_errors") == 0, "replay reported GL API errors");
    FinalOutputReceipt result;
    result.initial_copy_driver_counter_path = String(message, "initial_copy_driver_counter_path");
    Require(!result.initial_copy_driver_counter_path.empty(), "missing initial-copy audit identity");
    result.rdc_path = String(message, "rdc_path");
    result.replay_begin_event = Uint(message, "replay_begin_event");
    result.replay_end_event = Uint(message, "replay_end_event");
    result.trace_draw_actions = Uint(message, "trace_draw_actions");
    Require(result.replay_begin_event == 1 && result.replay_end_event >= result.replay_begin_event,
            "final-output receipt is not a complete replay from event one");
    result.color_output = Boolean(message, "color_output");
    if (result.color_output) {
      Require(String(message, "source") == "completed-replay-attachment", "PNG is not the completed replay attachment");
      result.png_path = String(message, "png_path");
      result.resource_id = String(message, "resource_id");
      result.format = String(message, "format");
      result.mip = Uint(message, "mip"); result.layer = Uint(message, "layer");
      result.sample = Uint(message, "sample");
      result.width = Uint(message, "width"); result.height = Uint(message, "height");
      Require(!result.png_path.empty() && !result.resource_id.empty() && !result.format.empty() &&
              result.width && result.height && result.width <= UINT32_MAX && result.height <= UINT32_MAX,
              "incomplete final attachment identity");
    }
    *receipt = std::move(result);
    return true;
  } catch (const std::exception &failure) { *error = failure.what(); return false; }
}

std::string FormatNativeReport(const NativeReport &report, const std::string &rdc_sha256,
                               const std::string &artifact_png) {
  const std::string common = "{\"protocol\":\"pvrgpu-jsonl\",\"version\":1,\"schema\":\"pvrgpu.counter.v1\",\"backend\":\"pvrgpu\",";
  std::string output = common + "\"type\":\"hello\",\"rdc_sha256\":" + Quote(rdc_sha256) + "}\n";
  output += common + "\"type\":\"counter\",\"frame\":1,\"source\":\"pvrgpu-systemc\",\"provenance\":\"modeled\",\"aggregation\":\"completed-native-submissions\",\"graphics_reports\":" + std::to_string(report.graphics_reports) +
      ",\"graphics_submissions\":" + std::to_string(report.graphics_submissions) +
      ",\"compute_dispatches\":" + std::to_string(report.compute_dispatches);
  if (!artifact_png.empty()) output += ",\"artifact_png\":" + Quote(artifact_png);
  output += ",\"counters\":{";
  bool first = true;
  for (auto field : kNativeCounterFields) {
    if (!first) output += ',';
    first = false;
    output += Quote(std::string(field)) + ':' + std::to_string(report.counters.at(std::string(field)));
  }
  output += "}}\n" + common + "\"type\":\"done\",\"frames\":1,\"pool_leaks\":0}\n";
  return output;
}

}  // namespace pvrgpu::rdc
