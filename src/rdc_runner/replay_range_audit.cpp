#include "rdc_runner/replay_range_audit.h"

#include <charconv>
#include <map>
#include <sstream>
#include <string_view>

namespace pvrgpu::rdc {
namespace {

bool ControlField(const std::string &key) {
  for (const std::string_view control : {
           "schema", "producer", "event", "status", "result", "error", "failed",
           "rejected", "recording_failure", "success", "supported", "reason",
           "command", "complete", "has_fence", "vertices", "draws", "grid"})
    if (key == control) return true;
  return false;
}

bool NativeApiEvent(const std::string &event) {
  return event == "systemc_api_submit" || event == "systemc_api_done" ||
         event == "compute_api_submit" || event == "compute_api_done";
}

struct Observations {
  bool native_api = false;
  bool accepted_graphics = false;
  bool nonzero_compute = false;
};

bool Unsigned(const std::string &text, std::uint64_t *value) {
  if (text.empty()) return false;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), *value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool NonzeroGrid(const std::string &text, char delimiter, bool *nonzero) {
  std::istringstream parts(text);
  std::string part;
  *nonzero = true;
  for (unsigned axis = 0; axis < 3; ++axis) {
    std::uint64_t value = 0;
    if (!std::getline(parts, part, delimiter) || !Unsigned(part, &value)) return false;
    *nonzero = *nonzero && value != 0;
  }
  return parts.eof();
}

std::string AuditDriver(const std::string &text, bool raw_transfers_only,
                        Observations *observed) {
  *observed = {};
  if (!text.empty() && text.back() != '\n') return "truncated driver event";
  std::istringstream input(text);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::map<std::string, std::string> fields;
    std::istringstream tokens(line);
    std::string token;
    while (tokens >> token) {
      const auto at = token.find('=');
      if (at == std::string::npos) continue;
      const auto key = token.substr(0, at);
      const bool inserted = fields.emplace(key, token.substr(at + 1)).second;
      if (!inserted && ControlField(key))
        return "duplicate driver control field: " + line;
    }
    if (fields["schema"] != "pvrgpu.driver-counter.v1" ||
        fields["producer"] != "pvrgpu-gallium-driver" || fields["event"].empty())
      return "invalid driver event provenance: " + line;
    const auto &event = fields["event"];
    // RenderDoc may try a current-FBO readback before mapping a different
    // resource. Only this precisely identified decline is benign; an upload
    // decline, for example, means requested work was actually lost.
    const bool benign_decline = event == "framebuffer_readback_declined" &&
                                fields["reason"] == "not_a_current_color_attachment";
    if (event.find("error") != std::string::npos ||
        event.find("fail") != std::string::npos ||
        event.find("unsupported") != std::string::npos ||
        event.find("not_supported") != std::string::npos ||
        event.find("reject") != std::string::npos ||
        event.find("cpu_present") != std::string::npos ||
        event.find("present_fallback") != std::string::npos ||
        event == "present_textured_quad" || event == "draw_present_textured_quad" ||
        event == "systemc_api_disabled" ||
        (event.find("declined") != std::string::npos && !benign_decline))
      return "driver failure: " + line;
    for (const auto &[key, value] : fields) {
      if ((key == "status" || key == "result") && value != "0" && value != "ok" &&
          value != "success" && value != "completed")
        return "driver status failure: " + line;
      if ((key == "error" || key == "failed" || key == "rejected" ||
           key == "recording_failure") && value != "0" && value != "false")
        return "driver failure flag: " + line;
      if ((key == "success" || key == "supported") && value != "1" && value != "true")
        return "driver success flag rejected: " + line;
    }
    if (event == "fence_finish" && fields["complete"] != "1")
      return "failed fence: " + line;
    if (raw_transfers_only &&
        (event.rfind("draw", 0) == 0 || event.rfind("systemc_api_", 0) == 0 ||
         event.rfind("compute_api_", 0) == 0 || event == "launch_grid"))
      return "raw-transfer scope requires shader/native execution: " + line;
    observed->native_api = observed->native_api || NativeApiEvent(event);
    // Only explicit accepted-work diagnostics create an obligation. Merely
    // seeing a draw_vbo/count is insufficient: incomplete primitives and
    // restart-only draws legitimately have no native shader work.
    if (event == "draw_array_primitive_recorded" ||
        event == "draw_array_primitive_sequence_command") {
      std::uint64_t count = 0;
      const char *field = event == "draw_array_primitive_recorded" ? "vertices" : "draws";
      if (!Unsigned(fields[field], &count)) return "invalid accepted graphics extent: " + line;
      observed->accepted_graphics = observed->accepted_graphics || count != 0;
    }
    if (event == "draw_pco_triangles_command" || event == "draw_pco_lit_mesh_command" ||
        event == "draw_pco_texture_command" || event == "draw_pco_ideas_command")
      observed->accepted_graphics = true;
    // This is only a nonempty-dispatch predicate, never an invocation estimate.
    // The actual done record remains the sole source of compute work counters.
    if (event == "compute_indirect_decoded" ||
        (event == "launch_grid" && fields.count("grid"))) {
      bool nonzero = false;
      if (!NonzeroGrid(fields["grid"], event == "compute_indirect_decoded" ? ',' : 'x', &nonzero))
        return "invalid decoded compute grid: " + line;
      observed->nonzero_compute = observed->nonzero_compute || nonzero;
    }
  }
  return {};
}

}  // namespace

std::string AuditNativeDriverEvents(const std::string &driver_events,
                                    bool raw_transfers_only) {
  Observations observed;
  return AuditDriver(driver_events, raw_transfers_only, &observed);
}

ReplayRangeAudit AuditNativeReplayRange(const std::string &model_jsonl,
                                       const std::string &driver_events) {
  ReplayRangeAudit result;
  Observations observed;
  result.error = AuditDriver(driver_events, false, &observed);
  if (!result.error.empty()) return result;
  if (!observed.native_api && model_jsonl.empty()) {
    if (observed.accepted_graphics || observed.nonzero_compute) {
      result.error = "accepted graphics or nonzero compute has no native completion";
      return result;
    }
    result.ok = true;
    return result;
  }
  if (!model_jsonl.empty() && model_jsonl.back() != '\n') {
    result.error = "truncated range-local model report";
    return result;
  }
  result.ok = ParseNativeReport(model_jsonl, driver_events, &result.report, &result.error);
  if (result.ok && ((observed.accepted_graphics && !result.report.graphics_reports) ||
                    (observed.nonzero_compute && !result.report.compute_dispatches))) {
    result.ok = false;
    result.report = {};
    result.error = "accepted work has no completion of its graphics/compute class";
  }
  result.native_work = result.ok &&
                       (result.report.graphics_reports || result.report.compute_dispatches);
  return result;
}

}  // namespace pvrgpu::rdc
