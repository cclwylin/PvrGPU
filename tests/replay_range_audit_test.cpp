#include "rdc_runner/replay_range_audit.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace pvrgpu::rdc;

namespace {
unsigned checks = 0;

void Check(bool condition, const std::string &message) {
  ++checks;
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

std::string Replace(std::string value, const std::string &from, const std::string &to) {
  const auto at = value.find(from);
  Check(at != std::string::npos, "fixture replacement exists");
  value.replace(at, from.size(), to);
  return value;
}

std::string Model(std::uint64_t value) {
  NativeReport report;
  for (const auto field : kNativeCounterFields) report.counters[std::string(field)] = value;
  report.counters["cs_invocations"] = 0;
  return FormatNativeReport(report, std::string(64, 'a'), "");
}

std::string Event(const std::string &event, const std::string &details = "") {
  return "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=" +
         event + " " + details + '\n';
}

std::string Graphics(const std::string &command = "draw_pco_sequence") {
  return Event("systemc_api_submit", "command=" + command) +
         Event("systemc_api_done", "command=" + command);
}

std::string ComputeDone() {
  return Event("compute_api_done",
      "workgroups=2 invocations=37 alu_instructions=111 memory_instructions=72 "
      "atomic_instructions=1 load_instructions=35 store_instructions=36 dram_read_bytes=100 "
      "dram_write_bytes=110 direct_read_bytes=0 direct_write_bytes=0 readback_bytes=144 "
      "pool_allocations=7 pool_releases=7 texture_requests=6 texel_fetches=17");
}

std::string Compute() {
  return Event("compute_api_submit", "grid=999x999x999") + ComputeDone();
}

void EmptyReport(const ReplayRangeAudit &audit) {
  Check(!audit.native_work && audit.report.counters.empty() &&
        audit.report.graphics_reports == 0 && audit.report.graphics_submissions == 0 &&
        audit.report.compute_dispatches == 0, "no fabricated/partial native evidence");
}

void Reject(const std::string &model, const std::string &driver, const std::string &why) {
  const auto audit = AuditNativeReplayRange(model, driver);
  Check(!audit.ok && !audit.error.empty(), why);
  EmptyReport(audit);
}

std::string Read(const char *path) {
  std::ifstream file(path, std::ios::binary);
  Check(file.good(), std::string("read evidence ") + path);
  const std::string data{std::istreambuf_iterator<char>(file), {}};
  Check(!file.bad(), "complete evidence read");
  return data;
}
}  // namespace

int main(int argc, char **argv) {
  // Optional read-only smoke check for externally saved, exact scope bytes.
  // CTest itself has no capture, RenderDoc, filesystem fixture or GPU dependency.
  if (argc == 3) {
    const auto audit = AuditNativeReplayRange(Read(argv[1]), Read(argv[2]));
    Check(audit.ok, audit.error);
    std::cout << "ok=1 native_work=" << audit.native_work
              << " graphics_submissions=" << audit.report.graphics_submissions
              << " graphics_reports=" << audit.report.graphics_reports
              << " compute_dispatches=" << audit.report.compute_dispatches << '\n';
    return 0;
  }
  Check(argc == 1, "usage: replay-range-audit-test [model.jsonl driver-events.txt]");

  for (const auto &driver : {std::string{}, std::string(" \t\r\n"),
       Event("draw_vbo", "count=0"), Event("flush"),
       Event("set_shader_buffers", "count=3") + Event("resource_copy_region")}) {
    const auto audit = AuditNativeReplayRange("", driver);
    Check(audit.ok && audit.error.empty(), "empty/zero-action scope is legitimate");
    EmptyReport(audit);
  }

  const auto one = Model(7);
  auto audit = AuditNativeReplayRange(one, Graphics());
  Check(audit.ok && audit.native_work && audit.error.empty(), "complete native draw");
  Check(audit.report.graphics_reports == 1 && audit.report.graphics_submissions == 1 &&
        audit.report.compute_dispatches == 0, "one actual report and acceptance");
  for (const auto field : kNativeCounterFields)
    Check(audit.report.counters.at(std::string(field)) == (field == "cs_invocations" ? 0 : 7),
          "preserve every actual counter");

  // Same schema as actual MRT clear + draw flushes: neither a second draw nor
  // a second model completion may be invented for the generic clear API call.
  const auto clear_draw = Graphics("clear_color") + Event("draw_vbo", "count=1") + Graphics();
  audit = AuditNativeReplayRange(one, clear_draw);
  Check(audit.ok && audit.report.graphics_submissions == 2 && audit.report.graphics_reports == 1,
        "clear plus draw may coalesce into one report");
  Check(audit.report.counters.at("ia_vertices") == 7, "do not multiply counters by API acceptances");
  audit = AuditNativeReplayRange(Model(2) + "@CAPTURE: diagnostic.png\n" + Model(5),
                                 clear_draw + Graphics());
  Check(audit.ok && audit.report.graphics_reports == 2 && audit.report.graphics_submissions == 3 &&
        audit.report.counters.at("ia_vertices") == 7, "several real flush reports add exactly once");

  audit = AuditNativeReplayRange("", Compute());
  Check(audit.ok && audit.native_work && audit.report.graphics_reports == 0 &&
        audit.report.compute_dispatches == 1 && audit.report.counters.at("cs_invocations") == 37 &&
        audit.report.counters.at("texel_fetches") == 17, "compute-only actual completion, not grid");
  audit = AuditNativeReplayRange(one, Compute() + clear_draw + Compute() + Compute() + Compute() + Compute());
  Check(audit.ok && audit.report.compute_dispatches == 5 && audit.report.graphics_submissions == 2 &&
        audit.report.graphics_reports == 1 && audit.report.counters.at("cs_invocations") == 185 &&
        audit.report.counters.at("texel_fetches") == 92, "independent compute and graphics actual work");
  audit = AuditNativeReplayRange(Model(9007199254740993ULL), Graphics());
  Check(audit.ok && audit.report.counters.at("ia_vertices") == 9007199254740993ULL,
        "64-bit counters do not pass through floating point");

  Reject(one, "", "stale model cannot authenticate no-work scope");
  Reject(one, Event("draw_vbo", "count=0"), "action alone cannot authenticate stale model");
  Reject("", Graphics(), "API acceptance without current model completion");
  Reject(one + one, Graphics(), "excess model completions cannot be stale padding");
  Reject(one, Event("systemc_api_submit"), "unterminated graphics submission");
  Reject(one, Event("systemc_api_done"), "orphan graphics completion");
  Reject(one, Event("systemc_api_submit") + Graphics(), "overlapping graphics submit");
  Reject(one, Event("systemc_api_submit") + Compute() + Event("systemc_api_done"),
         "overlapping graphics and compute submission");
  Reject("", Event("compute_api_submit"), "incomplete compute is not zero successful work");
  Reject("", ComputeDone(), "orphan compute completion");
  Reject("", Event("compute_api_submit") + Compute(), "overlapping compute submit");
  Reject("", Replace(Compute(), "pool_releases=7", "pool_releases=6"), "compute pool leak");
  Reject("", Replace(Compute(), "invocations=37", "invocations=37 invocations=1"),
         "duplicate compute work field");
  Reject("", Replace(Compute(), "load_instructions=35 ", ""), "missing actual compute field");
  Reject("", Replace(Compute(), "texel_fetches=17", "texel_fetches=-1"), "invalid compute integer");
  Reject(Replace(one, "\"cs_invocations\":0", "\"cs_invocations\":1"), Graphics() + Compute(),
         "ambiguous double-counted compute");

  for (const auto &bad : {
       one.substr(0, one.size() - 1),
       one.substr(0, one.find('\n') + 1),
       one.substr(one.find('\n') + 1),
       Replace(one, "\"pool_leaks\":0", "\"pool_leaks\":1"),
       Replace(one, "\"frames\":1", "\"frames\":2"),
       Replace(one, "\"provenance\":\"modeled\"", "\"provenance\":\"reference\""),
       Replace(one, "\"backend\":\"pvrgpu\"", "\"backend\":\"llvmpipe\""),
       Replace(one, "\"source\":\"pvrgpu-systemc\"", "\"source\":\"copied-image\""),
       Replace(one, "\"ia_vertices\":7", "\"ia_vertices\":7,\"ia_vertices\":0"),
       Replace(one, "\"ia_vertices\":7", "\"ia_vertices\":1.5"),
       Replace(one, "\"type\":\"done\"", "\"type\":\"error\"")})
    Reject(bad, Graphics(), "malformed/incomplete/non-native model refuses");

  const std::vector<std::string> failure_events = {
    "draw_color_triangle_pco_command_error", "unsupported_draw", "compute_launch_unsupported",
    "texture_subdata_declined", "framebuffer_readback_declined", "recording_failure",
    "cpu_present", "present_fallback", "present_textured_quad", "draw_present_textured_quad",
    "resource_not_supported", "resource_rejected", "systemc_api_disabled"
  };
  const std::vector<std::string> failure_flags = {
    "status=rejected", "result=not_supported", "error=1", "failed=true", "rejected=1",
    "recording_failure=1", "success=0", "supported=false", "status=", "result=unknown"
  };
  for (const auto &event : failure_events) {
    Reject("", Event(event), "driver error refuses even when no model was produced");
    Reject(one, Graphics() + Event(event), "completed draw cannot hide later driver failure");
    Check(!AuditNativeDriverEvents(Event(event), true).empty(), "raw transfer errors refuse");
  }
  for (const auto &flag : failure_flags) {
    Reject("", Event("texture_map", flag), "failure control field on transfer");
    Reject(one, Graphics() + Event("texture_map", flag), "failure flag after native completion");
  }
  for (const auto &key : {"schema", "producer", "event", "status", "result", "error", "failed",
       "rejected", "recording_failure", "success", "supported", "reason", "command", "complete",
       "has_fence"}) {
    const auto duplicate = std::string(key) + "=ok " + key + "=error";
    Reject("", Event("texture_map", duplicate), "duplicate control fields fail closed");
    Reject(one, Event("texture_map", duplicate) + Graphics(), "duplicate before native work");
  }
  for (const auto &bad : {std::string("broken event\n"),
       Replace(Event("flush"), "pvrgpu.driver-counter.v1", "foreign.schema"),
       Replace(Event("flush"), "pvrgpu-gallium-driver", "llvmpipe"),
       Replace(Event("flush"), "event=flush", "event="),
       Event("fence_finish", "complete=0"), Event("fence_finish"),
       Event("fence_finish", "complete=1 complete=0")})
    Reject("", bad, "invalid driver provenance or failed fence");
  const auto event = Event("flush");
  Reject("", event.substr(0, event.size() - 1), "truncated no-work driver scope");
  Reject(one, Graphics().substr(0, Graphics().size() - 1), "truncated driver completion");

  const auto benign = Event("framebuffer_readback_declined", "reason=not_a_current_color_attachment");
  audit = AuditNativeReplayRange("", benign);
  Check(audit.ok && !audit.native_work, "exact benign current-FBO mapping decline");
  Check(AuditNativeDriverEvents(benign, true).empty(), "raw resource mapping may decline current FBO");
  Reject("", benign + Event("texture_subdata_declined"), "benign decline cannot mask lost upload");
  Reject("", Replace(benign, "reason=not_a_current_color_attachment",
                      "reason=not_a_current_color_attachment reason=unsupported"),
         "duplicate reason cannot disguise a lost transfer");

  const auto raw = Event("compute_state_bind") + Event("bind_shader") + Event("resource_copy_region") +
                   Event("clear_buffer") + Event("texture_map") + Event("memory_barrier") +
                   Event("flush") + Event("fence_finish", "complete=1 has_fence=0") +
                   Event("present_textured_quad_skip", "reason=not_fullscreen") +
                   Event("buffer_unmap", "status=ok result=0 error=false failed=0 rejected=false "
                                         "recording_failure=0 success=true supported=1");
  Check(AuditNativeDriverEvents(raw, true).empty(), "state and actual raw transfers remain legal");
  for (const auto &name : {"draw_vbo", "draw_pco_triangles", "systemc_api_submit", "systemc_api_done",
       "compute_api_submit", "compute_api_done", "launch_grid"}) {
    Check(!AuditNativeDriverEvents(Event(name, "count=0"), true).empty(),
          "raw-only scope cannot execute even zero-work draws or dispatches");
    Check(AuditNativeDriverEvents(Event(name), false).empty(),
          "driver-error-only observer does not pretend to audit native completion");
  }
  Check(AuditNativeDriverEvents("", true).empty(), "empty raw observer scope is allowed");

  for (const auto &accepted : {
       Event("draw_array_primitive_recorded", "vertices=3"),
       Event("draw_array_primitive_sequence_command", "draws=1"),
       Event("draw_pco_triangles_command"), Event("draw_pco_lit_mesh_command"),
       Event("draw_pco_texture_command"), Event("draw_pco_ideas_command")}) {
    Reject("", accepted, "accepted graphics cannot become no-work");
    Reject("", accepted + Compute(), "compute completion cannot cover missing graphics");
    Check(AuditNativeReplayRange(one, accepted + clear_draw).ok,
          "accepted graphics still permits real clear/draw coalescing");
  }
  for (const auto &dispatch : {Event("launch_grid", "grid=1x2x3"),
       Event("compute_indirect_decoded", "grid=1,2,3")}) {
    Reject("", dispatch, "decoded nonempty dispatch cannot become no-work");
    Reject(one, dispatch + Graphics(), "graphics completion cannot cover missing compute");
    audit = AuditNativeReplayRange("", dispatch + Compute());
    Check(audit.ok && audit.report.counters.at("cs_invocations") == 37,
          "decoded grid requires completion but never estimates its actual invocations");
  }
  for (const auto &no_work : {
       Event("draw_vbo", "count=1 first_count=2") + Event("draw_array_primitive_empty", "count=2"),
       Event("draw_array_primitive_empty", "count=17 reason=point_restart_only"),
       Event("draw_indirect_empty", "reason=zero_work"),
       Event("draw_pco_triangles_command_skip", "reason=no_output_path"),
       Event("draw_pco_triangles_draw_info_miss", "count=1"),
       Event("draw_array_primitive_recorded", "vertices=0"),
       Event("draw_array_primitive_sequence_command", "draws=0"),
       Event("compute_indirect_decoded", "grid=0,2,3"),
       Event("launch_grid", "grid=1x0x3"), Event("launch_grid", "grid=1x2x0")}) {
    const auto empty = AuditNativeReplayRange("", no_work);
    Check(empty.ok, "known no-work observations do not require fabricated reports");
    EmptyReport(empty);
  }
  for (const auto &bad : {Event("draw_array_primitive_recorded"),
       Event("draw_array_primitive_recorded", "vertices=3 vertices=0"),
       Event("draw_array_primitive_recorded", "vertices=-1"),
       Event("draw_array_primitive_sequence_command", "draws=18446744073709551616"),
       Event("launch_grid", "grid=1x2"), Event("launch_grid", "grid=1x2x3x"),
       Event("launch_grid", "grid=1x2x3x4"), Event("launch_grid", "grid=0x-1x3"),
       Event("launch_grid", "grid=1x2x3 grid=0x0x0"),
       Event("compute_indirect_decoded", "grid=1,2,3,"),
       Event("compute_indirect_decoded", "grid=1,2,18446744073709551616")})
    Reject("", bad, "malformed accepted-work extent fails closed");
  std::cout << "Replay range audit PASS: " << checks << " checks\n";
  return 0;
}
