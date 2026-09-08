#include "rdc_runner/native_report.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace pvrgpu::rdc;
namespace {
unsigned checks = 0;
void Check(bool condition, const char *message) {
  ++checks;
  if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
std::string Replace(std::string value, const std::string &from, const std::string &to) {
  const auto position = value.find(from);
  Check(position != std::string::npos, "fixture replacement exists");
  value.replace(position, from.size(), to);
  return value;
}
std::string Model(std::uint64_t value) {
  NativeReport report;
  for (auto field : kNativeCounterFields) report.counters[std::string(field)] = value;
  report.counters["cs_invocations"] = 0;
  return FormatNativeReport(report, std::string(64, 'a'), "");
}
std::string Event(const std::string &event, const std::string &details = "") {
  return "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=" + event + " " + details + '\n';
}
std::string Graphics() { return Event("systemc_api_submit") + Event("systemc_api_done"); }
std::string ComputeDone() {
  return Event("compute_api_done", "workgroups=2 invocations=37 alu_instructions=111 memory_instructions=72 "
      "atomic_instructions=1 load_instructions=35 store_instructions=36 dram_read_bytes=100 "
      "dram_write_bytes=110 direct_read_bytes=0 direct_write_bytes=0 readback_bytes=144 "
      "pool_allocations=7 pool_releases=7 texture_requests=6 texel_fetches=17");
}
std::string Receipt() {
  return R"({"schema":"pvrgpu.rdc-final-output.v2","backend":"pvrgpu","status":"PASS",
    "api_error_capture":"synchronous-gl-debug-callback","debug_callback_verified":true,
    "initial_native_isolated":true,"replay_completed":true,"replay_context_finished":true,
    "initial_contents_restored":true,"initial_copy_driver_counter_path":"/output/initial-copy-driver-counter.txt",
    "api_errors":0,"rdc_path":"/captures/actual.rdc","replay_begin_event":1,"replay_end_event":91,
    "trace_draw_actions":0,"color_output":true,"source":"completed-replay-attachment",
    "png_path":"/output/final.png","resource_id":"1234","mip":3,"layer":2,"sample":0,
    "width":23,"height":17,"format":"B5G6R5_UNORM"})";
}
}

int main(int argc, char **argv) {
  std::string error;
  NativeReport report;
  if (argc == 3 && std::string(argv[1]) == "--initial-copy-audit") {
    std::ifstream input(argv[2]);
    Check(input.good(), "initial-copy audit file exists");
    const std::string audit{std::istreambuf_iterator<char>(input), {}};
    Check(!input.bad() && ValidateInitialCopyAudit(audit, &error), error.c_str());
    std::cout << "Initial-copy driver audit PASS\n";
    return 0;
  }
  if (argc == 3) {
    std::ifstream model(argv[1]), events(argv[2]);
    Check(model.good() && events.good(), "native evidence files exist");
    const std::string model_text{std::istreambuf_iterator<char>(model), {}};
    const std::string event_text{std::istreambuf_iterator<char>(events), {}};
    Check(ParseNativeReport(model_text, event_text, &report, &error), error.c_str());
    std::cout << FormatNativeReport(report, std::string(64, 'a'), "");
    return 0;
  }
  Check(argc == 1, "usage: native-report-test [model.jsonl driver-counter.txt]");
  Check(ValidateInitialCopyAudit(Event("buffer_copy_region") + Event("resource_copy_region") +
        Event("compute_state_bind") + Event("flush"), &error), error.c_str());
  Check(ValidateInitialCopyAudit(Event("flush"), &error), "empty-resource restoration still finishes");
  for (unsigned present = 0; present <= 1; ++present) {
    const auto reference = Event("fence_reference", "has_ptr=" + std::to_string(present) + " has_fence=0");
    const auto finish = Event("fence_finish", "has_context=" + std::to_string(present) +
                              " has_fence=0 timeout=18446744073709551615 complete=1");
    Check(ValidateInitialCopyAudit(reference + Event("buffer_copy_region") + Event("flush") +
          finish + reference, &error), "NULL fence lifecycle plus independent real flush is valid");
    Check(!ValidateInitialCopyAudit(reference + finish, &error), "NULL fences alone are not flush evidence");
  }
  for (const auto &audit : {std::string{}, Event("resource_copy_region"),
       Event("flush_error") + Event("flush"), Event("texture_subdata_declined") + Event("flush"),
       Event("unsupported_draw") + Event("flush"), Event("draw_vbo") + Event("flush"),
       Event("draw_pco_triangles") + Event("flush"), Event("systemc_api_disabled") + Event("flush"),
       Event("compute_api_submit") + Event("flush"), Event("launch_grid") + Event("flush"),
       Event("rejected") + Event("flush"), Event("not_supported") + Event("flush"),
       Event("new_unknown_copy") + Event("flush"), Event("resource_copy_region", "status=rejected") + Event("flush"),
       Event("resource_copy_region", "recording_failure=1") + Event("flush"),
       Event("resource_copy_region", "success=0") + Event("flush"),
       Event("resource_copy_region", "supported=false") + Event("flush"),
       Event("resource_copy_region", "status=ok status=rejected") + Event("flush"),
       Event("fence_reference", "has_ptr=1 has_fence=1") + Event("flush"),
       Event("fence_reference", "has_ptr=2 has_fence=0") + Event("flush"),
       Event("fence_reference", "has_ptr=1 has_fence=false") + Event("flush"),
       Event("fence_reference", "has_ptr=1") + Event("flush"),
       Event("fence_reference", "has_ptr=1 has_fence=0 has_fence=1") + Event("flush"),
       Event("fence_finish", "has_context=1 has_fence=1 timeout=0 complete=0") + Event("flush"),
       Event("fence_finish", "has_context=1 has_fence=1 timeout=0 complete=1") + Event("flush"),
       Event("fence_finish", "has_context=1 has_fence=0 timeout=0 complete=0") + Event("flush"),
       Event("fence_finish", "has_context=1 has_fence=0 timeout=-1 complete=1") + Event("flush"),
       Event("fence_finish", "has_context=1 has_fence=0 timeout=18446744073709551616 complete=1") + Event("flush"),
       Event("fence_finish", "has_context=1 has_fence=0 complete=1") + Event("flush"),
       Event("fence_finish", "has_context=false has_fence=0 timeout=0 complete=1") + Event("flush"),
       Event("fence_finish", "has_context=1 has_fence=0 timeout=0") + Event("flush"),
       std::string("schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=flush"),
       Replace(Event("flush"), "pvrgpu-gallium-driver", "foreign"),
       Event("flush", "event=resource_copy_region"), std::string("broken event\n")})
    Check(!ValidateInitialCopyAudit(audit, &error), "reject missing/failed/shader-dependent initial restoration");
  const auto two = Model(2) + "@CAPTURE: intermediate png=one.png\n" + Model(5);
  Check(ParseNativeReport(two, Graphics() + Graphics(), &report, &error), error.c_str());
  Check(report.graphics_reports == 2 && report.graphics_submissions == 2, "two complete native flushes");
  for (auto field : kNativeCounterFields)
    Check(report.counters.at(std::string(field)) == (field == "cs_invocations" ? 0 : 7), "add, not replace, each counter");
  const auto compute = Event("compute_api_submit", "grid=999x999x999") + ComputeDone();
  Check(ParseNativeReport(two, Graphics() + compute + Graphics(), &report, &error), error.c_str());
  Check(report.counters["cs_invocations"] == 37 && report.counters["texel_fetches"] == 24 &&
        report.compute_dispatches == 1, "actual compute stats, never dispatch-grid estimates");
  Check(ParseNativeReport("", compute, &report, &error), error.c_str());
  Check(report.graphics_reports == 0 && report.counters["drawlists"] == 0 &&
        report.counters["cs_invocations"] == 37, "compute-only native completion");
  Check(ParseNativeReport(Model(9007199254740993ULL), Graphics(), &report, &error), error.c_str());
  Check(report.counters["ia_vertices"] == 9007199254740993ULL, "integer precision exceeds double");
  Check(ParseNativeReport(Model(1), Graphics() + Graphics(), &report, &error), "coalesced native submits are not trace action counts");
  Check(!ParseNativeReport(two, Graphics(), &report, &error), "one accepted API submit cannot produce duplicate reports");

  const std::vector<std::string> bad_models = {
    Model(1).substr(0, Model(1).rfind('{')),
    Replace(Model(1), "\"hello\"", "\"done\""),
    Replace(Model(1), "\"counter\"", "\"hello\""),
    Replace(Model(1), "\"counter\"", "\"error\""),
    Replace(Model(1), "\"counter\"", "\"unexpected\""),
    Replace(Model(1), "\"pool_leaks\":0", "\"pool_leaks\":1"),
    Replace(Model(1), "\"pool_leaks\":0", "\"pool_leaks\":0,\"pool_bytes_in_flight\":4"),
    Replace(Model(1), "\"frames\":1", "\"frames\":2"),
    Replace(Model(1), "\"backend\":\"pvrgpu\"", "\"backend\":\"llvmpipe\""),
    Replace(Model(1), "\"provenance\":\"modeled\"", "\"provenance\":\"golden\""),
    Replace(Model(1), "\"version\":1", "\"version\":2"),
    Replace(Model(1), "\"ia_vertices\":1", "\"ia_vertices\":true"),
    Replace(Model(1), "\"ia_vertices\":1", "\"ia_vertices\":-1"),
    Replace(Model(1), "\"ia_vertices\":1", "\"ia_vertices\":1.0"),
    Replace(Model(1), "\"ia_vertices\":1", "\"ia_vertices\":1e0"),
    Replace(Model(1), "\"ia_vertices\":1", "\"ia_vertices\":01"),
    Replace(Model(1), "\"ia_vertices\":1", "\"ia_vertices\":18446744073709551616"),
    Replace(Model(1), "\"ia_vertices\":1", "\"ia_vertices\":1,\"ia_vertices\":2"),
    Replace(Model(1), "\"ia_vertices\":1,", ""),
    Replace(Model(1), "\"hello\"", "\"hel\\qlo\""),
    Model(UINT64_MAX) + Model(1),
    Model(1) + "{malformed}\n",
    "{\"type\":\"error\"}\n" + Model(1),
    Model(1) + "{\"type\":\"error\"}\n",
  };
  for (const auto &input : bad_models)
    Check(!ParseNativeReport(input, Graphics(), &report, &error), "reject malformed/incomplete/non-native report");
  const std::vector<std::string> bad_events = {
    "", Event("systemc_api_submit"), Event("systemc_api_done"),
    Graphics() + Event("compute_api_submit"), Graphics() + ComputeDone(),
    Graphics() + Event("compute_api_submit") + Event("compute_api_submit") + ComputeDone(),
    Graphics() + Event("compute_api_submit") + ComputeDone() + ComputeDone(),
    Graphics() + Replace(compute, "invocations=37", "invocations=nan"),
    Graphics() + Replace(compute, "invocations=37", "invocations=37 invocations=38"),
    Graphics() + Event("compute_api_submit") + "invocations=1 invocations=2 " + ComputeDone(),
    Graphics() + Replace(compute, "texel_fetches=17", ""),
    Graphics() + Replace(compute, "pool_releases=7", "pool_releases=6"),
    Graphics() + Replace(compute, "invocations=37", "invocations=18446744073709551616"),
    Graphics() + Event("clear_error"), Graphics() + Event("flush_error"),
    Graphics() + Event("texture_subdata_declined"),
    Graphics() + Event("framebuffer_boundary_flush_error"),
    Graphics() + Event("compute_launch_unsupported"), Graphics() + Event("unsupported_draw"),
    Graphics() + Event("systemc_api_disabled"), Graphics() + Event("dump_nir_failed"),
    Replace(Graphics(), "pvrgpu-gallium-driver", "foreign"),
  };
  for (const auto &events : bad_events)
    Check(!ParseNativeReport(Model(1), events, &report, &error), "reject failed/unpaired native API events");
  Check(ParseNativeReport(Model(1), Graphics() + Event("framebuffer_readback_declined",
          "reason=not_a_current_color_attachment"), &report, &error),
        "unrelated-resource map decline does not imply a lost upload");
  Check(!ParseNativeReport(Replace(Model(1), "\"cs_invocations\":0", "\"cs_invocations\":37"),
                          Graphics() + compute, &report, &error), "reject ambiguous CS double accounting");
  Check(!ParseNativeReport(Model(UINT64_MAX), Graphics() + compute, &report, &error), "compute texel sum overflow");

  FinalOutputReceipt receipt;
  Check(ParseFinalOutputReceipt(Receipt(), &receipt, &error), error.c_str());
  Check(receipt.mip == 3 && receipt.layer == 2 && receipt.width == 23 &&
        receipt.trace_draw_actions == 0, "actual nonzero subresource; zero actions are metadata");
  for (const auto &key : {"initial_native_isolated", "initial_contents_restored", "replay_completed", "replay_context_finished"})
    Check(!ParseFinalOutputReceipt(Replace(Receipt(), std::string("\"") + key + "\":true",
                std::string("\"") + key + "\":false"), &receipt, &error), "require all completion evidence");
  for (const auto &input : {
       Replace(Receipt(), "\"api_errors\":0", "\"api_errors\":1282"),
       Replace(Receipt(), "\"source\":\"completed-replay-attachment\"", "\"source\":\"model-intermediate\""),
       Replace(Receipt(), "\"replay_end_event\":91", "\"replay_end_event\":0"),
       Replace(Receipt(), "\"replay_begin_event\":1", "\"replay_begin_event\":2"),
       Replace(Receipt(), "\"width\":23", "\"width\":0"),
       Replace(Receipt(), "\"status\":\"PASS\"", "\"status\":\"FAIL\""),
       Replace(Receipt(), "pvrgpu.rdc-final-output.v2", "pvrgpu.rdc-final-output.v1"),
       Replace(Receipt(), "\"debug_callback_verified\":true", "\"debug_callback_verified\":false"),
       Replace(Receipt(), "synchronous-gl-debug-callback", "end-only-glGetError"),
       Replace(Receipt(), "/output/initial-copy-driver-counter.txt", ""),
       Replace(Receipt(), "\"mip\":3", "\"mip\":false"),
       Receipt() + "trailing"})
    Check(!ParseFinalOutputReceipt(input, &receipt, &error), "reject incorrect final-output evidence");
  Check(ParseFinalOutputReceipt(Replace(Receipt(), "\"color_output\":true", "\"color_output\":false"),
                                &receipt, &error) && !receipt.color_output, "explicit no-color completion");
  std::cout << "rdc native report: " << checks << " checks PASS\n";
}
