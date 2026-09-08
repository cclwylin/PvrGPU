#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace pvrgpu::rdc {

inline constexpr std::array<std::string_view, 17> kNativeCounterFields = {
    "ia_vertices", "ia_primitives", "vs_invocations", "gs_invocations",
    "gs_primitives", "c_invocations", "c_primitives", "ps_invocations",
    "hs_invocations", "ds_invocations", "cs_invocations", "ts_invocations",
    "ms_invocations", "ms_primitives", "drawlists", "setup_triangles",
    "texel_fetches"};

struct NativeReport {
  std::map<std::string, std::uint64_t> counters;
  std::uint64_t graphics_reports = 0;
  std::uint64_t graphics_submissions = 0;
  std::uint64_t compute_dispatches = 0;
};

// The model stream contains one complete report per flush, not a cumulative
// frame counter. Driver compute completions carry the actual native CS stats.
// Never derive work from a capture action count or a dispatch grid.
bool ParseNativeReport(const std::string &model_jsonl,
                       const std::string &driver_events,
                       NativeReport *report, std::string *error);

struct FinalOutputReceipt {
  bool color_output = false;
  std::string rdc_path;
  std::string png_path;
  std::string resource_id;
  std::string format;
  std::uint64_t replay_begin_event = 0;
  std::uint64_t replay_end_event = 0;
  std::uint64_t trace_draw_actions = 0;
  std::uint64_t mip = 0;
  std::uint64_t layer = 0;
  std::uint64_t sample = 0;
  std::uint64_t width = 0;
  std::uint64_t height = 0;
};

bool ParseFinalOutputReceipt(const std::string &json,
                             FinalOutputReceipt *receipt, std::string *error);

// A single capture-level report keeps existing consumers compatible. The raw
// per-flush JSONL and driver events remain separate, unmodified evidence.
std::string FormatNativeReport(const NativeReport &report,
                               const std::string &rdc_sha256,
                               const std::string &artifact_png);

}  // namespace pvrgpu::rdc
