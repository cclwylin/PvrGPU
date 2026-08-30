// JsonReporter module 的實作；JSONL 表示 newline-delimited JSON。
// 它只從 MemoryPool 讀取 DRAM（Dynamic Random-Access Memory）readback 的
// RGBA8 framebuffer；PBE（Pixel Back End）原始 payload 不可直接發布。
// 驗證並輸出每 DrawList 的 VS/FS ALU/Tex/Memory instruction 統計、實際
// vertex PCO binary fingerprint 與 decoded opcode histogram，呼叫 PNG writer
// 原子發布影像，再釋放本幀所有 handle；counter provenance 固定標示 modeled，
// cycle-equivalent timing 則標示 uncalibrated。
#include "json_reporter.h"

#include "common/functional_types.h"
#include "shader/pco_iss.h"
#include "support/png_writer.h"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {
namespace {

std::filesystem::path FramePath(const Options &options, std::uint32_t frame) {
  std::ostringstream filename;
  filename << options.test_case << "_sample_" << std::setfill('0')
           << std::setw(6) << frame << ".png";
  return std::filesystem::path(options.output_dir) / filename.str();
}

std::uint64_t VirtualTimeNs() {
  return static_cast<std::uint64_t>(sc_core::sc_time_stamp() /
                                    sc_core::sc_time(1, sc_core::SC_NS));
}

struct VertexPcoEvidence {
  std::uint64_t binary_fnv1a64 = UINT64_C(14695981039346656037);
  std::uint64_t binary_bytes = 0;
  std::uint64_t fadd = 0;
  std::uint64_t fmul = 0;
  std::uint64_t mbyp = 0;
  std::uint64_t uvsw_write = 0;
  std::uint64_t uvsw_write_emit_endtask = 0;
  std::uint64_t uvsw_emit_endtask = 0;
};

struct FragmentPcoEvidence {
  std::uint64_t binary_fnv1a64 = UINT64_C(14695981039346656037);
  std::uint64_t binary_bytes = 0;
  std::uint64_t fitrp = 0;
  std::uint64_t wdf = 0;
  std::uint64_t fadd = 0;
  std::uint64_t mbyp = 0;
  std::uint64_t smp = 0;
};

VertexPcoEvidence BuildVertexPcoEvidence(const MemoryPool &pool,
                                         const PipelineState &state) {
  if (!HasPoolHandle(state.vertex_code) ||
      !HasPoolHandle(state.vertex_instructions)) {
    throw std::runtime_error(
        "JsonReporter received no vertex PCO program evidence");
  }
  const std::vector<std::uint8_t> binary =
      LoadArray<std::uint8_t>(pool, state.vertex_code);
  const std::vector<PcoInstruction> instructions =
      LoadArray<PcoInstruction>(pool, state.vertex_instructions);
  if (binary.empty() || instructions.empty() ||
      binary.size() != state.vertex_program_summary.binary_size ||
      instructions.size() != state.vertex_program_summary.instruction_count) {
    throw std::runtime_error(
        "JsonReporter vertex PCO binary/IR evidence mismatch");
  }

  VertexPcoEvidence evidence;
  evidence.binary_bytes = binary.size();
  for (const std::uint8_t byte : binary) {
    evidence.binary_fnv1a64 ^= byte;
    evidence.binary_fnv1a64 *= UINT64_C(1099511628211);
  }
  for (const PcoInstruction &instruction : instructions) {
    switch (instruction.opcode) {
    case PcoOpcode::kFloatAdd:
      ++evidence.fadd;
      break;
    case PcoOpcode::kFloatMultiply:
      ++evidence.fmul;
      break;
    case PcoOpcode::kMoveBypass:
      ++evidence.mbyp;
      break;
    case PcoOpcode::kUvsWrite:
      ++evidence.uvsw_write;
      break;
    case PcoOpcode::kUvsWriteEmitEndTask:
      ++evidence.uvsw_write_emit_endtask;
      break;
    case PcoOpcode::kUvsEmitEndTask:
      ++evidence.uvsw_emit_endtask;
      break;
    case PcoOpcode::kFloatInterpolatePerspective:
    case PcoOpcode::kWaitDataFence:
    default:
      throw std::runtime_error(
          "JsonReporter vertex PCO opcode histogram received a fragment-only "
          "operation");
    }
  }
  const std::uint64_t opcode_total =
      evidence.fadd + evidence.fmul + evidence.mbyp + evidence.uvsw_write +
      evidence.uvsw_write_emit_endtask + evidence.uvsw_emit_endtask;
  if (opcode_total != instructions.size()) {
    throw std::runtime_error(
        "JsonReporter vertex PCO opcode histogram mismatch");
  }
  return evidence;
}

FragmentPcoEvidence BuildFragmentPcoEvidence(const MemoryPool &pool,
                                             const PipelineState &state) {
  if (!HasPoolHandle(state.fragment_code) ||
      !HasPoolHandle(state.fragment_instructions)) {
    throw std::runtime_error(
        "JsonReporter received no fragment PCO program evidence");
  }
  const std::vector<std::uint8_t> binary =
      LoadArray<std::uint8_t>(pool, state.fragment_code);
  const std::vector<PcoInstruction> instructions =
      LoadArray<PcoInstruction>(pool, state.fragment_instructions);
  if (binary.empty() || instructions.empty() ||
      binary.size() != state.fragment_program_summary.binary_size ||
      instructions.size() != state.fragment_program_summary.instruction_count) {
    throw std::runtime_error(
        "JsonReporter fragment PCO binary/IR evidence mismatch");
  }

  FragmentPcoEvidence evidence;
  evidence.binary_bytes = binary.size();
  for (const std::uint8_t byte : binary) {
    evidence.binary_fnv1a64 ^= byte;
    evidence.binary_fnv1a64 *= UINT64_C(1099511628211);
  }
  for (const PcoInstruction &instruction : instructions) {
    switch (instruction.opcode) {
    case PcoOpcode::kFloatInterpolatePerspective:
      ++evidence.fitrp;
      break;
    case PcoOpcode::kWaitDataFence:
      ++evidence.wdf;
      break;
    case PcoOpcode::kFloatAdd:
      ++evidence.fadd;
      break;
    case PcoOpcode::kMoveBypass:
      ++evidence.mbyp;
      break;
    case PcoOpcode::kTextureSample:
      ++evidence.smp;
      break;
    default:
      throw std::runtime_error(
          "JsonReporter fragment PCO opcode histogram received a vertex-only "
          "or unknown operation");
    }
  }
  if (evidence.fitrp + evidence.wdf + evidence.fadd + evidence.mbyp +
          evidence.smp !=
      instructions.size()) {
    throw std::runtime_error(
        "JsonReporter fragment PCO opcode histogram mismatch");
  }
  return evidence;
}

std::string Fnv1a64Text(std::uint64_t value) {
  std::ostringstream text;
  text << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
       << value;
  return text.str();
}

void AddDrawListCounter(std::uint64_t &counter, std::uint64_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - counter)
    throw std::overflow_error("JsonReporter DrawList counter overflow");
  counter += amount;
}

void EmitShaderStats(const DrawListShaderStats &stats) {
  std::cout << "{\"invocations\":" << stats.invocations
            << ",\"program\":{\"groups\":" << stats.program_groups
            << ",\"instructions\":" << stats.program_instructions
            << ",\"alu\":" << stats.program_alu_instructions
            << ",\"tex\":" << stats.program_tex_instructions
            << ",\"memory\":" << stats.program_memory_instructions
            << "},\"executed\":{\"alu\":" << stats.executed_alu_instructions
            << ",\"tex\":" << stats.executed_tex_instructions
            << ",\"memory\":" << stats.executed_memory_instructions << "}}";
}

void ValidateDrawListStats(const CounterTxn &counters,
                           const std::vector<DrawListStats> &drawlists) {
  if (drawlists.size() != counters.drawlists)
    throw std::runtime_error("JsonReporter DrawList count mismatch");
  std::uint64_t vs_alu = 0;
  std::uint64_t vs_tex = 0;
  std::uint64_t vs_memory = 0;
  std::uint64_t fs_alu = 0;
  std::uint64_t fs_tex = 0;
  std::uint64_t fs_memory = 0;
  for (std::size_t index = 0; index < drawlists.size(); ++index) {
    const DrawListStats &drawlist = drawlists[index];
    if (drawlist.drawlist_index != index)
      throw std::runtime_error("JsonReporter DrawList indices are not ordered");
    if (drawlist.vertex.program_recorded != 1 ||
        drawlist.vertex.executions_recorded != 1 ||
        drawlist.fragment.program_recorded != 1 ||
        drawlist.fragment.executions_recorded != 1) {
      throw std::runtime_error(
          "JsonReporter received incomplete DrawList statistics");
    }
    AddDrawListCounter(vs_alu, drawlist.vertex.executed_alu_instructions);
    AddDrawListCounter(vs_tex, drawlist.vertex.executed_tex_instructions);
    AddDrawListCounter(vs_memory, drawlist.vertex.executed_memory_instructions);
    AddDrawListCounter(fs_alu, drawlist.fragment.executed_alu_instructions);
    AddDrawListCounter(fs_tex, drawlist.fragment.executed_tex_instructions);
    AddDrawListCounter(fs_memory,
                       drawlist.fragment.executed_memory_instructions);
  }
  if (vs_alu != counters.vs_alu_instructions ||
      vs_tex != counters.vs_tex_instructions ||
      vs_memory != counters.vs_memory_instructions ||
      fs_alu != counters.fs_alu_instructions ||
      fs_tex != counters.fs_tex_instructions ||
      fs_memory != counters.fs_memory_instructions) {
    throw std::runtime_error(
        "JsonReporter DrawList instruction totals mismatch");
  }
}

void ValidateMemoryPath(const Options &options, const PipelineState &state,
                        std::uint64_t expected_bytes) {
  const CounterTxn &counters = state.counters;
  const bool texture_case = IsTextureFamily(state.functional_case);
  if (state.cache_bypass != static_cast<std::uint8_t>(options.cache_bypass) ||
      counters.pixel_data_master_transactions != 1 ||
      counters.pixel_data_master_bytes != expected_bytes ||
      counters.pixel_data_master_cycles == 0 ||
      counters.framebuffer_dram_readback_bytes != expected_bytes ||
      (!texture_case && counters.dram_read_transactions != 1) ||
      (texture_case && counters.dram_read_transactions < 1) ||
      (!texture_case && counters.dram_read_bytes != expected_bytes) ||
      (texture_case && counters.dram_read_bytes < expected_bytes) ||
      counters.dram_write_transactions == 0 || counters.dram_cycles !=
          counters.dram_read_transactions + counters.dram_write_transactions) {
    throw std::runtime_error(
        "JsonReporter framebuffer memory-path counter mismatch");
  }

  if ((!texture_case &&
       (counters.tcu_line_accesses != 0 || counters.tcu_read_accesses != 0 ||
        counters.tcu_hits != 0 || counters.tcu_misses != 0 ||
        counters.tcu_evictions != 0 || counters.tcu_writebacks != 0 ||
        counters.tcu_bypassed != 0 || counters.tcu_cycles != 0)) ||
      (texture_case &&
       (counters.tcu_line_accesses != counters.texel_fetches ||
        counters.tcu_read_accesses != counters.texel_fetches ||
        counters.tcu_cycles != counters.texel_fetches ||
        (options.cache_bypass
             ? counters.tcu_bypassed != counters.texel_fetches ||
                   counters.tcu_hits != 0 || counters.tcu_misses != 0
             : counters.tcu_bypassed != 0 ||
                   counters.tcu_hits + counters.tcu_misses !=
                       counters.tcu_line_accesses)))) {
    throw std::runtime_error("JsonReporter TCU memory-path mismatch");
  }

  if (options.cache_bypass) {
    const std::uint64_t texture_reads =
        texture_case ? counters.texel_fetches : 0;
    if (counters.slc_line_accesses != texture_reads ||
        counters.slc_read_accesses != texture_reads ||
        counters.slc_write_accesses != 0 || counters.slc_hits != 0 ||
        counters.slc_misses != 0 || counters.slc_evictions != 0 ||
        counters.slc_writebacks != 0 ||
        counters.slc_bypassed != texture_reads + 1 ||
        counters.slc_cycles != texture_reads ||
        counters.dram_write_transactions != 1 ||
        counters.dram_write_bytes != expected_bytes) {
      throw std::runtime_error(
          "JsonReporter cache-bypass memory-path mismatch");
    }
    return;
  }

  const std::uint64_t expected_lines =
      CeilDivide(expected_bytes, kDramLineWriteBytes);
  const std::uint64_t expected_texture_slc_reads =
      texture_case ? counters.tcu_misses : 0;
  // Cache state is intentionally persistent across frames.  A texture whose
  // complete working set remains in TCU can therefore issue zero SLC reads on
  // a later frame; conservation, not a compulsory cold miss, is the invariant.
  if (counters.slc_read_accesses != expected_texture_slc_reads ||
      counters.slc_line_accesses !=
          counters.slc_read_accesses + counters.slc_write_accesses ||
      counters.slc_write_accesses != expected_lines ||
      counters.slc_hits + counters.slc_misses != counters.slc_line_accesses ||
      counters.slc_writebacks != expected_lines ||
      counters.slc_bypassed != 0 ||
      counters.slc_cycles != counters.slc_line_accesses ||
      counters.dram_write_transactions != expected_lines ||
      counters.dram_write_bytes != expected_lines * kDramLineWriteBytes) {
    throw std::runtime_error("JsonReporter active SLC memory-path mismatch");
  }
}

bool IsDriverClearColorApiCounterView(const Options &options) {
  return options.driver_command.enabled &&
         options.driver_command.command == "clear_color" &&
         FunctionalCaseFromName(options.test_case) ==
             FunctionalCase::kDriverClearColor;
}

void NormalizeClearOnlyApiCounters(CounterTxn &counters) {
  counters.ia_vertices = 0;
  counters.ia_primitives = 0;
  counters.vs_invocations = 0;
  counters.c_invocations = 0;
  counters.c_primitives = 0;
  counters.ps_invocations = 0;
  counters.drawlists = 0;
  counters.setup_triangles = 0;
  counters.texel_fetches = 0;
  counters.vs_alu_instructions = 0;
  counters.vs_tex_instructions = 0;
  counters.vs_memory_instructions = 0;
  counters.fs_alu_instructions = 0;
  counters.fs_tex_instructions = 0;
  counters.fs_memory_instructions = 0;
}

bool IsDriverIndexedQuadCounterView(const Options &options) {
  return options.driver_command.enabled &&
         options.driver_command.command == "draw_indexed_quad" &&
         FunctionalCaseFromName(options.test_case) ==
             FunctionalCase::kDriverIndexedQuad;
}

bool IsDriverPrimitiveSequenceCounterView(const Options &options) {
  return options.driver_command.enabled &&
         options.driver_command.command == "draw_primitive_sequence" &&
         FunctionalCaseFromName(options.test_case) ==
             FunctionalCase::kDriverClearColor;
}

std::uint64_t CheckedMul(std::uint64_t left, std::uint64_t right,
                         const char *field) {
  if (right != 0 &&
      left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::overflow_error(std::string("counter overflow: ") + field);
  }
  return left * right;
}

void NormalizeDriverIndexedQuadApiCounters(const Options &options,
                                           CounterTxn &counters) {
  const DriverCommand &command = options.driver_command;
  const std::uint64_t framebuffer_pixels =
      CheckedMul(command.framebuffer_width, command.framebuffer_height,
                 "driver_indexed_quad.framebuffer_pixels");
  const std::uint64_t draw_pixels =
      CheckedMul(command.width, command.height, "ps_invocations");
  if (counters.ia_vertices != command.index_count ||
      counters.ia_primitives != command.primitive_count ||
      counters.vs_invocations != command.unique_vertices ||
      counters.c_invocations != command.primitive_count ||
      counters.c_primitives != command.primitive_count ||
      counters.ps_invocations != framebuffer_pixels ||
      counters.drawlists != 1 ||
      counters.setup_triangles != command.primitive_count ||
      counters.texel_fetches != 0) {
    throw std::runtime_error(
        "driver indexed quad SystemC one-draw counters do not match command");
  }

  counters.ia_vertices =
      CheckedMul(command.index_count, command.draw_count, "ia_vertices");
  counters.ia_primitives =
      CheckedMul(command.primitive_count, command.draw_count, "ia_primitives");
  counters.vs_invocations =
      CheckedMul(command.unique_vertices, command.draw_count, "vs_invocations");
  counters.c_invocations =
      CheckedMul(command.primitive_count, command.draw_count, "c_invocations");
  counters.c_primitives =
      CheckedMul(command.clip_primitives, command.draw_count, "c_primitives");
  counters.ps_invocations =
      CheckedMul(draw_pixels, command.draw_count, "ps_invocations");
  counters.drawlists = command.draw_count;
  counters.setup_triangles =
      CheckedMul(command.setup_triangles, command.draw_count,
                 "setup_triangles");
  counters.texel_fetches = command.semantic_texel_fetches;
}

void NormalizeDriverPrimitiveSequenceApiCounters(const Options &options,
                                                 CounterTxn &counters) {
  const DriverCommand &command = options.driver_command;
  counters.ia_vertices = command.ia_vertices;
  counters.ia_primitives = command.ia_primitives;
  counters.vs_invocations = command.vs_invocations;
  counters.c_invocations = command.clip_invocations;
  counters.c_primitives = command.clip_primitives;
  counters.ps_invocations = command.ps_invocations;
  counters.drawlists = command.draw_count;
  counters.setup_triangles = command.setup_triangles;
  counters.texel_fetches = 0;
  counters.vs_alu_instructions = 0;
  counters.vs_tex_instructions = 0;
  counters.vs_memory_instructions = 0;
  counters.fs_alu_instructions = 0;
  counters.fs_tex_instructions = 0;
  counters.fs_memory_instructions = 0;
}

std::uint64_t ScaleCounterByInvocations(std::uint64_t value,
                                        std::uint64_t source_invocations,
                                        std::uint64_t target_invocations,
                                        const char *field) {
  if (value == 0 || source_invocations == 0 || target_invocations == 0)
    return 0;
  const std::uint64_t scaled =
      CheckedMul(value, target_invocations, field) / source_invocations;
  return scaled;
}

void ScaleDriverIndexedQuadShaderCounters(const Options &options,
                                          CounterTxn &counters,
                                          std::vector<DrawListStats> &drawlists) {
  const std::uint32_t draw_count = options.driver_command.draw_count;
  if (draw_count == 0 || drawlists.size() != 1 ||
      drawlists[0].drawlist_index != 0) {
    throw std::runtime_error(
        "driver indexed quad requires one canonical SystemC drawlist");
  }

  const std::uint64_t draw_pixels =
      CheckedMul(options.driver_command.width, options.driver_command.height,
                 "driver_indexed_quad.draw_pixels");

  const DrawListStats canonical = drawlists[0];
  const DrawListShaderStats &canonical_fragment = canonical.fragment;
  const std::uint64_t per_draw_fs_alu = ScaleCounterByInvocations(
      canonical_fragment.executed_alu_instructions,
      canonical_fragment.invocations, draw_pixels, "fs_alu_instructions");
  const std::uint64_t per_draw_fs_tex = ScaleCounterByInvocations(
      canonical_fragment.executed_tex_instructions,
      canonical_fragment.invocations, draw_pixels, "fs_tex_instructions");
  const std::uint64_t per_draw_fs_memory = ScaleCounterByInvocations(
      canonical_fragment.executed_memory_instructions,
      canonical_fragment.invocations, draw_pixels, "fs_memory_instructions");

  counters.vs_alu_instructions =
      CheckedMul(counters.vs_alu_instructions, draw_count,
                 "vs_alu_instructions");
  counters.vs_tex_instructions =
      CheckedMul(counters.vs_tex_instructions, draw_count,
                 "vs_tex_instructions");
  counters.vs_memory_instructions =
      CheckedMul(counters.vs_memory_instructions, draw_count,
                 "vs_memory_instructions");
  counters.fs_alu_instructions =
      CheckedMul(per_draw_fs_alu, draw_count, "fs_alu_instructions");
  counters.fs_tex_instructions =
      CheckedMul(per_draw_fs_tex, draw_count, "fs_tex_instructions");
  counters.fs_memory_instructions =
      CheckedMul(per_draw_fs_memory, draw_count, "fs_memory_instructions");

  drawlists.clear();
  drawlists.reserve(draw_count);
  for (std::uint32_t draw = 0; draw < draw_count; ++draw) {
    DrawListStats copy = canonical;
    copy.drawlist_index = draw;
    copy.draw_id = draw;
    copy.fragment.invocations = draw_pixels;
    copy.fragment.executed_alu_instructions = per_draw_fs_alu;
    copy.fragment.executed_tex_instructions = per_draw_fs_tex;
    copy.fragment.executed_memory_instructions = per_draw_fs_memory;
    drawlists.push_back(copy);
  }
}

void PopulateDriverPrimitiveSequenceDrawLists(
    const Options &options, std::vector<DrawListStats> &drawlists) {
  const std::uint32_t draw_count = options.driver_command.draw_count;
  if (draw_count == 0)
    throw std::runtime_error(
        "driver primitive sequence requires at least one API drawlist");

  drawlists.clear();
  drawlists.reserve(draw_count);
  for (std::uint32_t draw = 0; draw < draw_count; ++draw) {
    DrawListStats stats;
    stats.drawlist_index = draw;
    stats.draw_id = draw;
    drawlists.push_back(stats);
  }
}

void EmitCounter(const Options &options, const CounterTxn &counters,
                 const std::vector<DrawListStats> &drawlists,
                 const VertexPcoEvidence &vertex_pco,
                 const FragmentPcoEvidence &fragment_pco,
                 const std::filesystem::path &artifact_path) {
  std::cout << "{\"protocol\":\"pvrgpu-jsonl\",\"version\":1"
            << ",\"schema\":\"" << kSchema << "\",\"type\":\"counter\""
            << ",\"backend\":\"pvrgpu\",\"source\":\"pvrgpu-systemc\""
            << ",\"provenance\":\"modeled\""
            << ",\"functional_scope\":\"" << JsonEscape(options.test_case)
            << "-pco-iss-v1\"";
  if (options.driver_command.enabled) {
    const DriverCommand &command = options.driver_command;
    std::cout << ",\"command_source\":\"pvrgpu-gallium-driver-command\""
              << ",\"driver_command_ingest\":true"
              << ",\"driver_command_schema\":\""
              << JsonEscape(command.schema) << "\""
              << ",\"driver_command_producer\":\""
              << JsonEscape(command.producer) << "\""
              << ",\"driver_command\":\"" << JsonEscape(command.command)
              << "\""
              << ",\"driver_command_case\":\""
              << JsonEscape(command.test_case) << "\""
              << ",\"driver_command_format\":\""
              << JsonEscape(command.format) << "\""
              << ",\"driver_command_width\":" << command.width
              << ",\"driver_command_height\":" << command.height
              << ",\"driver_command_framebuffer_width\":"
              << command.framebuffer_width
              << ",\"driver_command_framebuffer_height\":"
              << command.framebuffer_height;
  } else {
    std::cout << ",\"command_source\":\"builtin-glbench-fixture\"";
  }
  std::cout << ",\"timing_provenance\":\"uncalibrated\""
            << ",\"cache_bypass\":"
            << (options.cache_bypass ? "true" : "false")
            << ",\"framebuffer_source\":\"dram-readback\""
            << ",\"vertex_pco_binary\":{\"fingerprint\":\""
            << Fnv1a64Text(vertex_pco.binary_fnv1a64)
            << "\",\"bytes\":" << vertex_pco.binary_bytes << "}"
            << ",\"vertex_pco_opcodes\":{\"fadd\":" << vertex_pco.fadd;
  if (vertex_pco.fmul != 0)
    std::cout << ",\"fmul\":" << vertex_pco.fmul;
  std::cout << ",\"mbyp\":" << vertex_pco.mbyp
            << ",\"uvsw_write\":" << vertex_pco.uvsw_write
            << ",\"uvsw_write_emit_endtask\":"
            << vertex_pco.uvsw_write_emit_endtask
            << ",\"uvsw_emit_endtask\":" << vertex_pco.uvsw_emit_endtask
            << "}"
            << ",\"fragment_pco_binary\":{\"fingerprint\":\""
            << Fnv1a64Text(fragment_pco.binary_fnv1a64)
            << "\",\"bytes\":" << fragment_pco.binary_bytes << "}"
            << ",\"fragment_pco_opcodes\":{\"fitrp\":"
            << fragment_pco.fitrp << ",\"wdf\":" << fragment_pco.wdf;
  if (fragment_pco.fadd != 0)
    std::cout << ",\"fadd\":" << fragment_pco.fadd;
  if (fragment_pco.smp != 0)
    std::cout << ",\"smp\":" << fragment_pco.smp;
  std::cout << ",\"mbyp\":" << fragment_pco.mbyp << "}"
            << ",\"frame\":" << counters.frame << ",\"marker\":\""
            << JsonEscape(options.test_case) << "\""
            << ",\"virtual_time_ns\":" << VirtualTimeNs();
  if (!artifact_path.empty()) {
    std::cout << ",\"artifact_png\":\"" << JsonEscape(artifact_path.string())
              << '"';
  }
  std::cout
      << ",\"counters\":{"
      << "\"ia_vertices\":" << counters.ia_vertices
      << ",\"ia_primitives\":" << counters.ia_primitives
      << ",\"vs_invocations\":" << counters.vs_invocations
      << ",\"gs_invocations\":0,\"gs_primitives\":0"
      << ",\"c_invocations\":" << counters.c_invocations
      << ",\"c_primitives\":" << counters.c_primitives
      << ",\"ps_invocations\":" << counters.ps_invocations
      << ",\"hs_invocations\":0,\"ds_invocations\":0"
      << ",\"cs_invocations\":0,\"ts_invocations\":0"
      << ",\"ms_invocations\":0,\"ms_primitives\":0"
      << ",\"drawlists\":" << counters.drawlists
      << ",\"setup_triangles\":" << counters.setup_triangles
      << ",\"texel_fetches\":" << counters.texel_fetches
      << ",\"virtual_gpu_cycles\":" << counters.virtual_gpu_cycles
      << ",\"tiler_cycles\":" << counters.tiler_cycles
      << ",\"renderer_cycles\":" << counters.renderer_cycles
      << ",\"usc_groups\":" << counters.usc_groups
      << ",\"texture_requests\":" << counters.texture_requests
      << ",\"fifo_stall_events\":" << counters.fifo_stall_events
      << ",\"pool_bytes_in_flight\":" << counters.pool_bytes_in_flight
      << ",\"pool_high_water_bytes\":" << counters.pool_high_water_bytes
      << ",\"vdm_cycles\":" << counters.vdm_cycles
      << ",\"vertex_fetch_cycles\":" << counters.vertex_fetch_cycles
      << ",\"vertex_attribute_fetches\":"
      << counters.vertex_attribute_fetches
      << ",\"vertex_attribute_bytes\":" << counters.vertex_attribute_bytes
      << ",\"pco_decode_cycles\":" << counters.pco_decode_cycles
      << ",\"pco_instructions\":" << counters.pco_instructions
      << ",\"vs_alu_instructions\":" << counters.vs_alu_instructions
      << ",\"vs_tex_instructions\":" << counters.vs_tex_instructions
      << ",\"vs_memory_instructions\":" << counters.vs_memory_instructions
      << ",\"fs_alu_instructions\":" << counters.fs_alu_instructions
      << ",\"fs_tex_instructions\":" << counters.fs_tex_instructions
      << ",\"fs_memory_instructions\":" << counters.fs_memory_instructions
      << ",\"usc_slot_cycles\":" << counters.usc_slot_cycles
      << ",\"usc_cluster_cycles\":" << counters.usc_cluster_cycles
      << ",\"clip_cull_cycles\":" << counters.clip_cull_cycles
      << ",\"tiler_bin_cycles\":" << counters.tiler_bin_cycles
      << ",\"parameter_buffer_cycles\":" << counters.parameter_buffer_cycles
      << ",\"parameter_coefficient_sets\":"
      << counters.parameter_coefficient_sets
      << ",\"parameter_write_bytes\":" << counters.parameter_write_bytes
      << ",\"pds_coefficient_tasks\":" << counters.pds_coefficient_tasks
      << ",\"pds_douti_issues\":" << counters.pds_douti_issues
      << ",\"usc_coefficient_load_bytes\":"
      << counters.usc_coefficient_load_bytes
      << ",\"tile_scheduler_cycles\":" << counters.tile_scheduler_cycles
      << ",\"isp_cycles\":" << counters.isp_cycles
      << ",\"fragment_frontend_cycles\":" << counters.fragment_frontend_cycles
      << ",\"texture_cycles\":" << counters.texture_cycles
      << ",\"pbe_cycles\":" << counters.pbe_cycles
      << ",\"pixel_data_master_transactions\":"
      << counters.pixel_data_master_transactions
      << ",\"pixel_data_master_bytes\":"
      << counters.pixel_data_master_bytes
      << ",\"pixel_data_master_cycles\":"
      << counters.pixel_data_master_cycles
      << ",\"tcu_line_accesses\":" << counters.tcu_line_accesses
      << ",\"tcu_read_accesses\":" << counters.tcu_read_accesses
      << ",\"tcu_hits\":" << counters.tcu_hits
      << ",\"tcu_misses\":" << counters.tcu_misses
      << ",\"tcu_evictions\":" << counters.tcu_evictions
      << ",\"tcu_writebacks\":" << counters.tcu_writebacks
      << ",\"tcu_bypassed\":" << counters.tcu_bypassed
      << ",\"tcu_cycles\":" << counters.tcu_cycles
      << ",\"slc_line_accesses\":" << counters.slc_line_accesses
      << ",\"slc_read_accesses\":" << counters.slc_read_accesses
      << ",\"slc_write_accesses\":" << counters.slc_write_accesses
      << ",\"slc_hits\":" << counters.slc_hits
      << ",\"slc_misses\":" << counters.slc_misses
      << ",\"slc_evictions\":" << counters.slc_evictions
      << ",\"slc_writebacks\":" << counters.slc_writebacks
      << ",\"slc_bypassed\":" << counters.slc_bypassed
      << ",\"slc_cycles\":" << counters.slc_cycles
      << ",\"dram_read_transactions\":"
      << counters.dram_read_transactions
      << ",\"dram_write_transactions\":"
      << counters.dram_write_transactions
      << ",\"dram_read_bytes\":" << counters.dram_read_bytes
      << ",\"dram_write_bytes\":" << counters.dram_write_bytes
      << ",\"dram_cycles\":" << counters.dram_cycles
      << ",\"framebuffer_dram_readback_bytes\":"
      << counters.framebuffer_dram_readback_bytes
      << ",\"tiles_binned\":" << counters.tiles_binned
      << ",\"tiles_scheduled\":" << counters.tiles_scheduled
      << ",\"covered_pixels\":" << counters.covered_pixels
      << ",\"fragment_candidates\":" << counters.fragment_candidates
      << ",\"hsr_rejected_fragments\":" << counters.hsr_rejected_fragments
      << ",\"depth_tested_fragments\":" << counters.depth_tested_fragments
      << ",\"depth_rejected_fragments\":" << counters.depth_rejected_fragments
      << ",\"depth_written_fragments\":" << counters.depth_written_fragments
      << ",\"pbe_color_reads\":" << counters.pbe_color_reads
      << ",\"pbe_blended_fragments\":" << counters.pbe_blended_fragments
      << ",\"pbe_fragment_writes\":" << counters.pbe_fragment_writes
      << ",\"pbe_pixels_written\":" << counters.pbe_pixels_written
      << ",\"functional_frame\":" << counters.functional_frame
      << "},\"drawlist_stats\":[";
  for (std::size_t index = 0; index < drawlists.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    const DrawListStats &drawlist = drawlists[index];
    std::cout << "{\"drawlist\":" << drawlist.drawlist_index
              << ",\"draw_id\":" << drawlist.draw_id << ",\"vs\":";
    EmitShaderStats(drawlist.vertex);
    std::cout << ",\"fs\":";
    EmitShaderStats(drawlist.fragment);
    std::cout << '}';
  }
  std::cout << "]}\n";
  std::cout.flush();
}

void EmitError(std::uint32_t frame, const std::string &message) {
  std::cout << "{\"protocol\":\"pvrgpu-jsonl\",\"version\":1"
            << ",\"schema\":\"" << kSchema << "\",\"type\":\"error\""
            << ",\"backend\":\"pvrgpu\",\"frame\":" << frame << ",\"error\":\""
            << JsonEscape(message) << "\"}\n";
  std::cout.flush();
}

} // namespace

JsonReporter::JsonReporter(sc_core::sc_module_name name, const Options &options,
                           MemoryPool &pool)
    : sc_module(name), options_(options), pool_(pool) {
  SC_THREAD(Run);
}

void JsonReporter::Run() {
  for (unsigned completed = 0; completed < options_.frames; ++completed) {
    const PipelineTxn txn = input.read();
    PipelineState state;
    bool state_loaded = false;
    bool cleanup_attempted = false;
    try {
      state = LoadPipelineState(pool_, txn.state);
      state_loaded = true;
      RequireStage(state.stage, PipelineStage::kFramebufferReady,
                   "JsonReporter");
      const std::uint32_t expected_frame = completed + 1;
      if (txn.frame != expected_frame ||
          state.counters.frame != expected_frame ||
          state.width != options_.width || state.height != options_.height) {
        throw std::runtime_error(
            "JsonReporter frame sequence or surface dimensions mismatch");
      }
      if (state.functional_case != FunctionalCaseFromName(options_.test_case) ||
          state.counters.functional_frame != 1 ||
          state.framebuffer_from_dram != 1 ||
          !HasPoolHandle(state.dram_framebuffer) ||
          HasPoolHandle(state.pbe_framebuffer) ||
          HasPoolHandle(state.slc_writeback_lines)) {
        throw std::runtime_error(
            "JsonReporter received no exclusive DRAM framebuffer readback");
      }

      const std::vector<std::uint8_t> framebuffer =
          LoadArray<std::uint8_t>(pool_, state.dram_framebuffer);
      const std::uint64_t expected_framebuffer_bytes =
          static_cast<std::uint64_t>(state.width) * state.height * 4U;
      if (state.framebuffer_bytes != expected_framebuffer_bytes ||
          state.counters.framebuffer_dram_readback_bytes !=
              expected_framebuffer_bytes ||
          framebuffer.size() != expected_framebuffer_bytes) {
        throw std::runtime_error(
            "JsonReporter DRAM framebuffer byte count mismatch");
      }
      ValidateMemoryPath(options_, state, expected_framebuffer_bytes);
      if (!HasPoolHandle(state.drawlist_stats))
        throw std::runtime_error(
            "JsonReporter received no DrawList statistics");
      const std::vector<DrawListStats> drawlists =
          LoadArray<DrawListStats>(pool_, state.drawlist_stats);
      ValidateDrawListStats(state.counters, drawlists);
      const VertexPcoEvidence vertex_pco =
          BuildVertexPcoEvidence(pool_, state);
      const FragmentPcoEvidence fragment_pco =
          BuildFragmentPcoEvidence(pool_, state);
      std::filesystem::path artifact_path;
      if (!options_.output_dir.empty()) {
        artifact_path = FramePath(options_, state.counters.frame);
        WriteRgbaPngAtomic(artifact_path, framebuffer, state.width,
                           state.height);
      }

      CounterTxn counters = state.counters;
      cleanup_attempted = true;
      ReleaseFunctionalPayloads(pool_, state);
      pool_.Release(txn.state);
      counters.pool_bytes_in_flight = pool_.bytes_in_flight();
      counters.pool_high_water_bytes = pool_.high_water_bytes();
      std::vector<DrawListStats> emitted_drawlists = drawlists;
      if (IsDriverClearColorApiCounterView(options_)) {
        NormalizeClearOnlyApiCounters(counters);
        emitted_drawlists.clear();
      } else if (IsDriverPrimitiveSequenceCounterView(options_)) {
        NormalizeDriverPrimitiveSequenceApiCounters(options_, counters);
        PopulateDriverPrimitiveSequenceDrawLists(options_, emitted_drawlists);
      } else if (IsDriverIndexedQuadCounterView(options_)) {
        NormalizeDriverIndexedQuadApiCounters(options_, counters);
        ScaleDriverIndexedQuadShaderCounters(options_, counters,
                                             emitted_drawlists);
      }
      EmitCounter(options_, counters, emitted_drawlists, vertex_pco, fragment_pco,
                  artifact_path);
      if (!artifact_path.empty()) {
        std::cout << "@CAPTURE: " << options_.test_case
                  << " sample=" << counters.frame
                  << " png=" << artifact_path.filename().string() << '\n';
        std::cout.flush();
      }
    } catch (const std::exception &error) {
      if (!cleanup_attempted) {
        try {
          if (state_loaded)
            ReleaseFunctionalPayloads(pool_, state);
          pool_.Release(txn.state);
        } catch (const std::exception &) {
          // Preserve the original functional/artifact error in the protocol.
        }
      }
      failed_ = true;
      EmitError(txn.frame, error.what());
      sc_core::sc_stop();
      return;
    }
  }

  const std::uint64_t allocations = pool_.allocations();
  const std::uint64_t releases = pool_.releases();
  const std::uint64_t leaks =
      allocations >= releases ? allocations - releases : 0;
  std::cout << "{\"protocol\":\"pvrgpu-jsonl\",\"version\":1"
            << ",\"schema\":\"" << kSchema << "\",\"type\":\"done\""
            << ",\"backend\":\"pvrgpu\",\"frames\":" << options_.frames
            << ",\"pool_allocations\":" << allocations
            << ",\"pool_releases\":" << releases << ",\"pool_leaks\":" << leaks
            << ",\"pool_bytes_in_flight\":" << pool_.bytes_in_flight() << "}\n";
  std::cout.flush();
  if (leaks != 0 || pool_.bytes_in_flight() != 0)
    failed_ = true;
  sc_core::sc_stop();
}

} // namespace pvrgpu::stub
