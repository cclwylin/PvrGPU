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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pvrgpu::stub {

VertexPcoTextureEvidenceClass
ClassifyVertexPcoTextureEvidenceOpcode(PcoOpcode opcode) {
  switch (opcode) {
  case PcoOpcode::kTextureSample:
    return VertexPcoTextureEvidenceClass::kTextureSample;
  case PcoOpcode::kWaitDataFence:
    return VertexPcoTextureEvidenceClass::kWaitDataFence;
  default:
    return VertexPcoTextureEvidenceClass::kUnsupported;
  }
}

namespace {

std::filesystem::path FramePath(const Options &options, std::uint32_t frame) {
  std::ostringstream filename;
  filename << options.test_case << "_sample_" << std::setfill('0')
           << std::setw(6) << frame << ".png";
  return std::filesystem::path(options.output_dir) / filename.str();
}

std::uint64_t CheckedMul(std::uint64_t left, std::uint64_t right,
                         const char *field);

bool HasPrefix(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

void ResetOpaqueBlackFramebuffer(std::uint32_t width,
                                 std::uint32_t height,
                                 std::vector<std::uint8_t> *framebuffer) {
  const std::uint64_t expected_size =
      CheckedMul(CheckedMul(width, height, "opaque black pixels"), 4,
                 "opaque black framebuffer bytes");
  framebuffer->assign(static_cast<std::size_t>(expected_size), 0);
  for (std::uint32_t y = 0; y < height; ++y) {
    const std::size_t alpha_offset =
        (static_cast<std::size_t>(y) * width) * 4U + 3U;
    for (std::uint32_t x = 0; x < width; ++x)
      (*framebuffer)[alpha_offset + static_cast<std::size_t>(x) * 4U] = 255;
  }
}

bool LoadDriverFramebufferSnapshot(const Options &options,
                                   std::uint32_t width,
                                   std::uint32_t height,
                                   std::vector<std::uint8_t> *framebuffer) {
  if (!framebuffer || !options.driver_command.enabled ||
      options.driver_command.framebuffer_rgba8_path.empty()) {
    return false;
  }

  const std::uint64_t expected_size =
      CheckedMul(CheckedMul(width, height, "snapshot pixels"), 4,
                 "snapshot rgba8 bytes");
  std::ifstream input(options.driver_command.framebuffer_rgba8_path,
                      std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "driver framebuffer snapshot is missing: " +
        options.driver_command.framebuffer_rgba8_path);
  }

  framebuffer->assign(static_cast<std::size_t>(expected_size), 0);
  input.read(reinterpret_cast<char *>(framebuffer->data()),
             static_cast<std::streamsize>(framebuffer->size()));
  if (input.gcount() != static_cast<std::streamsize>(framebuffer->size())) {
    throw std::runtime_error("driver framebuffer snapshot is truncated");
  }
  char extra = 0;
  if (input.read(&extra, 1) || input.gcount() != 0) {
    throw std::runtime_error("driver framebuffer snapshot has extra bytes");
  }
  return true;
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
  std::uint64_t internal = 0;
  std::uint64_t fneg = 0;
  std::uint64_t fabs = 0;
  std::uint64_t movi = 0;
  std::uint64_t ffloor = 0;
  std::uint64_t fsub = 0;
  std::uint64_t fge = 0;
  std::uint64_t feq = 0;
  std::uint64_t flt = 0;
  std::uint64_t bitwise_and = 0;
  std::uint64_t csel = 0;
  std::uint64_t fmad = 0;
  std::uint64_t fmin = 0;
  std::uint64_t fmax = 0;
  std::uint64_t frcp = 0;
  std::uint64_t frsq = 0;
  std::uint64_t flog2 = 0;
  std::uint64_t fexp2 = 0;
  std::uint64_t pck_f16 = 0;
  std::uint64_t unpck_f16 = 0;
  std::uint64_t smp = 0;
  std::uint64_t wdf = 0;
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
  std::uint64_t fmul = 0;
  std::uint64_t internal = 0;
  std::uint64_t fneg = 0;
  std::uint64_t fabs = 0;
  std::uint64_t movi = 0;
  std::uint64_t fragcoord = 0;
  std::uint64_t ffloor = 0;
  std::uint64_t fsub = 0;
  std::uint64_t fge = 0;
  std::uint64_t feq = 0;
  std::uint64_t flt = 0;
  std::uint64_t bitwise_and = 0;
  std::uint64_t bitwise_xnor = 0;
  std::uint64_t csel = 0;
  std::uint64_t fmad = 0;
  std::uint64_t fmin = 0;
  std::uint64_t fmax = 0;
  std::uint64_t frcp = 0;
  std::uint64_t frsq = 0;
  std::uint64_t flog2 = 0;
  std::uint64_t fexp2 = 0;
  std::uint64_t pck_f16 = 0;
  std::uint64_t unpck_f16 = 0;
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
    const VertexPcoTextureEvidenceClass texture_class =
        ClassifyVertexPcoTextureEvidenceOpcode(instruction.opcode);
    if (texture_class == VertexPcoTextureEvidenceClass::kTextureSample) {
      ++evidence.smp;
      continue;
    }
    if (texture_class == VertexPcoTextureEvidenceClass::kWaitDataFence) {
      ++evidence.wdf;
      continue;
    }
    switch (instruction.opcode) {
    case PcoOpcode::kInternal:
      ++evidence.internal;
      break;
    case PcoOpcode::kFloatAdd:
    case PcoOpcode::kFloatAddNegateSource0:
      ++evidence.fadd;
      break;
    case PcoOpcode::kFloatMultiply:
      ++evidence.fmul;
      break;
    case PcoOpcode::kMoveBypass:
      ++evidence.mbyp;
      break;
    case PcoOpcode::kMoveImmediate:
      ++evidence.movi;
      break;
    case PcoOpcode::kFloatNegate:
      ++evidence.fneg;
      break;
    case PcoOpcode::kFloatAbs:
      ++evidence.fabs;
      break;
    case PcoOpcode::kFloatFloor:
      ++evidence.ffloor;
      break;
    case PcoOpcode::kFloatSubtract:
      ++evidence.fsub;
      break;
    case PcoOpcode::kFloatGreaterEqual:
      ++evidence.fge;
      break;
    case PcoOpcode::kFloatEqual:
      ++evidence.feq;
      break;
    case PcoOpcode::kFloatLess:
      ++evidence.flt;
      break;
    case PcoOpcode::kBitwiseAnd:
      ++evidence.bitwise_and;
      break;
    case PcoOpcode::kConditionalSelect:
    case PcoOpcode::kConditionalSelectNegateTrue:
    case PcoOpcode::kConditionalSelectGreaterZero:
      ++evidence.csel;
      break;
    case PcoOpcode::kFloatMad:
    case PcoOpcode::kFloatMadNegateSource2:
    case PcoOpcode::kFloatMadNegateSource0:
    case PcoOpcode::kFloatMadNegateSource0Source2:
      ++evidence.fmad;
      break;
    case PcoOpcode::kFloatMin:
      ++evidence.fmin;
      break;
    case PcoOpcode::kFloatMax:
      ++evidence.fmax;
      break;
    case PcoOpcode::kReciprocal:
      ++evidence.frcp;
      break;
    case PcoOpcode::kReciprocalSquareRoot:
      ++evidence.frsq;
      break;
    case PcoOpcode::kFloatLog2:
      ++evidence.flog2;
      break;
    case PcoOpcode::kFloatExp2:
      ++evidence.fexp2;
      break;
    case PcoOpcode::kFloatPackHalfRtne:
    case PcoOpcode::kFloatPackHalfRtz:
      ++evidence.pck_f16;
      break;
    case PcoOpcode::kFloatUnpackHalf:
      ++evidence.unpck_f16;
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
    case PcoOpcode::kFragmentCoordinate:
    case PcoOpcode::kFloatInterpolatePerspective:
    default:
      throw std::runtime_error(
          "JsonReporter vertex PCO opcode histogram received a fragment-only "
          "or unknown operation: " +
          std::to_string(static_cast<unsigned>(instruction.opcode)));
    }
  }
  const std::uint64_t opcode_total =
      evidence.fadd + evidence.fmul + evidence.mbyp + evidence.uvsw_write +
      evidence.uvsw_write_emit_endtask + evidence.uvsw_emit_endtask +
      evidence.internal + evidence.fneg + evidence.fabs + evidence.movi + evidence.ffloor +
      evidence.fsub + evidence.fge + evidence.feq + evidence.flt +
      evidence.bitwise_and + evidence.csel + evidence.fmad + evidence.fmin + evidence.fmax +
      evidence.frcp + evidence.frsq + evidence.flog2 + evidence.fexp2 +
      evidence.pck_f16 + evidence.unpck_f16 + evidence.smp + evidence.wdf;
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
    case PcoOpcode::kInternal:
      ++evidence.internal;
      break;
    case PcoOpcode::kFloatInterpolatePerspective:
      ++evidence.fitrp;
      break;
    case PcoOpcode::kWaitDataFence:
      ++evidence.wdf;
      break;
    case PcoOpcode::kFloatAdd:
    case PcoOpcode::kFloatAddNegateSource0:
      ++evidence.fadd;
      break;
    case PcoOpcode::kFloatMultiply:
      ++evidence.fmul;
      break;
    case PcoOpcode::kMoveBypass:
      ++evidence.mbyp;
      break;
    case PcoOpcode::kMoveImmediate:
      ++evidence.movi;
      break;
    case PcoOpcode::kFloatNegate:
      ++evidence.fneg;
      break;
    case PcoOpcode::kFloatAbs:
      ++evidence.fabs;
      break;
    case PcoOpcode::kFragmentCoordinate:
      ++evidence.fragcoord;
      break;
    case PcoOpcode::kFloatFloor:
      ++evidence.ffloor;
      break;
    case PcoOpcode::kFloatSubtract:
      ++evidence.fsub;
      break;
    case PcoOpcode::kFloatGreaterEqual:
      ++evidence.fge;
      break;
    case PcoOpcode::kFloatEqual:
      ++evidence.feq;
      break;
    case PcoOpcode::kFloatLess:
      ++evidence.flt;
      break;
    case PcoOpcode::kBitwiseAnd:
      ++evidence.bitwise_and;
      break;
    case PcoOpcode::kBitwiseXnor:
      ++evidence.bitwise_xnor;
      break;
    case PcoOpcode::kConditionalSelect:
    case PcoOpcode::kConditionalSelectNegateTrue:
    case PcoOpcode::kConditionalSelectGreaterZero:
      ++evidence.csel;
      break;
    case PcoOpcode::kFloatMad:
    case PcoOpcode::kFloatMadNegateSource2:
    case PcoOpcode::kFloatMadNegateSource0:
    case PcoOpcode::kFloatMadNegateSource0Source2:
      ++evidence.fmad;
      break;
    case PcoOpcode::kFloatMin:
      ++evidence.fmin;
      break;
    case PcoOpcode::kFloatMax:
      ++evidence.fmax;
      break;
    case PcoOpcode::kReciprocal:
      ++evidence.frcp;
      break;
    case PcoOpcode::kReciprocalSquareRoot:
      ++evidence.frsq;
      break;
    case PcoOpcode::kFloatLog2:
      ++evidence.flog2;
      break;
    case PcoOpcode::kFloatExp2:
      ++evidence.fexp2;
      break;
    case PcoOpcode::kFloatPackHalfRtne:
    case PcoOpcode::kFloatPackHalfRtz:
      ++evidence.pck_f16;
      break;
    case PcoOpcode::kFloatUnpackHalf:
      ++evidence.unpck_f16;
      break;
    case PcoOpcode::kTextureSample:
      ++evidence.smp;
      break;
    default:
      throw std::runtime_error(
          "JsonReporter fragment PCO opcode histogram received a vertex-only "
          "or unknown operation: " +
          std::to_string(static_cast<unsigned>(instruction.opcode)));
    }
  }
  if (evidence.fitrp + evidence.wdf + evidence.fadd + evidence.fmul +
          evidence.mbyp + evidence.smp + evidence.internal + evidence.fneg +
          evidence.fabs +
          evidence.movi + evidence.fragcoord + evidence.ffloor +
          evidence.fsub + evidence.fge + evidence.feq + evidence.flt +
          evidence.bitwise_and + evidence.bitwise_xnor + evidence.csel +
          evidence.fmad +
          evidence.fmin + evidence.fmax + evidence.frcp + evidence.frsq +
          evidence.flog2 + evidence.fexp2 +
          evidence.pck_f16 + evidence.unpck_f16 !=
      instructions.size()) {
    throw std::runtime_error(
        "JsonReporter fragment PCO opcode histogram mismatch");
  }
  return evidence;
}

void AddEvidenceCounter(std::uint64_t *destination, std::uint64_t value) {
  if (!destination ||
      value > std::numeric_limits<std::uint64_t>::max() - *destination) {
    throw std::overflow_error("JsonReporter PCO evidence counter overflow");
  }
  *destination += value;
}

void AppendVertexPcoEvidence(const MemoryPool &pool,
                             const PipelineState &state,
                             VertexPcoEvidence *aggregate) {
  if (!aggregate)
    throw std::invalid_argument("missing vertex PCO evidence aggregate");
  const VertexPcoEvidence evidence = BuildVertexPcoEvidence(pool, state);
  const std::vector<std::uint8_t> binary =
      LoadArray<std::uint8_t>(pool, state.vertex_code);
  for (const std::uint8_t byte : binary) {
    aggregate->binary_fnv1a64 ^= byte;
    aggregate->binary_fnv1a64 *= UINT64_C(1099511628211);
  }
#define PVRGPU_ADD_VERTEX_EVIDENCE(field)                                    \
  AddEvidenceCounter(&aggregate->field, evidence.field)
  PVRGPU_ADD_VERTEX_EVIDENCE(binary_bytes);
  PVRGPU_ADD_VERTEX_EVIDENCE(fadd);
  PVRGPU_ADD_VERTEX_EVIDENCE(fmul);
  PVRGPU_ADD_VERTEX_EVIDENCE(mbyp);
  PVRGPU_ADD_VERTEX_EVIDENCE(internal);
  PVRGPU_ADD_VERTEX_EVIDENCE(fneg);
  PVRGPU_ADD_VERTEX_EVIDENCE(fabs);
  PVRGPU_ADD_VERTEX_EVIDENCE(movi);
  PVRGPU_ADD_VERTEX_EVIDENCE(ffloor);
  PVRGPU_ADD_VERTEX_EVIDENCE(fsub);
  PVRGPU_ADD_VERTEX_EVIDENCE(fge);
  PVRGPU_ADD_VERTEX_EVIDENCE(feq);
  PVRGPU_ADD_VERTEX_EVIDENCE(flt);
  PVRGPU_ADD_VERTEX_EVIDENCE(bitwise_and);
  PVRGPU_ADD_VERTEX_EVIDENCE(csel);
  PVRGPU_ADD_VERTEX_EVIDENCE(fmad);
  PVRGPU_ADD_VERTEX_EVIDENCE(fmin);
  PVRGPU_ADD_VERTEX_EVIDENCE(fmax);
  PVRGPU_ADD_VERTEX_EVIDENCE(frcp);
  PVRGPU_ADD_VERTEX_EVIDENCE(frsq);
  PVRGPU_ADD_VERTEX_EVIDENCE(flog2);
  PVRGPU_ADD_VERTEX_EVIDENCE(fexp2);
  PVRGPU_ADD_VERTEX_EVIDENCE(pck_f16);
  PVRGPU_ADD_VERTEX_EVIDENCE(unpck_f16);
  PVRGPU_ADD_VERTEX_EVIDENCE(smp);
  PVRGPU_ADD_VERTEX_EVIDENCE(wdf);
  PVRGPU_ADD_VERTEX_EVIDENCE(uvsw_write);
  PVRGPU_ADD_VERTEX_EVIDENCE(uvsw_write_emit_endtask);
  PVRGPU_ADD_VERTEX_EVIDENCE(uvsw_emit_endtask);
#undef PVRGPU_ADD_VERTEX_EVIDENCE
}

void AppendFragmentPcoEvidence(const MemoryPool &pool,
                               const PipelineState &state,
                               FragmentPcoEvidence *aggregate) {
  if (!aggregate)
    throw std::invalid_argument("missing fragment PCO evidence aggregate");
  const FragmentPcoEvidence evidence = BuildFragmentPcoEvidence(pool, state);
  const std::vector<std::uint8_t> binary =
      LoadArray<std::uint8_t>(pool, state.fragment_code);
  for (const std::uint8_t byte : binary) {
    aggregate->binary_fnv1a64 ^= byte;
    aggregate->binary_fnv1a64 *= UINT64_C(1099511628211);
  }
#define PVRGPU_ADD_FRAGMENT_EVIDENCE(field)                                  \
  AddEvidenceCounter(&aggregate->field, evidence.field)
  PVRGPU_ADD_FRAGMENT_EVIDENCE(binary_bytes);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fitrp);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(wdf);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fadd);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(mbyp);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(smp);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fmul);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(internal);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fneg);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fabs);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(movi);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fragcoord);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(ffloor);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fsub);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fge);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(feq);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(flt);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(bitwise_and);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(bitwise_xnor);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(csel);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fmad);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fmin);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fmax);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(frcp);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(frsq);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(flog2);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(fexp2);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(pck_f16);
  PVRGPU_ADD_FRAGMENT_EVIDENCE(unpck_f16);
#undef PVRGPU_ADD_FRAGMENT_EVIDENCE
}

std::string Fnv1a64Text(std::uint64_t value) {
  std::ostringstream text;
  text << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
       << value;
  return text.str();
}

std::uint64_t Fnv1a64Bytes(const void *data, std::size_t bytes) {
  const auto *source = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (std::size_t index = 0; index < bytes; ++index) {
    hash ^= source[index];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

void DebugSequenceAttachments(const MemoryPool &pool,
                              const PipelineState &state,
                              const DriverCommand &command,
                              std::size_t ordinal) {
  const char *enabled =
      std::getenv("PVRGPU_SEQUENCE_DEBUG_ATTACHMENTS");
  if (!enabled || std::string_view(enabled) != "1")
    return;
  if (!HasPoolHandle(state.dram_framebuffer))
    throw std::runtime_error(
        "JsonReporter attachment debug has no DRAM framebuffer");
  const std::vector<std::uint8_t> color =
      LoadArray<std::uint8_t>(pool, state.dram_framebuffer);
  const std::uint64_t expected_color_bytes =
      static_cast<std::uint64_t>(state.width) * state.height * 4U;
  if (color.size() != expected_color_bytes)
    throw std::runtime_error(
        "JsonReporter attachment debug color byte count mismatch");

  std::size_t nonzero_rgb_pixels = 0;
  std::size_t nonopaque_black_pixels = 0;
  for (std::size_t pixel = 0; pixel < color.size() / 4U; ++pixel) {
    const std::size_t offset = pixel * 4U;
    if (color[offset] != 0 || color[offset + 1U] != 0 ||
        color[offset + 2U] != 0) {
      ++nonzero_rgb_pixels;
    }
    if (color[offset] != 0 || color[offset + 1U] != 0 ||
        color[offset + 2U] != 0 || color[offset + 3U] != 255) {
      ++nonopaque_black_pixels;
    }
  }

  std::cerr << "sequence-attachment-hash ordinal=" << ordinal
            << " extent=" << state.width << 'x' << state.height
            << " color_address=0x" << std::hex
            << state.framebuffer_gpu_address << std::dec
            << " color_source="
            << command.color_attachment_source_command_index
            << " color_load="
            << static_cast<unsigned>(state.color_attachment_load_enable)
            << " color_bytes=" << color.size() << " color_fnv1a64="
            << std::hex << std::setw(16) << std::setfill('0')
            << Fnv1a64Bytes(color.data(), color.size()) << std::dec
            << std::setfill(' ') << " nonzero_rgb=" << nonzero_rgb_pixels
            << " nonopaque_black=" << nonopaque_black_pixels;

  const std::array<std::pair<std::uint32_t, std::uint32_t>, 5> samples = {{
      {0, 0},
      {state.width / 2U, state.height / 2U},
      {state.width == 0 ? 0U : state.width - 1U, 0},
      {0, state.height == 0 ? 0U : state.height - 1U},
      {state.width == 0 ? 0U : state.width - 1U,
       state.height == 0 ? 0U : state.height - 1U},
  }};
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const std::uint32_t x = samples[index].first;
    const std::uint32_t y = samples[index].second;
    if (x >= state.width || y >= state.height)
      continue;
    const std::size_t offset =
        (static_cast<std::size_t>(y) * state.width + x) * 4U;
    std::uint32_t rgba = 0;
    for (std::size_t component = 0; component < 4; ++component)
      rgba |= static_cast<std::uint32_t>(color[offset + component])
              << (component * 8U);
    std::cerr << " p" << index << '=' << x << ',' << y << ",0x"
              << std::hex << std::setw(8) << std::setfill('0') << rgba
              << std::dec << std::setfill(' ');
  }

  const auto print_payload_hash = [&](const char *name, PoolHandle handle,
                                      std::uint64_t bytes) {
    std::cerr << ' ' << name << "_present="
              << (HasPoolHandle(handle) ? 1 : 0);
    if (!HasPoolHandle(handle))
      return;
    const std::vector<std::uint8_t> payload =
        LoadArray<std::uint8_t>(pool, handle);
    if (payload.size() != bytes)
      throw std::runtime_error(
          "JsonReporter attachment debug payload byte count mismatch");
    std::cerr << ' ' << name << "_bytes=" << payload.size() << ' ' << name
              << "_fnv1a64=" << std::hex << std::setw(16)
              << std::setfill('0')
              << Fnv1a64Bytes(payload.data(), payload.size()) << std::dec
              << std::setfill(' ');
  };
  print_payload_hash("color_load", state.color_attachment_load,
                     state.color_attachment_load_bytes);
  print_payload_hash("depth_load", state.depth_attachment_load,
                     state.depth_attachment_load_bytes);
  print_payload_hash("depth_output", state.depth_attachment,
                     state.depth_attachment_bytes);
  std::cerr << '\n';

  const char *dump_dir = std::getenv("PVRGPU_SEQUENCE_DEBUG_DUMP_DIR");
  if (!dump_dir || !dump_dir[0])
    return;
  const auto write_dump = [&](const char *suffix, const void *data,
                              std::size_t size) {
    const std::string path = std::string(dump_dir) + "/ordinal" +
                             std::to_string(ordinal) + suffix;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error(
          "JsonReporter cannot open attachment debug dump");
    output.write(reinterpret_cast<const char *>(data),
                 static_cast<std::streamsize>(size));
    if (!output)
      throw std::runtime_error(
          "JsonReporter cannot write attachment debug dump");
  };
  write_dump("-color.rgba", color.data(), color.size());
  const auto write_payload = [&](const char *suffix, PoolHandle handle) {
    if (!HasPoolHandle(handle))
      return;
    const std::vector<std::uint8_t> payload =
        LoadArray<std::uint8_t>(pool, handle);
    write_dump(suffix, payload.data(), payload.size());
  };
  write_payload("-color-load.bin", state.color_attachment_load);
  write_payload("-depth-load.bin", state.depth_attachment_load);
  write_payload("-depth-output.bin", state.depth_attachment);
}

void DebugSequenceVertexOutputs(const MemoryPool &pool,
                                const PipelineState &state,
                                const DriverCommand &command,
                                std::size_t ordinal) {
  const char *enabled = std::getenv("PVRGPU_SEQUENCE_DEBUG_HASHES");
  if (!enabled || std::string_view(enabled) != "1" ||
      !HasPoolHandle(state.vertex_lanes)) {
    return;
  }
  const std::vector<VertexLane> lanes =
      LoadArray<VertexLane>(pool, state.vertex_lanes);
  std::vector<std::uint32_t> packed;
  std::vector<std::uint32_t> positions;
  std::vector<std::uint32_t> varyings;
  packed.reserve(lanes.size() *
                 (command.position_output_count +
                  command.varying_output_count));
  positions.reserve(lanes.size() * command.position_output_count);
  varyings.reserve(lanes.size() * command.varying_output_count);
  for (const VertexLane &lane : lanes) {
    for (std::size_t component = 0;
         component < command.position_output_count; ++component) {
      const std::uint32_t value = lane.vertex_output[
          command.position_output_start + component];
      packed.push_back(value);
      positions.push_back(value);
    }
    for (std::size_t component = 0;
         component < command.varying_output_count; ++component) {
      const std::uint32_t value = lane.vertex_output[
          command.varying_output_start + component];
      packed.push_back(value);
      varyings.push_back(value);
    }
  }
  const auto print_hash = [&](const char *name,
                              const std::vector<std::uint32_t> &values) {
    std::cerr << "sequence-vertex-hash ordinal=" << ordinal
              << " payload=" << name << " lanes=" << lanes.size()
              << " dwords=" << values.size() << " fnv1a64=" << std::hex
              << std::setw(16) << std::setfill('0')
              << Fnv1a64Bytes(values.data(),
                             values.size() * sizeof(std::uint32_t))
              << std::dec << std::setfill(' ') << '\n';
  };
  print_hash("postvs", packed);
  print_hash("position", positions);
  print_hash("varying", varyings);

  const char *dump_dir = std::getenv("PVRGPU_SEQUENCE_DEBUG_DUMP_DIR");
  if (dump_dir && dump_dir[0]) {
    const auto write_dump = [&](const char *suffix, const void *data,
                                std::size_t size) {
      const std::string path = std::string(dump_dir) + "/ordinal" +
                               std::to_string(ordinal) + suffix;
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      if (!output)
        throw std::runtime_error("JsonReporter cannot open vertex debug dump");
      output.write(reinterpret_cast<const char *>(data),
                   static_cast<std::streamsize>(size));
      if (!output)
        throw std::runtime_error("JsonReporter cannot write vertex debug dump");
    };
    write_dump("-postvs.bin", packed.data(),
               packed.size() * sizeof(std::uint32_t));
    write_dump("-raw-vbo.bin", command.raw_vertex_data.data(),
               command.raw_vertex_data.size());
    write_dump("-vertex-shared.bin", command.vertex_shared.data(),
               command.vertex_shared.size() * sizeof(std::uint32_t));
    write_dump("-vertex-pco.bin", command.vertex_pco.data(),
               command.vertex_pco.size());
    write_dump("-fragment-shared.bin", command.fragment_shared.data(),
               command.fragment_shared.size() * sizeof(std::uint32_t));
    write_dump("-fragment-pco.bin", command.fragment_pco.data(),
               command.fragment_pco.size());

    // Preserve the complete setup inputs/evidence when a native sequence is
    // being investigated.  The post-VS stream alone cannot identify which
    // clipped primitive owned a pixel, while the coefficient bank alone
    // cannot prove the entry varyings from which it was derived.  These are
    // diagnostic copies of real model payloads; they never feed rendering.
    const auto write_pool_payload = [&](const char *suffix,
                                        const PoolHandle handle,
                                        const auto *type_tag) {
      using Value =
          std::remove_cv_t<std::remove_pointer_t<decltype(type_tag)>>;
      if (!HasPoolHandle(handle))
        return;
      const std::vector<Value> values = LoadArray<Value>(pool, handle);
      write_dump(suffix, values.data(), values.size() * sizeof(Value));
    };
    write_pool_payload("-raster-triangles.bin", state.raster_triangles,
                       static_cast<const RasterTriangle *>(nullptr));
    write_pool_payload("-raster-vtxout.bin", state.raster_vertex_outputs,
                       static_cast<const std::uint32_t *>(nullptr));
    write_pool_payload("-usc-coefficients.bin", state.usc_coefficient_banks,
                       static_cast<const std::uint32_t *>(nullptr));
    write_pool_payload("-usc-tasks.bin", state.usc_fragment_tasks,
                       static_cast<const UscFragmentTask *>(nullptr));
    write_pool_payload("-fragment-quads.bin", state.fragment_quads,
                       static_cast<const FragmentQuad *>(nullptr));
    write_pool_payload("-fragment-lanes.bin", state.fragment_shader_lanes,
                       static_cast<const FragmentShaderLane *>(nullptr));
  }
}

void AddDrawListCounter(std::uint64_t &counter, std::uint64_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - counter)
    throw std::overflow_error("JsonReporter DrawList counter overflow");
  counter += amount;
}

void AccumulatePhysicalCounters(CounterTxn *aggregate,
                                const CounterTxn &physical) {
  if (!aggregate)
    throw std::invalid_argument("missing physical counter aggregate");
#define PVRGPU_ADD_COUNTER(field)                                             \
  AddDrawListCounter(aggregate->field, physical.field)
  PVRGPU_ADD_COUNTER(ia_vertices);
  PVRGPU_ADD_COUNTER(ia_primitives);
  PVRGPU_ADD_COUNTER(vs_invocations);
  PVRGPU_ADD_COUNTER(gs_invocations);
  PVRGPU_ADD_COUNTER(gs_primitives);
  PVRGPU_ADD_COUNTER(c_invocations);
  PVRGPU_ADD_COUNTER(c_primitives);
  PVRGPU_ADD_COUNTER(ps_invocations);
  PVRGPU_ADD_COUNTER(hs_invocations);
  PVRGPU_ADD_COUNTER(ds_invocations);
  PVRGPU_ADD_COUNTER(drawlists);
  PVRGPU_ADD_COUNTER(setup_triangles);
  PVRGPU_ADD_COUNTER(texel_fetches);
  PVRGPU_ADD_COUNTER(virtual_gpu_cycles);
  PVRGPU_ADD_COUNTER(tiler_cycles);
  PVRGPU_ADD_COUNTER(renderer_cycles);
  PVRGPU_ADD_COUNTER(usc_groups);
  PVRGPU_ADD_COUNTER(texture_requests);
  aggregate->fifo_stall_events =
      std::max(aggregate->fifo_stall_events, physical.fifo_stall_events);
  aggregate->pool_bytes_in_flight = physical.pool_bytes_in_flight;
  aggregate->pool_high_water_bytes =
      std::max(aggregate->pool_high_water_bytes,
               physical.pool_high_water_bytes);
  PVRGPU_ADD_COUNTER(vdm_cycles);
  PVRGPU_ADD_COUNTER(vertex_fetch_cycles);
  PVRGPU_ADD_COUNTER(vertex_attribute_fetches);
  PVRGPU_ADD_COUNTER(vertex_attribute_bytes);
  PVRGPU_ADD_COUNTER(pco_decode_cycles);
  PVRGPU_ADD_COUNTER(pco_instructions);
  PVRGPU_ADD_COUNTER(vs_alu_instructions);
  PVRGPU_ADD_COUNTER(vs_tex_instructions);
  PVRGPU_ADD_COUNTER(vs_memory_instructions);
  PVRGPU_ADD_COUNTER(fs_alu_instructions);
  PVRGPU_ADD_COUNTER(fs_tex_instructions);
  PVRGPU_ADD_COUNTER(fs_memory_instructions);
  PVRGPU_ADD_COUNTER(usc_slot_cycles);
  PVRGPU_ADD_COUNTER(usc_cluster_cycles);
  PVRGPU_ADD_COUNTER(clip_cull_cycles);
  PVRGPU_ADD_COUNTER(tiler_bin_cycles);
  PVRGPU_ADD_COUNTER(parameter_buffer_cycles);
  PVRGPU_ADD_COUNTER(parameter_coefficient_sets);
  PVRGPU_ADD_COUNTER(parameter_write_bytes);
  PVRGPU_ADD_COUNTER(pds_coefficient_tasks);
  PVRGPU_ADD_COUNTER(pds_douti_issues);
  PVRGPU_ADD_COUNTER(usc_coefficient_load_bytes);
  PVRGPU_ADD_COUNTER(tile_scheduler_cycles);
  PVRGPU_ADD_COUNTER(isp_cycles);
  PVRGPU_ADD_COUNTER(fragment_frontend_cycles);
  PVRGPU_ADD_COUNTER(texture_cycles);
  PVRGPU_ADD_COUNTER(pbe_cycles);
  PVRGPU_ADD_COUNTER(pixel_data_master_transactions);
  PVRGPU_ADD_COUNTER(pixel_data_master_bytes);
  PVRGPU_ADD_COUNTER(pixel_data_master_cycles);
  PVRGPU_ADD_COUNTER(tcu_line_accesses);
  PVRGPU_ADD_COUNTER(tcu_read_accesses);
  PVRGPU_ADD_COUNTER(tcu_hits);
  PVRGPU_ADD_COUNTER(tcu_misses);
  PVRGPU_ADD_COUNTER(tcu_evictions);
  PVRGPU_ADD_COUNTER(tcu_writebacks);
  PVRGPU_ADD_COUNTER(tcu_bypassed);
  PVRGPU_ADD_COUNTER(tcu_cycles);
  PVRGPU_ADD_COUNTER(slc_line_accesses);
  PVRGPU_ADD_COUNTER(slc_read_accesses);
  PVRGPU_ADD_COUNTER(slc_write_accesses);
  PVRGPU_ADD_COUNTER(slc_hits);
  PVRGPU_ADD_COUNTER(slc_misses);
  PVRGPU_ADD_COUNTER(slc_evictions);
  PVRGPU_ADD_COUNTER(slc_writebacks);
  PVRGPU_ADD_COUNTER(slc_bypassed);
  PVRGPU_ADD_COUNTER(slc_cycles);
  PVRGPU_ADD_COUNTER(dram_read_transactions);
  PVRGPU_ADD_COUNTER(dram_write_transactions);
  PVRGPU_ADD_COUNTER(dram_read_bytes);
  PVRGPU_ADD_COUNTER(dram_write_bytes);
  PVRGPU_ADD_COUNTER(dram_cycles);
  PVRGPU_ADD_COUNTER(memory_direct_read_bytes);
  PVRGPU_ADD_COUNTER(memory_direct_write_bytes);
  PVRGPU_ADD_COUNTER(framebuffer_dram_readback_bytes);
  PVRGPU_ADD_COUNTER(tiles_binned);
  PVRGPU_ADD_COUNTER(tiles_scheduled);
  PVRGPU_ADD_COUNTER(covered_pixels);
  PVRGPU_ADD_COUNTER(fragment_candidates);
  PVRGPU_ADD_COUNTER(hsr_rejected_fragments);
  PVRGPU_ADD_COUNTER(depth_tested_fragments);
  PVRGPU_ADD_COUNTER(depth_rejected_fragments);
  PVRGPU_ADD_COUNTER(depth_written_fragments);
  PVRGPU_ADD_COUNTER(pbe_color_reads);
  PVRGPU_ADD_COUNTER(pbe_blended_fragments);
  PVRGPU_ADD_COUNTER(pbe_fragment_writes);
  PVRGPU_ADD_COUNTER(pbe_pixels_written);
#undef PVRGPU_ADD_COUNTER
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
  if (state.memory_mode != options.memory_mode ||
      state.cache_bypass != static_cast<std::uint8_t>(options.cache_bypass) ||
      counters.pixel_data_master_transactions != 1 ||
      counters.pixel_data_master_bytes != expected_bytes ||
      counters.pixel_data_master_cycles == 0 ||
      counters.framebuffer_dram_readback_bytes != expected_bytes ||
      counters.dram_cycles !=
          counters.dram_read_transactions + counters.dram_write_transactions) {
    throw std::runtime_error(
        "JsonReporter framebuffer memory-path counter mismatch");
  }

  if (counters.tcu_line_accesses != 0 || counters.tcu_read_accesses != 0 ||
      counters.tcu_hits != 0 || counters.tcu_misses != 0 ||
      counters.tcu_evictions != 0 || counters.tcu_writebacks != 0 ||
      counters.tcu_bypassed != 0 || counters.tcu_cycles != 0) {
    throw std::runtime_error(
        "JsonReporter TCU counters should be idle under unified memory");
  }

  if (options.memory_mode == MemoryMode::kDirect) {
    if (counters.slc_line_accesses != 0 ||
        counters.slc_read_accesses != 0 ||
        counters.slc_write_accesses != 0 || counters.slc_hits != 0 ||
        counters.slc_misses != 0 || counters.slc_evictions != 0 ||
        counters.slc_writebacks != 0 || counters.slc_bypassed != 0 ||
        counters.slc_cycles != 0 ||
        counters.dram_read_transactions != 0 ||
        counters.dram_write_transactions != 0 ||
        counters.dram_read_bytes != 0 || counters.dram_write_bytes != 0 ||
        counters.memory_direct_read_bytes < expected_bytes ||
        counters.memory_direct_write_bytes < expected_bytes) {
      throw std::runtime_error(
          "JsonReporter direct memory-path mismatch");
    }
    return;
  }

  if (counters.memory_direct_read_bytes != 0 ||
      counters.memory_direct_write_bytes != 0 ||
      counters.dram_read_transactions == 0 ||
      counters.dram_write_transactions == 0 ||
      counters.dram_read_bytes < expected_bytes ||
      counters.dram_write_bytes < expected_bytes) {
    throw std::runtime_error("JsonReporter modeled memory-path mismatch");
  }

  if (options.memory_mode == MemoryMode::kBypass) {
    if (counters.slc_line_accesses != 0 ||
        counters.slc_read_accesses != 0 ||
        counters.slc_write_accesses != 0 || counters.slc_hits != 0 ||
        counters.slc_misses != 0 || counters.slc_evictions != 0 ||
        counters.slc_writebacks != 0 || counters.slc_cycles != 0 ||
        counters.slc_bypassed == 0) {
      throw std::runtime_error(
          "JsonReporter cache-bypass memory-path mismatch");
    }
    return;
  }

  if (options.memory_mode != MemoryMode::kCache) {
    throw std::runtime_error("JsonReporter unknown memory mode");
  }
  if (counters.slc_line_accesses == 0 ||
      counters.slc_line_accesses !=
          counters.slc_read_accesses + counters.slc_write_accesses ||
      counters.slc_hits + counters.slc_misses != counters.slc_line_accesses ||
      counters.slc_bypassed != 0 ||
      counters.slc_cycles != counters.slc_line_accesses ||
      counters.slc_writebacks == 0 ||
      counters.dram_write_transactions != counters.slc_writebacks ||
      counters.dram_write_bytes !=
          counters.slc_writebacks * kDramLineWriteBytes) {
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

bool IsDriverPcoTrianglesCounterView(const Options &options) {
  return options.driver_command.enabled &&
         (options.driver_command.command == "draw_pco_triangles" ||
          options.driver_command.command == "draw_pco_sequence");
}

// Compares the counters a driver command carries against what SystemC
// measured.  Nothing is adopted; the comparison exists so a regression run can
// enumerate exactly which cases still disagree with the model, which is the
// list of pipeline features that remain to be implemented.
void ReportAdoptedCounterDrift(const Options &options,
                               const CounterTxn &counters) {
  const DriverCommand &command = options.driver_command;
  struct Field {
    const char *name;
    std::uint64_t claimed;
    std::uint64_t measured;
  };
  const Field fields[] = {
      {"vs_invocations", command.vs_invocations, counters.vs_invocations},
      {"gs_invocations", command.gs_invocations, counters.gs_invocations},
      {"gs_primitives", command.gs_primitives, counters.gs_primitives},
      {"c_primitives", command.clip_primitives, counters.c_primitives},
      {"ps_invocations", command.ps_invocations, counters.ps_invocations},
      {"hs_invocations", command.hs_invocations, counters.hs_invocations},
      {"ds_invocations", command.ds_invocations, counters.ds_invocations},
      {"setup_triangles", command.setup_triangles, counters.setup_triangles},
      {"texel_fetches", command.semantic_texel_fetches, counters.texel_fetches},
  };
  for (const Field &field : fields) {
    if (field.claimed == 0 || field.claimed == field.measured)
      continue;
    std::fprintf(stderr,
                 "pvrgpu-counter-drift case=%s field=%s claimed=%llu "
                 "measured=%llu\n",
                 options.test_case.c_str(), field.name,
                 static_cast<unsigned long long>(field.claimed),
                 static_cast<unsigned long long>(field.measured));
  }
}

void NormalizeDriverPcoTrianglesApiCounters(const Options &options,
                                            CounterTxn &counters) {
  const DriverCommand &command = options.driver_command;
  if (!options.driver_commands.empty()) {
    const bool generic_sequence = command.command == "draw_pco_sequence";
    const bool has_captured_totals =
        (generic_sequence ||
         command.draw_count == options.driver_commands.size()) &&
        command.draw_count != 0 && command.ia_vertices != 0 &&
        command.ia_primitives != 0 && command.clip_invocations != 0;
    if (!has_captured_totals) {
      throw std::runtime_error(
          "ordered PCO sequence has no complete API counter metadata");
    }
    // Input-assembly totals are decided by the draw calls themselves, so the
    // driver knows them exactly and they are adopted.  These are facts about
    // the submitted geometry, not about what the hardware did with it.
    counters.ia_vertices = command.ia_vertices;
    counters.ia_primitives = command.ia_primitives;
    counters.c_invocations = command.clip_invocations;
    counters.drawlists = command.draw_count;

    // Everything from vertex shading onwards is what the pipeline actually
    // did, and only SystemC can know it.  A driver-supplied value is treated
    // as a cross-check against the measurement, never as a substitute for it:
    // adopting one would let a capture profile decide the answer without any
    // rasterization taking place.
    ReportAdoptedCounterDrift(options, counters);
    return;
  }
  if (command.depth_enable == 0) {
    counters.ps_invocations = counters.pbe_fragment_writes;
    return;
  }
  if (command.depth_enable != 1 || command.depth_write != 1 ||
      counters.depth_written_fragments > counters.depth_tested_fragments ||
      counters.depth_written_fragments < counters.pbe_fragment_writes) {
    throw std::runtime_error(
        "driver PCO early-depth counter view is inconsistent");
  }

  /* Mesa's PS_INVOCATIONS query counts samples that pass early depth before
   * PowerVR HSR collapses several opaque owners to the final visible owner.
   * Keep DrawList execution statistics as the actual ISS lanes, but expose
   * the API counter at the same observation point as the llvmpipe golden. */
  counters.ps_invocations = counters.depth_written_fragments;
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
  // texel_fetches stays at what the texture unit measured.  The driver can
  // estimate the figure from the sampler state it saw, but an estimate is not
  // an observation and must not replace one.
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

std::uint64_t PartitionCounter(std::uint64_t value, std::size_t partitions,
                               std::size_t partition) {
  if (partitions == 0 || partition >= partitions)
    throw std::invalid_argument("invalid DrawList counter partition");
  return value / partitions +
         (partition < value % partitions ? UINT64_C(1) : UINT64_C(0));
}

void PartitionShaderExecution(const DrawListShaderStats &physical,
                              std::size_t partitions,
                              std::size_t partition,
                              DrawListShaderStats *logical) {
  if (!logical)
    throw std::invalid_argument("missing logical DrawList shader stats");
  *logical = physical;
  logical->invocations =
      PartitionCounter(physical.invocations, partitions, partition);
  logical->executed_alu_instructions = PartitionCounter(
      physical.executed_alu_instructions, partitions, partition);
  logical->executed_tex_instructions = PartitionCounter(
      physical.executed_tex_instructions, partitions, partition);
  logical->executed_memory_instructions = PartitionCounter(
      physical.executed_memory_instructions, partitions, partition);
}

void ExpandGenericPcoSequenceDrawLists(
    std::uint32_t logical_count,
    std::vector<DrawListStats> *drawlists) {
  if (!drawlists || drawlists->empty() || logical_count < drawlists->size())
    throw std::runtime_error(
        "generic PCO sequence DrawList mapping is invalid");
  const std::vector<DrawListStats> physical = *drawlists;
  std::vector<std::size_t> bucket_sizes(physical.size(), 0);
  for (std::size_t logical = 0; logical < logical_count; ++logical) {
    const std::size_t physical_index =
        logical * physical.size() / logical_count;
    ++bucket_sizes.at(physical_index);
  }
  if (std::any_of(bucket_sizes.begin(), bucket_sizes.end(),
                  [](std::size_t count) { return count == 0; })) {
    throw std::runtime_error(
        "generic PCO sequence lost physical DrawList evidence");
  }

  std::vector<std::size_t> bucket_ordinals(physical.size(), 0);
  drawlists->clear();
  drawlists->reserve(logical_count);
  for (std::size_t logical = 0; logical < logical_count; ++logical) {
    const std::size_t physical_index =
        logical * physical.size() / logical_count;
    const std::size_t partition = bucket_ordinals[physical_index]++;
    DrawListStats stats = physical[physical_index];
    stats.drawlist_index = static_cast<std::uint32_t>(logical);
    stats.draw_id = stats.drawlist_index;
    PartitionShaderExecution(physical[physical_index].vertex,
                             bucket_sizes[physical_index], partition,
                             &stats.vertex);
    PartitionShaderExecution(physical[physical_index].fragment,
                             bucket_sizes[physical_index], partition,
                             &stats.fragment);
    drawlists->push_back(stats);
  }
}

void EmitCounter(const Options &options, const CounterTxn &counters,
                 const std::vector<DrawListStats> &drawlists,
                 const VertexPcoEvidence &vertex_pco,
                 const FragmentPcoEvidence &fragment_pco,
                 const std::filesystem::path &artifact_path) {
  const std::uint64_t cs_invocations =
      options.driver_command.enabled ? options.driver_command.cs_invocations
                                     : 0;
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
    if (!options.driver_commands.empty()) {
      std::cout << ",\"driver_command_sequence_length\":"
                << options.driver_commands.size();
    }
    if (command.command == "draw_textured_triangles") {
      std::cout << ",\"driver_texture_width\":" << command.texture_width
                << ",\"driver_texture_height\":" << command.texture_height;
    }
  } else {
    std::cout << ",\"command_source\":\"builtin-glbench-fixture\"";
  }
  std::cout << ",\"timing_provenance\":\"uncalibrated\""
            << ",\"cache_bypass\":"
            << (options.cache_bypass ? "true" : "false")
            << ",\"memory_mode\":\""
            << MemoryModeName(options.memory_mode) << "\""
            << ",\"cache_simulated\":"
            << (options.memory_mode == MemoryMode::kCache ? "true" : "false")
            << ",\"framebuffer_source\":\"dram-readback\""
            << ",\"vertex_pco_binary\":{\"fingerprint\":\""
            << Fnv1a64Text(vertex_pco.binary_fnv1a64)
            << "\",\"bytes\":" << vertex_pco.binary_bytes << "}"
            << ",\"vertex_pco_opcodes\":{\"fadd\":" << vertex_pco.fadd;
  if (vertex_pco.fmul != 0)
    std::cout << ",\"fmul\":" << vertex_pco.fmul;
  if (vertex_pco.fmad != 0)
    std::cout << ",\"fmad\":" << vertex_pco.fmad;
  if (vertex_pco.fmin != 0)
    std::cout << ",\"fmin\":" << vertex_pco.fmin;
  if (vertex_pco.fmax != 0)
    std::cout << ",\"fmax\":" << vertex_pco.fmax;
  if (vertex_pco.frcp != 0)
    std::cout << ",\"frcp\":" << vertex_pco.frcp;
  if (vertex_pco.frsq != 0)
    std::cout << ",\"frsq\":" << vertex_pco.frsq;
  if (vertex_pco.flog2 != 0)
    std::cout << ",\"flog2\":" << vertex_pco.flog2;
  if (vertex_pco.fexp2 != 0)
    std::cout << ",\"fexp2\":" << vertex_pco.fexp2;
  if (vertex_pco.pck_f16 != 0)
    std::cout << ",\"pck_f16\":" << vertex_pco.pck_f16;
  if (vertex_pco.unpck_f16 != 0)
    std::cout << ",\"unpck_f16\":" << vertex_pco.unpck_f16;
  if (vertex_pco.smp != 0)
    std::cout << ",\"smp\":" << vertex_pco.smp;
  if (vertex_pco.wdf != 0)
    std::cout << ",\"wdf\":" << vertex_pco.wdf;
  if (vertex_pco.movi != 0)
    std::cout << ",\"movi\":" << vertex_pco.movi;
  if (vertex_pco.fneg != 0)
    std::cout << ",\"fneg\":" << vertex_pco.fneg;
  if (vertex_pco.fabs != 0)
    std::cout << ",\"fabs\":" << vertex_pco.fabs;
  if (vertex_pco.ffloor != 0)
    std::cout << ",\"ffloor\":" << vertex_pco.ffloor;
  if (vertex_pco.fsub != 0)
    std::cout << ",\"fsub\":" << vertex_pco.fsub;
  if (vertex_pco.fge != 0)
    std::cout << ",\"fge\":" << vertex_pco.fge;
  if (vertex_pco.feq != 0)
    std::cout << ",\"feq\":" << vertex_pco.feq;
  if (vertex_pco.flt != 0)
    std::cout << ",\"flt\":" << vertex_pco.flt;
  if (vertex_pco.bitwise_and != 0)
    std::cout << ",\"bitwise_and\":" << vertex_pco.bitwise_and;
  if (vertex_pco.csel != 0)
    std::cout << ",\"csel\":" << vertex_pco.csel;
  if (vertex_pco.internal != 0)
    std::cout << ",\"internal\":" << vertex_pco.internal;
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
  if (fragment_pco.fmul != 0)
    std::cout << ",\"fmul\":" << fragment_pco.fmul;
  if (fragment_pco.fmad != 0)
    std::cout << ",\"fmad\":" << fragment_pco.fmad;
  if (fragment_pco.fmin != 0)
    std::cout << ",\"fmin\":" << fragment_pco.fmin;
  if (fragment_pco.fmax != 0)
    std::cout << ",\"fmax\":" << fragment_pco.fmax;
  if (fragment_pco.frcp != 0)
    std::cout << ",\"frcp\":" << fragment_pco.frcp;
  if (fragment_pco.frsq != 0)
    std::cout << ",\"frsq\":" << fragment_pco.frsq;
  if (fragment_pco.flog2 != 0)
    std::cout << ",\"flog2\":" << fragment_pco.flog2;
  if (fragment_pco.fexp2 != 0)
    std::cout << ",\"fexp2\":" << fragment_pco.fexp2;
  if (fragment_pco.movi != 0)
    std::cout << ",\"movi\":" << fragment_pco.movi;
  if (fragment_pco.fneg != 0)
    std::cout << ",\"fneg\":" << fragment_pco.fneg;
  if (fragment_pco.fabs != 0)
    std::cout << ",\"fabs\":" << fragment_pco.fabs;
  if (fragment_pco.fragcoord != 0)
    std::cout << ",\"fragcoord\":" << fragment_pco.fragcoord;
  if (fragment_pco.ffloor != 0)
    std::cout << ",\"ffloor\":" << fragment_pco.ffloor;
  if (fragment_pco.fsub != 0)
    std::cout << ",\"fsub\":" << fragment_pco.fsub;
  if (fragment_pco.fge != 0)
    std::cout << ",\"fge\":" << fragment_pco.fge;
  if (fragment_pco.feq != 0)
    std::cout << ",\"feq\":" << fragment_pco.feq;
  if (fragment_pco.flt != 0)
    std::cout << ",\"flt\":" << fragment_pco.flt;
  if (fragment_pco.bitwise_and != 0)
    std::cout << ",\"bitwise_and\":" << fragment_pco.bitwise_and;
  if (fragment_pco.bitwise_xnor != 0)
    std::cout << ",\"bitwise_xnor\":" << fragment_pco.bitwise_xnor;
  if (fragment_pco.csel != 0)
    std::cout << ",\"csel\":" << fragment_pco.csel;
  if (fragment_pco.pck_f16 != 0)
    std::cout << ",\"pck_f16\":" << fragment_pco.pck_f16;
  if (fragment_pco.unpck_f16 != 0)
    std::cout << ",\"unpck_f16\":" << fragment_pco.unpck_f16;
  if (fragment_pco.internal != 0)
    std::cout << ",\"internal\":" << fragment_pco.internal;
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
      << ",\"gs_invocations\":" << counters.gs_invocations
      << ",\"gs_primitives\":" << counters.gs_primitives
      << ",\"c_invocations\":" << counters.c_invocations
      << ",\"c_primitives\":" << counters.c_primitives
      << ",\"ps_invocations\":" << counters.ps_invocations
      << ",\"hs_invocations\":" << counters.hs_invocations
      << ",\"ds_invocations\":" << counters.ds_invocations
      << ",\"cs_invocations\":" << cs_invocations
      << ",\"ts_invocations\":0"
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
      << ",\"memory_direct_read_bytes\":"
      << counters.memory_direct_read_bytes
      << ",\"memory_direct_write_bytes\":"
      << counters.memory_direct_write_bytes
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
                           MemoryPool &pool,
                           sc_core::sc_event *sequence_completion)
    : sc_module(name), options_(options), pool_(pool),
      sequence_completion_(sequence_completion) {
  SC_THREAD(Run);
}

void JsonReporter::Run() {
  if (!options_.driver_commands.empty()) {
    const bool generic_sequence =
        options_.driver_command.command == "draw_pco_sequence";
    CounterTxn aggregate;
    aggregate.frame = 1;
    aggregate.functional_frame = 1;
    std::vector<DrawListStats> aggregate_drawlists;
    aggregate_drawlists.reserve(options_.driver_commands.size());
    VertexPcoEvidence vertex_pco;
    FragmentPcoEvidence fragment_pco;
    std::vector<std::uint8_t> final_framebuffer;
    std::uint32_t final_width = 0;
    std::uint32_t final_height = 0;
    std::vector<std::uint64_t> sequence_color_addresses(
        options_.driver_commands.size(), 0);
    std::vector<std::uint64_t> sequence_depth_addresses(
        options_.driver_commands.size(), 0);
    if (generic_sequence &&
        !ResolveSequenceAttachmentAddresses(options_.driver_commands,
                                            &sequence_color_addresses,
                                            &sequence_depth_addresses)) {
      throw std::runtime_error(
          "JsonReporter PCO sequence attachment dependencies do not fit the "
          "address map");
    }

    for (std::size_t completed = 0;
         completed < options_.driver_commands.size(); ++completed) {
      const PipelineTxn txn = input.read();
      PipelineState state;
      bool state_loaded = false;
      bool cleanup_attempted = false;
      try {
        state = LoadPipelineState(pool_, txn.state);
        state_loaded = true;
        RequireStage(state.stage, PipelineStage::kFramebufferReady,
                     "JsonReporter");
        const std::uint32_t expected_submission =
            static_cast<std::uint32_t>(completed + 1U);
        const DriverCommand &physical_command =
            options_.driver_commands[completed];
        const std::uint32_t expected_width =
            generic_sequence ? physical_command.framebuffer_width
                             : options_.width;
        const std::uint32_t expected_height =
            generic_sequence ? physical_command.framebuffer_height
                             : options_.height;
        if (txn.frame != expected_submission ||
            txn.sequence != expected_submission ||
            state.sequence != expected_submission ||
            state.counters.frame != expected_submission ||
            state.width != expected_width || state.height != expected_height) {
          throw std::runtime_error(
              "JsonReporter PCO submission sequence mismatch");
        }
        if (state.functional_case !=
                FunctionalCaseFromName(options_.test_case) ||
            state.counters.functional_frame != 1 ||
            state.framebuffer_from_dram != 1 ||
            !HasPoolHandle(state.dram_framebuffer) ||
            HasPoolHandle(state.pbe_framebuffer) ||
            HasPoolHandle(state.slc_writeback_lines)) {
          throw std::runtime_error(
              "JsonReporter PCO sequence has no exclusive DRAM readback");
        }
        if (generic_sequence) {
          const std::uint64_t expected_color_address =
              sequence_color_addresses[completed];
          if (state.framebuffer_gpu_address != expected_color_address) {
            throw std::runtime_error(
                "JsonReporter PCO sequence color attachment address mismatch");
          }
          bool depth_producer = false;
          for (std::size_t future = completed + 1U;
               future < options_.driver_commands.size(); ++future) {
            depth_producer =
                depth_producer ||
                options_.driver_commands[future]
                        .depth_attachment_source_command_index == completed;
            for (const DriverPcoSampledTexture &texture :
                 options_.driver_commands[future].sampled_textures) {
              depth_producer =
                  depth_producer ||
                  (texture.source ==
                       DriverPcoTextureSource::kPreviousDepthAttachment &&
                   texture.producer_command_index == completed);
            }
          }
          const bool has_depth = physical_command.depth_format != 0;
          const std::uint64_t expected_depth_bytes =
              has_depth
                  ? static_cast<std::uint64_t>(state.width) * state.height *
                        DepthAttachmentBytesPerPixel(
                            physical_command.depth_format)
                  : 0;
          const bool color_load =
              physical_command.color_attachment_source_command_index !=
              kDriverPcoNewAttachment;
          const bool depth_load =
              has_depth &&
              physical_command.depth_attachment_source_command_index !=
                  kDriverPcoNewAttachment;
          const std::uint64_t expected_color_bytes =
              static_cast<std::uint64_t>(state.width) * state.height * 4U;
          if (state.color_attachment_load_enable != (color_load ? 1U : 0U) ||
              state.color_attachment_load_bytes !=
                  (color_load ? expected_color_bytes : 0U) ||
              HasPoolHandle(state.color_attachment_load) != color_load ||
              state.depth_attachment_load_enable != (depth_load ? 1U : 0U) ||
              state.depth_attachment_load_bytes !=
                  (depth_load ? expected_depth_bytes : 0U) ||
              HasPoolHandle(state.depth_attachment_load) != depth_load ||
              state.depth_attachment_format !=
                  physical_command.depth_format ||
              state.depth_attachment_gpu_address !=
                  sequence_depth_addresses[completed]) {
            throw std::runtime_error(
                "JsonReporter PCO sequence attachment LOAD evidence mismatch");
          }
          if (depth_producer
                  ? (state.capture_depth_attachment != 1 ||
                     state.depth_attachment_ready != 1 ||
                     !HasPoolHandle(state.depth_attachment) ||
                     state.depth_attachment_bytes != expected_depth_bytes)
                  : (state.capture_depth_attachment != 0 ||
                     state.depth_attachment_ready != 0 ||
                     HasPoolHandle(state.depth_attachment) ||
                     state.depth_attachment_bytes != 0)) {
            throw std::runtime_error(
                "JsonReporter PCO sequence depth attachment evidence mismatch");
          }
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
              "JsonReporter PCO sequence DRAM byte count mismatch");
        }
        ValidateMemoryPath(options_, state, expected_framebuffer_bytes);
        DebugSequenceAttachments(pool_, state, physical_command, completed);
        DebugSequenceVertexOutputs(pool_, state, physical_command, completed);
        if (!HasPoolHandle(state.drawlist_stats)) {
          throw std::runtime_error(
              "JsonReporter PCO sequence has no DrawList statistics");
        }
        std::vector<DrawListStats> drawlists =
            LoadArray<DrawListStats>(pool_, state.drawlist_stats);
        ValidateDrawListStats(state.counters, drawlists);
        AppendVertexPcoEvidence(pool_, state, &vertex_pco);
        AppendFragmentPcoEvidence(pool_, state, &fragment_pco);
        AccumulatePhysicalCounters(&aggregate, state.counters);
        for (DrawListStats &drawlist : drawlists) {
          drawlist.drawlist_index =
              static_cast<std::uint32_t>(aggregate_drawlists.size());
          drawlist.draw_id = drawlist.drawlist_index;
          aggregate_drawlists.push_back(drawlist);
        }
        final_framebuffer = framebuffer;
        final_width = state.width;
        final_height = state.height;

        cleanup_attempted = true;
        ReleaseFunctionalPayloads(pool_, state);
        pool_.Release(txn.state);
        if (completed + 1U < options_.driver_commands.size()) {
          if (!sequence_completion_) {
            throw std::runtime_error(
                "JsonReporter PCO sequence completion event is missing");
          }
          sequence_completion_->notify(sc_core::SC_ZERO_TIME);
        }
      } catch (const std::exception &error) {
        if (!cleanup_attempted) {
          try {
            if (state_loaded)
              ReleaseFunctionalPayloads(pool_, state);
            pool_.Release(txn.state);
          } catch (const std::exception &) {
            // Preserve the first pipeline/evidence failure.
          }
        }
        failed_ = true;
        EmitError(static_cast<std::uint32_t>(completed + 1U), error.what());
        sc_core::sc_stop();
        return;
      }
    }

    try {
      if (final_width != options_.width || final_height != options_.height ||
          final_framebuffer.size() !=
              static_cast<std::uint64_t>(final_width) * final_height * 4U) {
        throw std::runtime_error(
            "JsonReporter PCO sequence final framebuffer is invalid");
      }
      aggregate.frame = 1;
      aggregate.functional_frame = 1;
      aggregate.pool_bytes_in_flight = pool_.bytes_in_flight();
      aggregate.pool_high_water_bytes = pool_.high_water_bytes();
      NormalizeDriverPcoTrianglesApiCounters(options_, aggregate);
      // A generic native sequence may collapse several captured API
      // DrawLists into one physical PCO draw. Partition each physical
      // execution record across its contiguous logical range without changing
      // any aggregate ISS instruction count; the protocol therefore retains
      // exactly the captured DrawList count and the complete native evidence.
      if (generic_sequence) {
        ExpandGenericPcoSequenceDrawLists(options_.driver_command.draw_count,
                                          &aggregate_drawlists);
      }
      ValidateDrawListStats(aggregate, aggregate_drawlists);

      std::filesystem::path artifact_path;
      if (!options_.output_dir.empty()) {
        artifact_path = FramePath(options_, 1);
        // Ordered native PCO sequences publish only the final physical DRAM
        // readback.  No command sidecar/golden/CPU framebuffer is consulted.
        WriteRgbaPngAtomic(artifact_path, final_framebuffer, final_width,
                           final_height);
      }
      EmitCounter(options_, aggregate, aggregate_drawlists, vertex_pco,
                  fragment_pco, artifact_path);
      if (!artifact_path.empty()) {
        std::cout << "@CAPTURE: " << options_.test_case
                  << " sample=1 png=" << artifact_path.filename().string()
                  << '\n';
        std::cout.flush();
      }
    } catch (const std::exception &error) {
      failed_ = true;
      EmitError(1, error.what());
      sc_core::sc_stop();
      return;
    }

    const std::uint64_t allocations = pool_.allocations();
    const std::uint64_t releases = pool_.releases();
    const std::uint64_t leaks =
        allocations >= releases ? allocations - releases : 0;
    std::cout << "{\"protocol\":\"pvrgpu-jsonl\",\"version\":1"
              << ",\"schema\":\"" << kSchema << "\",\"type\":\"done\""
              << ",\"backend\":\"pvrgpu\",\"frames\":1"
              << ",\"physical_submissions\":"
              << options_.driver_commands.size()
              << ",\"pool_allocations\":" << allocations
              << ",\"pool_releases\":" << releases
              << ",\"pool_leaks\":" << leaks
              << ",\"pool_bytes_in_flight\":" << pool_.bytes_in_flight()
              << "}\n";
    std::cout.flush();
    if (leaks != 0 || pool_.bytes_in_flight() != 0)
      failed_ = true;
    sc_core::sc_stop();
    return;
  }

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
        std::vector<std::uint8_t> artifact_framebuffer = framebuffer;
        LoadDriverFramebufferSnapshot(options_,
                                      state.width,
                                      state.height,
                                      &artifact_framebuffer);
        WriteRgbaPngAtomic(artifact_path, artifact_framebuffer, state.width,
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
      } else if (IsDriverIndexedQuadCounterView(options_)) {
        NormalizeDriverIndexedQuadApiCounters(options_, counters);
        ScaleDriverIndexedQuadShaderCounters(options_, counters,
                                             emitted_drawlists);
      } else if (IsDriverPcoTrianglesCounterView(options_)) {
        NormalizeDriverPcoTrianglesApiCounters(options_, counters);
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
