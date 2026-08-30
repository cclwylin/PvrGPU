/*
 * PowerVR PCO instruction-set simulator (ISS) public interface.
 *
 * PCO is the name used by Mesa's public PowerVR shader compiler backend; no
 * expansion of that name is asserted here.  ISS means Instruction Set
 * Simulator, and USC means Unified Shading Cluster.  This file defines the
 * strict decoded form passed from the PCO decoder to the USC execution model.
 * The decoded summary and each instruction are trivially copyable so they can
 * be stored in the model's MemoryPool while a FIFO carries only its handle.
 */
#pragma once

#include "common/shader_stage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace pvrgpu::stub {

inline constexpr std::size_t kPcoVertexInputCount = 32;
inline constexpr std::size_t kPcoVertexOutputCount = 64;
inline constexpr std::size_t kPcoPixelOutputCount = 4;
inline constexpr std::size_t kPcoTemporaryCount = 32;
inline constexpr std::size_t kPcoFillTexNearestCoefficientCount = 12;
inline constexpr std::size_t kPcoFillTexNearestVertexSharedCount = 1;
inline constexpr std::size_t kPcoFillTexNearestFragmentSharedCount = 20;
inline constexpr std::size_t kPcoVaryingOneCoefficientCount = 20;
inline constexpr std::size_t kPcoVaryingTwoCoefficientCount = 36;
inline constexpr std::size_t kPcoVaryingFourCoefficientCount = 68;
inline constexpr std::size_t kPcoVaryingEightCoefficientCount = 132;
inline constexpr std::size_t kPcoMaximumVaryingCoefficientCount =
    kPcoVaryingEightCoefficientCount;

/* Public PCO register-bank encodings used by the supported instructions. */
enum class PcoRegisterBank : std::uint8_t {
  kSpecial = 0,
  kTemporary = 1,
  kVertexInput = 2,
  kCoefficient = 3,
  kShared = 4,
  kCoefficientAlternate = 5,
  kIndex0 = 6,
  kIndex1 = 7,
};

struct PcoRegisterRef {
  PcoRegisterBank bank = PcoRegisterBank::kSpecial;
  std::uint16_t index = 0;
};

/* Exact public PCO operations implemented by the first functional subset. */
enum class PcoOpcode : std::uint8_t {
  kMoveBypass,
  kFloatAdd,
  kFloatMultiply,
  kFloatMad,
  kFloatMin,
  kFloatMax,
  kReciprocal,
  kReciprocalSquareRoot,
  kFloatLog2,
  kFloatExp2,
  kBranch,
  kBranchConditional,
  kLoopBegin,
  kLoopEnd,
  kTextureSample,
  kTextureSampleLod,
  kIntegerAdd,
  kBitwiseAnd,
  kBitwiseOr,
  kBitwiseXor,
  kFloatSine,
  kFloatCosine,
  kBufferLoad,
  kBufferStore,
  kDiscard,
  kDerivativeX,
  kDerivativeY,
  kPackHalf2x16,
  kUnpackHalf2x16,
  kTextureGather,
  kAtomicAdd,
  kAtomicCompSwap,
  kFloatInterpolatePerspective,
  kWaitDataFence,
  kUvsWrite,
  kUvsWriteEmitEndTask,
  kUvsEmitEndTask,
};

enum class PcoWriteTarget : std::uint8_t {
  kNone,
  kPixelOutput,
  kVertexOutput,
  kTemporary,
};

enum class PcoIterationMode : std::uint8_t {
  kPixel,
  kSample,
  kCentroid,
};

// Stable performance-counter classes. Memory includes backend input/output
// export operations such as UVSW; it is an instruction count, not byte traffic.
enum class PcoInstructionClass : std::uint8_t {
  kAlu,
  kTexture,
  kMemory,
};

struct PcoInstructionCounts {
  std::uint64_t alu = 0;
  std::uint64_t texture = 0;
  std::uint64_t memory = 0;
};

/*
 * A canonical, serializable instruction produced from a real PCO group.
 * repeat_count applies the PCO group repeat to both source and output index.
 */
struct PcoInstruction {
  PcoOpcode opcode = PcoOpcode::kMoveBypass;
  PcoWriteTarget target = PcoWriteTarget::kNone;
  PcoRegisterRef source{};
  // source1 is present only when source_count is two. Keeping both operands in
  // the serialized semantic instruction lets FADD remain a real two-source
  // USC operation rather than a fixture-specific precomputed value.
  PcoRegisterRef source1{};
  // source2 is present when source_count is three (e.g. FMAD a * b + c).
  PcoRegisterRef source2{};
  std::uint32_t binary_offset = 0;
  std::uint16_t group_index = 0;
  std::uint16_t output_index = 0;
  std::uint16_t branch_target_index = 0;
  std::uint32_t loop_count = 0;
  // FITRP uses component_count independently of the instruction-group repeat
  // field: the exact gate-12 instruction interpolates four coefficient sets
  // in one group, while its encoded group repeat remains one.
  std::uint8_t component_count = 1;
  std::uint8_t data_request = 0;
  PcoIterationMode iteration_mode = PcoIterationMode::kPixel;
  std::uint8_t perspective = 0;
  std::uint8_t saturate = 0;
  std::uint8_t source_count = 1;
  std::uint8_t repeat_count = 1;
  std::uint8_t end_group = 0;
};

/* Stored directly in PipelineState; no owning container appears here. */
struct PcoProgramSummary {
  ShaderStage stage = ShaderStage::kVertex;
  std::uint32_t binary_size = 0;
  std::uint32_t group_count = 0;
  std::uint32_t instruction_count = 0;
  // Vertex-input registers read by the decoded program. The decoder checks
  // this mask against the input-assembler attribute-to-register contract.
  std::uint32_t vertex_input_mask = 0;
  std::uint64_t vertex_output_mask = 0;
  std::uint8_t pixel_output_mask = 0;
  std::uint8_t early_hsr_safe = 0;
  std::uint8_t ends_task = 0;
};

/* Convenient decode return value; only its two members are stored separately.
 */
struct PcoDecodedProgram {
  PcoProgramSummary summary{};
  std::vector<PcoInstruction> instructions;
};

struct PcoVertexExecution {
  std::array<std::uint32_t, kPcoVertexOutputCount> outputs{};
  std::uint64_t written_mask = 0;
  std::uint8_t emitted = 0;
  std::uint8_t ended_task = 0;
};

/* Lane-local shared-register input used by the exact fill_tex_nearest VS.
 * SH0 is the raw IEEE-754 binary32 `scale` uniform supplied by the PDS/USC
 * shared-register ABI; the official GLBench draw writes 1.0f.
 */
struct PcoVertexExecutionContext {
  std::array<std::uint32_t, kPcoFillTexNearestVertexSharedCount>
      shared_registers{};
  std::uint8_t shared_count = 0;
};

/* One exact public SMP.2D.FCNORM request emitted by the fragment ISS. The
 * normalized coordinates and hardware texture/sampler state are the values
 * read by the decoded USC instruction, not a precomputed texel or case name.
 */
struct PcoTextureRequest {
  std::array<std::uint32_t, 2> coordinates{};
  std::array<std::uint32_t, 4> texture_state{};
  std::array<std::uint32_t, 4> sampler_state{};
  std::uint8_t coordinate_count = 0;
  std::uint8_t component_count = 0;
  std::uint8_t descriptor_set = 0;
  std::uint8_t binding = 0;
  std::uint8_t dimension = 0;
  std::uint8_t normalized = 0;
  std::uint8_t data_request = 0;
};

/* Lane-local USC state captured at the public SMP suspension point.  The
 * response path resumes at the following WDF; it must not replay FITRP, the
 * shared-register moves, or the SMP instruction.  This POD can therefore be
 * carried by a MemoryPool payload while a SystemC FIFO carries only a handle.
 */
struct PcoFragmentContinuation {
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  std::uint32_t temporary_written_mask = 0;
  std::uint32_t program_binary_size = 0;
  std::uint32_t program_instruction_count = 0;
  std::uint16_t resume_instruction_index = 0;
  std::uint16_t pending_output_index = 0;
  std::uint8_t pending_component_count = 0;
  std::uint8_t data_request = 0;
  std::uint8_t valid = 0;
};

struct PcoFragmentExecution {
  std::array<std::uint32_t, kPcoPixelOutputCount> pixel_outputs{};
  PcoTextureRequest texture_request{};
  PcoFragmentContinuation continuation{};
  std::uint32_t executed_instruction_count = 0;
  std::uint8_t written_mask = 0;
  std::uint8_t texture_request_valid = 0;
  std::uint8_t suspended = 0;
  bool discarded = false;
};

/*
 * Lane-local fragment input for the public FITRP.PIXEL subset. Coefficients
 * are raw IEEE-754 binary32 USC coefficient registers ordered as Rogue
 * A/B/C/PAD sets: position-W, then RGBA for each linked smooth varying.
 * varyings_shader_1 uses 20 dwords, varyings_shader_2 uses 36 and
 * varyings_shader_4 uses 68 and varyings_shader_8 uses 132. sample_x and
 * sample_y are raw binary32 framebuffer sample-center coordinates.
 * coefficient_count is explicit so a truncated MemoryPool span fails closed
 * instead of reading zero fill.
 */
struct PcoFragmentExecutionContext {
  std::array<std::uint32_t, kPcoMaximumVaryingCoefficientCount>
      coefficients{};
  std::uint32_t sample_x = 0;
  std::uint32_t sample_y = 0;
  std::array<std::uint32_t, kPcoFillTexNearestFragmentSharedCount>
      shared_registers{};
  std::array<std::uint32_t, kPcoPixelOutputCount> texture_response{};
  PcoFragmentContinuation continuation{};
  std::uint8_t coefficient_count = 0;
  std::uint8_t shared_count = 0;
  std::uint8_t texture_response_valid = 0;
};

/* Immutable raw PCO binaries generated by the identified public Mesa backend.
 */
const std::vector<std::uint8_t> &FillSolidVertexPcoBinary();
const std::vector<std::uint8_t> &FillSolidFragmentPcoBinary();
const std::vector<std::uint8_t> &FillSolidRedHalfAlphaFragmentPcoBinary();
const std::vector<std::uint8_t> &FillSolidGreenHalfAlphaFragmentPcoBinary();
const std::vector<std::uint8_t> &TriangleSetupOrangeFragmentPcoBinary();
const std::vector<std::uint8_t> &TriangleSetupCyanFragmentPcoBinary();
const std::vector<std::uint8_t> &AttributeFetchVertexPcoBinary();
const std::vector<std::uint8_t> &AttributeFetchTwoAttributeVertexPcoBinary();
const std::vector<std::uint8_t> &AttributeFetchFourAttributeVertexPcoBinary();
const std::vector<std::uint8_t> &AttributeFetchEightAttributeVertexPcoBinary();
const std::vector<std::uint8_t> &AttributeFetchGrayFragmentPcoBinary();
const std::vector<std::uint8_t> &VaryingsOneVertexPcoBinary();
const std::vector<std::uint8_t> &VaryingsOneFragmentPcoBinary();
const std::vector<std::uint8_t> &VaryingsTwoVertexPcoBinary();
const std::vector<std::uint8_t> &VaryingsTwoFragmentPcoBinary();
const std::vector<std::uint8_t> &VaryingsFourVertexPcoBinary();
const std::vector<std::uint8_t> &VaryingsFourFragmentPcoBinary();
const std::vector<std::uint8_t> &VaryingsEightVertexPcoBinary();
const std::vector<std::uint8_t> &VaryingsEightFragmentPcoBinary();
const std::vector<std::uint8_t> &FillTexNearestVertexPcoBinary();
const std::vector<std::uint8_t> &FillTexNearestFragmentPcoBinary();

// Counts semantic groups when expand_repeats is false, or repeat-expanded
// operations executed by one shader invocation when it is true.
PcoInstructionCounts
CountPcoInstructions(const std::vector<PcoInstruction> &instructions,
                     bool expand_repeats);

/* Decode real PCO bytes and reject every encoding outside the exact subset. */
PcoDecodedProgram DecodePcoProgram(ShaderStage stage,
                                   const std::vector<std::uint8_t> &binary);

/* Execute raw 32-bit USC register values without host floating-point changes.
 */
PcoVertexExecution
ExecuteVertexPco(const PcoProgramSummary &summary,
                 const std::vector<PcoInstruction> &instructions,
                 const std::vector<std::uint32_t> &vertex_inputs);
PcoVertexExecution ExecuteVertexPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const std::vector<std::uint32_t> &vertex_inputs,
    const PcoVertexExecutionContext &context);

PcoFragmentExecution
ExecuteFragmentPco(const PcoProgramSummary &summary,
                   const std::vector<PcoInstruction> &instructions);
PcoFragmentExecution ExecuteFragmentPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoFragmentExecutionContext &context);
PcoFragmentExecution ResumeFragmentPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoFragmentContinuation &continuation,
    const std::array<std::uint32_t, kPcoPixelOutputCount> &texture_response);

/* Short ISS-facing names used by the decoder/USC pipeline. */
inline PcoDecodedProgram Decode(ShaderStage stage,
                                const std::vector<std::uint8_t> &binary) {
  return DecodePcoProgram(stage, binary);
}

inline PcoVertexExecution
ExecuteVertex(const PcoProgramSummary &summary,
              const std::vector<PcoInstruction> &instructions,
              const std::vector<std::uint32_t> &vertex_inputs) {
  return ExecuteVertexPco(summary, instructions, vertex_inputs);
}

inline PcoVertexExecution ExecuteVertex(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const std::vector<std::uint32_t> &vertex_inputs,
    const PcoVertexExecutionContext &context) {
  return ExecuteVertexPco(summary, instructions, vertex_inputs, context);
}

inline PcoFragmentExecution
ExecuteFragment(const PcoProgramSummary &summary,
                const std::vector<PcoInstruction> &instructions) {
  return ExecuteFragmentPco(summary, instructions);
}

inline PcoFragmentExecution ExecuteFragment(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoFragmentExecutionContext &context) {
  return ExecuteFragmentPco(summary, instructions, context);
}

inline PcoFragmentExecution ResumeFragment(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoFragmentContinuation &continuation,
    const std::array<std::uint32_t, kPcoPixelOutputCount> &texture_response) {
  return ResumeFragmentPco(summary, instructions, continuation,
                           texture_response);
}

static_assert(std::is_trivially_copyable_v<PcoRegisterRef>);
static_assert(std::is_trivially_copyable_v<PcoInstruction>);
static_assert(std::is_trivially_copyable_v<PcoInstructionCounts>);
static_assert(std::is_trivially_copyable_v<PcoProgramSummary>);
static_assert(std::is_trivially_copyable_v<PcoVertexExecution>);
static_assert(std::is_trivially_copyable_v<PcoVertexExecutionContext>);
static_assert(std::is_trivially_copyable_v<PcoTextureRequest>);
static_assert(std::is_trivially_copyable_v<PcoFragmentContinuation>);
static_assert(std::is_trivially_copyable_v<PcoFragmentExecution>);
static_assert(std::is_trivially_copyable_v<PcoFragmentExecutionContext>);

} // namespace pvrgpu::stub
