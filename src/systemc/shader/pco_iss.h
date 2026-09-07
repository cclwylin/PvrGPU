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
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace pvrgpu::stub {

/* VTXIN0..63.  Each bound attribute occupies four registers, so this is the
 * sixteen attributes a generically lowered draw may bind.  Like the temporary
 * file below, it is an explicit ABI bound rather than a Rogue hardware limit. */
inline constexpr std::size_t kPcoVertexInputCount = 64;
inline constexpr std::size_t kPcoVertexOutputCount = 64;
/* A sampler's response is four components, one per channel. */
inline constexpr std::size_t kPcoTextureResponseCount = 4;
/* Mesa I_LD_IMMBL uses a four-bit positive wrapped burst length: 1..15
 * encode themselves and zero encodes 16 DWORDs. This is not the SMP width. */
inline constexpr std::size_t kPcoMaximumBufferLoadDwords = 16;
/*
 * The driver's four-render-target ABI provides four dwords per attachment.
 * Mesa pco_map lowers PIXOUT indices below four to special 32 + index and
 * all later indices to special 164 + index - 4.  The public ISA names only
 * PIXOUT0..7; extending that lowering to PIXOUT15 is a model ABI bound, not
 * a claim about the number of physical pixel-output registers on Rogue.
 */
inline constexpr std::size_t kPcoPixelOutputCount = 16;
/* The driver command ABI transports up to 256 temporary registers. This is
 * an explicit model bound, not the larger index limit of the public ISA. */
inline constexpr std::size_t kPcoTemporaryCount = 256;

/* Lane/continuation ownership covers the full TEMP file, without shifting a
 * 64-bit integer by a register index >=64. Keep this payload trivially
 * copyable for the MemoryPool; it never owns memory or a host callback. */
struct PcoTemporaryMask {
  std::array<std::uint64_t, kPcoTemporaryCount / 64> words{};

  bool test(std::size_t index) const {
    return index < kPcoTemporaryCount &&
           (words[index / 64] & (UINT64_C(1) << (index % 64))) != 0;
  }
  void set(std::size_t index) {
    if (index >= kPcoTemporaryCount)
      throw std::out_of_range("PCO TEMP mask index exceeds the modeled file");
    words[index / 64] |= UINT64_C(1) << (index % 64);
  }
  bool contains_range(std::size_t first, std::size_t count) const {
    if (first > kPcoTemporaryCount || count > kPcoTemporaryCount - first)
      return false;
    for (std::size_t index = first; index < first + count; ++index) {
      if (!test(index))
        return false;
    }
    return true;
  }
  bool operator==(const PcoTemporaryMask &other) const {
    return words == other.words;
  }
  bool operator!=(const PcoTemporaryMask &other) const {
    return !(*this == other);
  }
};
static_assert(std::is_trivially_copyable_v<PcoTemporaryMask>);
/* A combined image/sampler descriptor occupies one 20-dword public PDS/USC
 * shared-register slot.  Descriptor count, sequential SMP count, and the
 * transported shared-register span are independent bounds.  The captured
 * terrain main draw transports 56 VS push DWORDs plus two descriptors (96
 * total), and 64 FS push DWORDs plus five descriptors (164 total).  GFXBench
 * Manhattan samples six textures in one fragment stage, so the descriptor
 * bound is eight and the fragment transport 256 DWORDs.  These are current
 * public workload/transport gates, not Rogue hardware-file limits. */
inline constexpr std::size_t kPcoTextureDescriptorDwordCount = 20;
inline constexpr std::size_t kPcoMaximumTextureDescriptorSets = 8;
/* Sequential SMP instructions are not descriptor sets.  The captured
 * terrain post-process shaders issue as many as nine samples from one set,
 * so continuation depth has its own strict program bound. */
inline constexpr std::size_t kPcoMaximumTextureSampleInstructions = 9;
inline constexpr std::size_t kPcoMaximumVertexSharedCount = 96;
inline constexpr std::size_t kPcoMaximumFragmentSharedCount = 256;
inline constexpr std::size_t kPcoMaximumSharedCount =
    kPcoMaximumFragmentSharedCount;
inline constexpr std::size_t kPcoConditionalsVertexSharedCount = 16;
inline constexpr std::size_t kPcoConditionalsFragmentSharedCount = 4;
inline constexpr std::size_t kPcoFillTexNearestCoefficientCount = 12;
inline constexpr std::size_t kPcoFillTexNearestVertexSharedCount = 1;
inline constexpr std::size_t kPcoFillTexNearestFragmentSharedCount = 20;
inline constexpr std::size_t kPcoVaryingOneCoefficientCount = 20;
inline constexpr std::size_t kPcoVaryingTwoCoefficientCount = 36;
inline constexpr std::size_t kPcoVaryingFourCoefficientCount = 68;
inline constexpr std::size_t kPcoVaryingEightCoefficientCount = 132;
/* Coefficient registers the fragment file holds: four for position plus four
 * per smooth-varying component.  Sized for the sixty-four components a
 * generically lowered draw may pass, not for the eight-varying capture that
 * named the constants above. */
inline constexpr std::size_t kPcoMaximumVaryingCoefficientCount = 260;

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

/* Public PCO semantic operations implemented by the functional ISS subset. */
enum class PcoOpcode : std::uint8_t {
  // Semantic operations emitted by the strict public conditionals profile.
  // A compiler group can contain several internal phase operations; the
  // decoded operation records their externally visible register effect.
  kInternal,
  kMoveBypass,
  kFloatNegate,
  kFloatAbs,
  kMoveImmediate,
  kPackCoverageMask,
  kFloatFloor,
  kFloatSubtract,
  kFloatGreaterEqual,
  kFloatEqual,
  kFloatLess,
  // BCMP with any other PCO TST operation/type pair: the operation and the
  // operand type it compares as travel on the instruction rather than in the
  // opcode, so a new comparison the compiler emits costs no new opcode.
  kBooleanCompare,
  kConditionalSelect,
  kConditionalSelectNegateTrue,
  // TST + MOVC with any other unary test operation/operand type: like
  // kBooleanCompare, the test travels on the instruction.
  kTestConditionalSelect,
  kFloatAdd,
  kFloatAddNegateSource0,
  kFloatMultiply,
  kFloatMad,
  kFloatMadNegateSource2,
  kFloatMadNegateSource0,
  kFloatMadNegateSource0Source2,
  kFloatMin,
  kFloatMax,
  kIntegerMaxSigned,
  kIntegerMinSigned,
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
  kIntegerMultiplyAdd32,
  kBitfieldInsert,
  kBitfieldExtractUnsigned,
  // The same three-phase group with an arithmetic shift in phase 2, which is
  // what GL's signed bitfieldExtract lowers to: the extracted field keeps its
  // sign.  A narrow signed vertex attribute read as an int reaches the shader
  // through one of these per component.
  kBitfieldExtractSigned,
  kIntegerAdd64_32,
  kBitwiseAnd,
  kBitwiseOr,
  kBitwiseXor,
  kBitwiseXnor,
  kShiftRight,
  kShiftLeft,
  kTestZero,
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
  /* Scalar PCO PCK/UNPCK conversion operations.  The F16F16 forms are
   * distinct from the GLSL packHalf2x16/unpackHalf2x16 vector operations
   * above: only the low 16-bit lane is transferred and one temporary is
   * written.  U32/S32 forms convert the whole word. */
  kFloatPackHalfRtne,
  kFloatPackHalfRtz,
  kFloatToUint32Rtne,
  kFloatToUint32Rtz,
  kFloatToInt32Rtne,
  kFloatToInt32Rtz,
  kFloatUnpackHalf,
  /* UNPCK.U32 / UNPCK.S32: the packed source is a whole 32-bit integer, so
   * the unpack is the integer-to-float conversion GLSL spells uint(x) and
   * int(x).  Named for the format the encoding selects. */
  kUnpackUnsignedToFloat,
  kUnpackSignedToFloat,
  /* UNPCK of a packed vector format -- 8888, 1616 or 1010102, signed or not,
   * with or without the normalizing scale.  One source word yields one
   * component per group repeat, which is how a packed vertex attribute
   * reaches the shader. */
  kUnpackVector,
  /* Mesa fcsel_gt lowered as TST.F32.GZ + MOVC.  Keep this distinct from
   * Boolean BCSEL: the condition is an ordered float comparison with +0. */
  kConditionalSelectGreaterZero,
  kDepthFeedback,
};

enum class PcoWriteTarget : std::uint8_t {
  kNone,
  kPixelOutput,
  kVertexOutput,
  kTemporary,
  // PCO register allocation reuses dead vertex-input registers for ALU values.
  kVertexInput,
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
/*
 * One phase of an instruction group that computes a value into an internal
 * result register.  PCO's select-shaped groups -- bcsel, fceil, fsign, isign
 * -- all have the same layout: phase 0 and phase 1 each run a main-ALU
 * operation into ft0 and ft1, phase 2 tests and then MOVCs between them.
 * Modelling one group as one ALU instruction could describe only the case
 * where both phases are plain moves, which is bcsel; ceil adds one to a
 * floor in phase 0 and takes the floor itself in phase 1.
 */
struct PcoPhaseOperation {
  PcoOpcode opcode = PcoOpcode::kMoveBypass;
  PcoRegisterRef source{};
  PcoRegisterRef source1{};
  PcoRegisterRef source2{};
  std::uint8_t source_count = 1;
  std::uint8_t source0_floor = 0;
  std::uint8_t source0_absolute = 0;
  std::uint8_t source1_absolute = 0;
  std::uint8_t source2_absolute = 0;
  std::uint8_t source2_floor = 0;
  std::uint8_t saturate = 0;
  /*
   * A 64-bit multiply-add produces two results: its low word into the phase's
   * own internal register and its high word into the feed-through one.  That
   * is how the integer sign is built -- the low word is the value tested and
   * the high word is its sign extension, which is the answer for everything
   * that is not positive.
   */
  std::uint8_t produces_feed_through = 0;
  std::uint8_t integer_signed = 0;
};

/* Which internal result a phase-2 MOVC moves when its test passes. */
enum class PcoInternalResult : std::uint8_t {
  kPhase0 = 0,
  kPhase1 = 1,
  kPhase2 = 2,
  kFeedThrough = 3,
};

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
  // source3 is present only for the four-source bitfield insert.
  PcoRegisterRef source3{};
  std::uint32_t binary_offset = 0;
  std::uint16_t group_index = 0;
  std::uint16_t output_index = 0;
  // The high half's destination for a two-output op (add64_32).
  std::uint16_t output_index1 = 0;
  // ADD64_32 sign-extends its 32-bit offset when the ISA S bit is set.
  std::uint8_t address_offset_signed = 0;
  std::uint16_t branch_target_index = 0;
  std::uint32_t loop_count = 0;
  // Raw binary32/integer payload for compiler-emitted immediate groups.
  std::uint32_t immediate = 0;
  // FITRP uses component_count independently of the instruction-group repeat
  // field: one group interpolates one through four coefficient sets while its
  // encoded group repeat remains one.
  std::uint8_t component_count = 1;
  // SMP's dmn field: 1d=0b01, 2d=0b10, 3d=0b11.  It is also the number of
  // coordinate temporaries the instruction reads, which is why a 2D-array,
  // cube or 3D sample needs three where a 2D sample needs two.
  std::uint8_t texture_dimension = 2;
  // SMP `.tao`: the sample takes its texture base from a shader-computed
  // 64-bit address (array layer folded in) at coordinate_base+3/+4.
  std::uint8_t texture_address_offset = 0;
  // SMP FCNORM converts the sampled channels to floating-point results.
  // Mesa leaves it clear for integer samplers; it does not control whether
  // the texture coordinates are normalized.
  std::uint8_t texture_fcnorm = 1;
  std::uint8_t data_request = 0;
  PcoIterationMode iteration_mode = PcoIterationMode::kPixel;
  std::uint8_t perspective = 0;
  std::uint8_t saturate = 0;
  // Public scalar ALU groups encode FLR as a modifier on source 0.  Keep it
  // orthogonal to the base opcode so FADD and FMUL share the same exact
  // decode/validation/execution contract.
  std::uint8_t source0_floor = 0;
  // Source ABS is one bit per operand in the I_MAIN encoding byte, and
  // pco_ops declares both on FADD and on FMUL, so they are carried
  // independently of each other and of the opcode.
  std::uint8_t source0_absolute = 0;
  std::uint8_t source1_absolute = 0;
  // I_FMAD_EXT adds source-2 floor/absolute before its compact negate bit.
  std::uint8_t source2_floor = 0;
  std::uint8_t source2_absolute = 0;
  // BCMP normally materializes canonical Boolean bits (all ones or zero).
  // Mesa's PCK.ONE form instead materializes binary32 1.0 or 0.0; retain the
  // distinction without inventing a second comparison opcode/histogram bin.
  std::uint8_t comparison_result_float_one = 0;
  // PCO F_TST_OP / F_TST_TYPE as the TST phase encodes them, for
  // kBooleanCompare.  The three float forms that predate this keep their own
  // opcodes, so these stay at the f32-equal encoding for every other opcode.
  std::uint8_t comparison_test_op = 0;
  std::uint8_t comparison_test_type = 0;
  /* IMADD32's source modifiers, one bit each in the extended byte: a
   * two's-complement absolute and then a negate, applied in that order to
   * each operand.  `-a * b + c` is an s0 negate, and GL's integer abs() is
   * an s0 absolute against a multiply by one. */
  std::uint8_t source0_integer_negate = 0;
  std::uint8_t source0_integer_absolute = 0;
  std::uint8_t source1_integer_negate = 0;
  std::uint8_t source1_integer_absolute = 0;
  std::uint8_t source2_integer_absolute = 0;
  /* BCMP moves both comparison operands through phase-0/phase-1 MBYPs, and
   * each MBYP may negate its source after taking its absolute value.  The
   * same source-1 flag is also used by TST/MOVC's true-value MBYP, which is
   * how the compiler spells `cond ? -a : b`. */
  std::uint8_t source0_negate = 0;
  std::uint8_t source1_negate = 0;
  /* Which way the MOVC that follows a TST reads the predicate.  In the
   * TST/MOVC select form a passing test takes the value phase 0 supplies;
   * in the BCSEL group it takes internal source 4 instead, so a passing test
   * there selects the operand the other form calls the false one. */
  std::uint8_t conditional_select_inverted = 0;
  /*
   * The two computing phases of a select-shaped group, and how phase 2 reads
   * them.  `phase_composed` says the group has them at all: without it the
   * select's operands are plain register sources, which is what a bcsel is.
   * `select_true_result` is the MOVC's movw0 -- the internal result it moves
   * when the test passes -- and `select_false_result` is its is4 selector.
   * `test_source1_result` is the ISS is2 selector, which is what a binary
   * test compares the fed-through operand against.
   */
  std::uint8_t phase_composed = 0;
  PcoPhaseOperation phase0{};
  PcoPhaseOperation phase1{};
  PcoInternalResult select_true_result = PcoInternalResult::kPhase1;
  PcoInternalResult select_false_result = PcoInternalResult::kPhase0;
  PcoInternalResult test_source0_result = PcoInternalResult::kFeedThrough;
  PcoInternalResult test_source1_result = PcoInternalResult::kPhase1;
  /* PCO's F_PCK_FORMAT for kUnpackVector, and its `scale` bit: scaling
   * normalizes the field to [0,1] or [-1,1] instead of yielding its integer
   * value as a float. */
  std::uint8_t unpack_format = 0;
  std::uint8_t unpack_scale = 0;
  /* The bitwise phase-0 shift1 operation for kBitfieldInsert.  MSK.LSL
   * shifts the inserted value into place, which is what GL's bitfieldInsert
   * needs; MSK alone leaves it where it is, which is how fcopysign is built
   * -- a full-width mask at offset zero selecting magnitude from one operand
   * and sign from the other. */
  std::uint8_t bitfield_insert_shifts = 1;
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
  std::uint64_t vertex_input_mask = 0;
  std::uint64_t vertex_output_mask = 0;
  std::uint16_t pixel_output_mask = 0;
  std::uint8_t early_hsr_safe = 0;
  std::uint8_t writes_depth = 0;
  std::uint8_t ends_task = 0;
};

/* Convenient decode return value; only its two members are stored separately.
 */
struct PcoDecodedProgram {
  PcoProgramSummary summary{};
  std::vector<PcoInstruction> instructions;
};

/* One decoded public SMP request emitted by either shader-stage ISS. The
 * normalized coordinates and hardware texture/sampler state are the values
 * read by the decoded USC instruction, not a precomputed texel or case name.
 */
struct PcoTextureRequest {
  // Three normalized coordinates: s, t and (for a 3D image) the depth r.
  std::array<std::uint32_t, 3> coordinates{};
  std::array<std::uint32_t, 4> texture_state{};
  std::array<std::uint32_t, 4> sampler_state{};
  // The `.tao` sample's shader-computed 64-bit texture base address.
  std::uint32_t texture_address_lo = 0;
  std::uint32_t texture_address_hi = 0;
  std::uint8_t coordinate_count = 0;
  std::uint8_t component_count = 0;
  std::uint8_t descriptor_set = 0;
  std::uint8_t binding = 0;
  std::uint8_t dimension = 0;
  std::uint8_t normalized = 0;
  std::uint8_t fcnorm = 1;
  std::uint8_t data_request = 0;
};

/* Complete lane-local vertex state captured immediately after an SMP request.
 * Vertex sampling resumes at the following WDF with one four-DWORD response.
 * Inputs and shared registers are owned here because the resumed lane may run
 * after its original PDS/USC work item has left the execution call stack.
 */
struct PcoVertexContinuation {
  std::array<std::uint32_t, kPcoVertexInputCount> vertex_inputs{};
  std::array<std::uint32_t, kPcoMaximumVertexSharedCount> shared_registers{};
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  std::array<std::uint32_t, kPcoVertexOutputCount> outputs{};
  PcoTemporaryMask temporary_written_mask{};
  std::uint64_t output_written_mask = 0;
  std::uint32_t program_binary_size = 0;
  std::uint32_t program_instruction_count = 0;
  std::uint16_t resume_instruction_index = 0;
  std::uint16_t pending_output_index = 0;
  std::uint8_t pending_component_count = 0;
  std::uint8_t data_request = 0;
  std::uint8_t vertex_input_count = 0;
  std::uint16_t shared_count = 0;
  std::uint8_t emitted = 0;
  std::uint8_t ended_task = 0;
  std::uint8_t valid = 0;
};

struct PcoVertexExecution {
  std::array<std::uint32_t, kPcoVertexOutputCount> outputs{};
  PcoTextureRequest texture_request{};
  PcoVertexContinuation continuation{};
  std::uint64_t written_mask = 0;
  std::uint32_t executed_instruction_count = 0;
  std::uint8_t emitted = 0;
  std::uint8_t ended_task = 0;
  std::uint8_t texture_request_valid = 0;
  std::uint8_t suspended = 0;
};

/* Stack-only bridge to the modeled memory system. No host pointers are saved
 * in a shader continuation or a MemoryPool payload. The caller validates the
 * complete address range and may throw on an invalid request. */
using PcoMemoryReadCallback = void (*)(void *user_data, std::uint64_t address,
                                      std::uint32_t dword_count,
                                      std::uint32_t *destination);

/* Lane-local shared-register input supplied by the PDS/USC ABI. The explicit
 * shared_count keeps every access outside the producer-declared transport
 * span fail-closed.  texture_response is accepted only with a valid saved
 * continuation.
 */
struct PcoVertexExecutionContext {
  std::array<std::uint32_t, kPcoMaximumSharedCount> shared_registers{};
  std::array<std::uint32_t, kPcoTextureResponseCount> texture_response{};
  PcoVertexContinuation continuation{};
  std::uint16_t shared_count = 0;
  std::uint8_t texture_response_valid = 0;
  PcoMemoryReadCallback memory_read = nullptr;
  void *memory_user_data = nullptr;
};

/* Lane-local USC state captured at the public SMP suspension point.  The
 * response path resumes at the following WDF; it must not replay FITRP, the
 * shared-register moves, or the SMP instruction.  This POD can therefore be
 * carried by a MemoryPool payload while a SystemC FIFO carries only a handle.
 */
struct PcoFragmentContinuation {
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  PcoTemporaryMask temporary_written_mask{};
  std::uint32_t program_binary_size = 0;
  std::uint32_t program_instruction_count = 0;
  std::uint16_t resume_instruction_index = 0;
  std::uint16_t pending_output_index = 0;
  std::uint8_t pending_component_count = 0;
  std::uint8_t data_request = 0;
  std::uint8_t valid = 0;
  std::uint32_t depth = 0;
  std::uint8_t depth_written = 0;
};

struct PcoFragmentExecution {
  std::array<std::uint32_t, kPcoPixelOutputCount> pixel_outputs{};
  PcoTextureRequest texture_request{};
  PcoFragmentContinuation continuation{};
  std::uint32_t executed_instruction_count = 0;
  std::uint16_t written_mask = 0;
  std::uint8_t texture_request_valid = 0;
  std::uint8_t suspended = 0;
  bool discarded = false;
  std::uint32_t depth = 0;
  std::uint8_t depth_written = 0;
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
  std::array<std::uint32_t, kPcoMaximumSharedCount> shared_registers{};
  std::array<std::uint32_t, kPcoTextureResponseCount> texture_response{};
  PcoFragmentContinuation continuation{};
  std::uint8_t coefficient_count = 0;
  std::uint16_t shared_count = 0;
  std::uint8_t texture_response_valid = 0;
  PcoMemoryReadCallback memory_read = nullptr;
  void *memory_user_data = nullptr;
};

/* Immutable raw PCO binaries generated by the identified public Mesa backend.
 */
const std::vector<std::uint8_t> &FillSolidVertexPcoBinary();
const std::vector<std::uint8_t> &FillSolidFragmentPcoBinary();
const std::vector<std::uint8_t> &FillSolidBlackFragmentPcoBinary();
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
const std::vector<std::uint8_t> &ConditionalsVertexPcoBinary();
const std::vector<std::uint8_t> &ConditionalsFragmentPcoBinary();

// Counts semantic groups when expand_repeats is false, or repeat-expanded
// operations executed by one shader invocation when it is true.
PcoInstructionCounts
CountPcoInstructions(const std::vector<PcoInstruction> &instructions,
                     bool expand_repeats);

/* Decode real PCO bytes and reject every encoding outside the modeled public
 * subset, including malformed register ranges and non-canonical phase forms. */
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
PcoVertexExecution ResumeVertexPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoVertexContinuation &continuation,
    const std::array<std::uint32_t, kPcoTextureResponseCount> &texture_response,
    PcoMemoryReadCallback memory_read = nullptr,
    void *memory_user_data = nullptr);

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
    const std::array<std::uint32_t, kPcoTextureResponseCount> &texture_response);

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

inline PcoVertexExecution ResumeVertex(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoVertexContinuation &continuation,
    const std::array<std::uint32_t, kPcoTextureResponseCount> &texture_response,
    PcoMemoryReadCallback memory_read = nullptr, void *memory_user_data = nullptr) {
  return ResumeVertexPco(summary, instructions, continuation,
                         texture_response, memory_read, memory_user_data);
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
    const std::array<std::uint32_t, kPcoTextureResponseCount> &texture_response) {
  return ResumeFragmentPco(summary, instructions, continuation,
                           texture_response);
}

static_assert(std::is_trivially_copyable_v<PcoRegisterRef>);
static_assert(std::is_trivially_copyable_v<PcoInstruction>);
static_assert(std::is_trivially_copyable_v<PcoInstructionCounts>);
static_assert(std::is_trivially_copyable_v<PcoProgramSummary>);
static_assert(std::is_trivially_copyable_v<PcoTextureRequest>);
static_assert(std::is_trivially_copyable_v<PcoVertexContinuation>);
static_assert(std::is_trivially_copyable_v<PcoVertexExecution>);
static_assert(std::is_trivially_copyable_v<PcoVertexExecutionContext>);
static_assert(std::is_trivially_copyable_v<PcoFragmentContinuation>);
static_assert(std::is_trivially_copyable_v<PcoFragmentExecution>);
static_assert(std::is_trivially_copyable_v<PcoFragmentExecutionContext>);

} // namespace pvrgpu::stub
