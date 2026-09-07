/* SPDX-License-Identifier: MIT */
/* Pure native ALU tests: no SystemC session, shader substitution, or case-name
 * dispatch. Operands are raw PCO words and modifiers use the decoded ISA ABI. */
#include "shader/pco_iss.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;

void Check(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
float Float(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
std::int32_t Signed(std::uint32_t bits) {
  std::int32_t value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
std::uint32_t IntegerModifier(std::uint32_t bits, bool absolute, bool negate) {
  // Use a wider signed reference so INT_MIN never causes host signed UB.
  std::int64_t value = Signed(bits);
  if (absolute && value < 0) value = -value;
  if (negate) value = -value;
  return static_cast<std::uint32_t>(value);
}

// Independent 16-bit partial-product reference for unsigned IMADD64 high.
std::uint32_t MultiplyAddHigh(std::uint32_t a, std::uint32_t b,
                              std::uint32_t lo, std::uint32_t hi) {
  const auto p0 = (a & 65535U) * (b & 65535U);
  const auto p1 = (a >> 16U) * (b & 65535U);
  const auto p2 = (a & 65535U) * (b >> 16U);
  const auto p3 = (a >> 16U) * (b >> 16U);
  const auto middle = (p0 >> 16U) + (p1 & 65535U) + (p2 & 65535U);
  const auto product_lo = (middle << 16U) | (p0 & 65535U);
  const auto product_hi = p3 + (p1 >> 16U) + (p2 >> 16U) + (middle >> 16U);
  const auto added_lo = product_lo + lo;
  return product_hi + hi + (added_lo < product_lo ? 1U : 0U);
}

void ConstantsAndIntegers() {
  for (std::uint16_t index = 0; index < 32; ++index) {
    std::uint32_t bits = UINT32_MAX;
    Check(PcoSpecialConstantBits(index, &bits) && bits == index,
          "small special constant is an integer word, not binary32");
  }
  const std::array<std::array<std::uint32_t, 2>, 8> constants{{
      {{64, 0x3f800000}}, {{75, 0x3f000000}}, {{95, 0x40490fdb}},
      {{141, 0x80000000}}, {{142, 0x7f800000}}, {{143, 0xffffffff}},
      {{147, 127}}, {{148, 0x7f7fffff}},
  }};
  for (const auto &item : constants) {
    std::uint32_t bits = 0;
    Check(PcoSpecialConstantBits(static_cast<std::uint16_t>(item[0]), &bits) &&
              bits == item[1], "named special constant bit pattern changed");
  }
  for (std::uint16_t index : {32, 63, 96, 127, 160, 65535}) {
    std::uint32_t bits = 0x12345678;
    Check(!PcoSpecialConstantBits(index, &bits) && bits == 0x12345678,
          "unknown special constant must not silently supply zero");
  }
  const std::array<std::uint32_t, 8> values{{
      0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xffff, 0x10000, 0xdeadbeef}};
  PcoInstruction instruction;
  for (auto a : values) for (auto b : values) for (auto c : values) {
    instruction = {};
    instruction.opcode = PcoOpcode::kIntegerAdd;
    Check(EvaluatePcoAluInstruction(instruction, {a, b, 0, 0}) ==
              static_cast<std::uint32_t>(std::uint64_t{a} + b),
          "IADD must wrap modulo 2^32");
    for (unsigned modifiers = 0; modifiers < 32; ++modifiers) {
      instruction.opcode = PcoOpcode::kIntegerMultiplyAdd32;
      instruction.source0_integer_absolute = (modifiers >> 0) & 1;
      instruction.source0_integer_negate = (modifiers >> 1) & 1;
      instruction.source1_integer_absolute = (modifiers >> 2) & 1;
      instruction.source1_integer_negate = (modifiers >> 3) & 1;
      instruction.source2_integer_absolute = (modifiers >> 4) & 1;
      const auto ma = IntegerModifier(a, modifiers & 1, modifiers & 2);
      const auto mb = IntegerModifier(b, modifiers & 4, modifiers & 8);
      const auto mc = IntegerModifier(c, modifiers & 16, false);
      const auto expected = static_cast<std::uint32_t>(std::uint64_t{ma} * mb + mc);
      Check(EvaluatePcoAluInstruction(instruction, {a, b, c, 0}) == expected,
            "IMADD32 ABS/NEG order or integer overflow changed");
    }
    instruction = {};
    instruction.opcode = PcoOpcode::kIntegerMultiplyAdd64High;
    for (auto high : values)
      Check(EvaluatePcoAluInstruction(instruction, {a, b, c, high}) ==
                MultiplyAddHigh(a, b, c, high),
            "IMADD64 high lost unsigned multiplication or addend carry");
  }
}

void Bitfields() {
  const std::array<std::uint32_t, 5> values{{0, 0xffffffff, 0x80000000,
                                          0x12345678, 0xfedcba98}};
  PcoInstruction instruction;
  for (auto value : values) for (unsigned offset = 0; offset < 32; ++offset) {
    for (unsigned width = 0; width <= 31 && width + offset <= 32; ++width) {
      std::uint32_t field = 0;
      for (unsigned bit = 0; bit < width; ++bit)
        field |= ((value >> (offset + bit)) & 1U) << bit;
      instruction.opcode = PcoOpcode::kBitfieldExtractUnsigned;
      Check(EvaluatePcoAluInstruction(instruction, {value, offset, width, 0}) == field,
            "unsigned native bitfield extraction changed");
      instruction.opcode = PcoOpcode::kBitfieldExtractSigned;
      const auto sign_extended = width && (field & (1U << (width - 1)))
          ? field | (UINT32_MAX << width) : field;
      Check(EvaluatePcoAluInstruction(instruction, {value, offset, width, 0}) ==
                sign_extended, "signed native bitfield extraction changed");
      instruction.opcode = PcoOpcode::kBitfieldInsert;
      for (unsigned shift = 0; shift < 2; ++shift) {
        instruction.bitfield_insert_shifts = shift;
        const auto insert = shift ? value << offset : value;
        std::uint32_t expected = 0xa5a5a5a5;
        for (unsigned bit = offset; bit < offset + width; ++bit)
          expected = (expected & ~(1U << bit)) | (insert & (1U << bit));
        Check(EvaluatePcoAluInstruction(instruction,
                  {width, offset, value, 0xa5a5a5a5}) == expected,
              "native MSK/LSL bitfield insertion changed");
      }
    }
    instruction = {};
    instruction.opcode = PcoOpcode::kShiftLeft;
    Check(EvaluatePcoAluInstruction(instruction, {value, offset + 32, 0, 0}) ==
              (value << offset), "LSL must use the native low five shift bits");
    instruction.opcode = PcoOpcode::kShiftRight;
    Check(EvaluatePcoAluInstruction(instruction, {value, offset + 32, 0, 0}) ==
              (value >> offset), "SHR must be unsigned and mask its shift count");
  }
}

template <typename T> bool Predicate(unsigned operation, T a, T b) {
  switch (operation) {
  case 0: return a == T{0};
  case 1: return a > T{0};
  case 2: return a >= T{0};
  case 4: return a == b;
  case 5: return a > b;
  case 6: return a >= b;
  case 7: return a != b;
  case 8: return a < b;
  case 9: return a <= b;
  default: throw std::runtime_error("invalid reference test operation");
  }
}

void Comparisons() {
  const std::array<std::uint32_t, 9> values{{0, 1, 0x80000000, 0x7fffffff,
      0xffffffff, 0x3f800000, 0xbf800000, 0x7f800000, 0x7fc00000}};
  for (unsigned type : {0U, 5U, 6U})
    for (unsigned operation : {0U, 1U, 2U, 4U, 5U, 6U, 7U, 8U, 9U})
      for (auto a : values) for (auto b : values) {
        PcoInstruction instruction;
        instruction.opcode = PcoOpcode::kBooleanCompare;
        instruction.comparison_test_type = type;
        instruction.comparison_test_op = operation;
        instruction.source_count = operation < 3 ? 1 : 2;
        const bool expected = type == 0 ? Predicate(operation, Float(a), Float(b)) :
            type == 5 ? Predicate(operation, a, b) : Predicate(operation, Signed(a), Signed(b));
        Check(EvaluatePcoPredicate(instruction, {a, b, 0, 0}) == expected,
              "TST integer/F32 comparison predicate changed");
        for (unsigned one = 0; one < 2; ++one) {
          instruction.comparison_result_float_one = one;
          Check(EvaluatePcoAluInstruction(instruction, {a, b, 0, 0}) ==
                    (expected ? (one ? 0x3f800000U : UINT32_MAX) : 0U),
                "TST Boolean/PCK.ONE materialization changed");
        }
      }
}

void FloatModifiersAndSelect() {
  PcoInstruction i;
  i.opcode = PcoOpcode::kFloatAddNegateSource0;
  i.source0_floor = i.source0_absolute = i.source1_absolute = 1;
  Check(EvaluatePcoAluInstruction(i, {Bits(-2.25f), Bits(-5.0f), 0, 0}) == Bits(2.0f),
        "FADD modifier order must be floor, absolute, then opcode negate");
  i.saturate = 1;
  Check(EvaluatePcoAluInstruction(i, {Bits(-2.25f), Bits(-5.0f), 0, 0}) == Bits(1.0f),
        "FADD saturation must apply to the final result");
  i = {};
  i.opcode = PcoOpcode::kFloatMultiply;
  i.source0_absolute = i.source0_negate = i.source1_absolute = 1;
  Check(EvaluatePcoAluInstruction(i, {Bits(-2.0f), Bits(-3.0f), 0, 0}) == Bits(-6.0f),
        "FMUL source-zero MBYP negate/ABS ordering changed");
  i = {};
  i.opcode = PcoOpcode::kFloatMadNegateSource2;
  i.source1_absolute = i.source1_negate = 1;
  i.source2_floor = i.source2_absolute = 1;
  Check(EvaluatePcoAluInstruction(i, {Bits(2.0f), Bits(-3.0f), Bits(-2.25f), 0}) == Bits(-9.0f),
        "FMAD source-one negate and source-two floor/ABS/NEG ordering changed");
  for (unsigned mods = 0; mods < 16; ++mods) {
    i = {};
    i.opcode = PcoOpcode::kTestConditionalSelect;
    i.phase_composed = 1;
    i.phase0.source0_absolute = mods & 1;
    i.phase0.opcode = mods & 2 ? PcoOpcode::kFloatNegate : PcoOpcode::kMoveBypass;
    i.phase1.source0_absolute = (mods >> 2) & 1;
    i.phase1.opcode = mods & 8 ? PcoOpcode::kFloatNegate : PcoOpcode::kMoveBypass;
    i.test_source0_result = PcoInternalResult::kPhase0;
    i.test_source1_result = PcoInternalResult::kPhase1;
    i.select_true_result = PcoInternalResult::kPhase0;
    i.select_false_result = PcoInternalResult::kPhase1;
    i.comparison_test_type = 0;
    i.comparison_test_op = 5;
    auto left = Bits(-3.0f), right = Bits(-2.0f);
    if (mods & 1) left &= 0x7fffffff;
    if (mods & 2) left ^= 0x80000000;
    if (mods & 4) right &= 0x7fffffff;
    if (mods & 8) right ^= 0x80000000;
    Check(EvaluatePcoAluInstruction(i, {}, 0,
              {Bits(-3.0f), 0, 0}, {Bits(-2.0f), 0, 0}) ==
              (Float(left) > Float(right) ? left : right),
          "phase-composed MOVC must select modified phase results");
  }
  i = {};
  i.opcode = PcoOpcode::kTestConditionalSelect;
  i.comparison_test_op = 1;
  i.comparison_test_type = 6;
  i.source1_negate = 1;
  for (unsigned inverted = 0; inverted < 2; ++inverted) {
    i.conditional_select_inverted = inverted;
    for (auto condition : {0U, 1U, UINT32_MAX}) {
      const auto expected = ((Signed(condition) > 0) != (inverted != 0))
          ? Bits(-3.0f) : Bits(5.0f);
      Check(EvaluatePcoAluInstruction(i, {condition, Bits(3.0f), Bits(5.0f), 0}) == expected,
            "unary TST/MOVC inversion or selected source negate changed");
    }
  }
}

void NumericConversions() {
  // Expected saturation/NaN results are the shared VS/FS native PCK policy.
  // Halfway values distinguish ties-to-even from round-toward-zero.
  const std::array<std::array<std::uint32_t, 5>, 14> integers{{
      {{Bits(0.0f), 0, 0, 0, 0}},
      {{Bits(-0.0f), 0, 0, 0, 0}},
      {{Bits(0.5f), 0, 0, 0, 0}},
      {{Bits(1.5f), 2, 1, 2, 1}},
      {{Bits(2.5f), 2, 2, 2, 2}},
      {{Bits(3.5f), 4, 3, 4, 3}},
      {{Bits(-1.5f), 0xfffffffe, 0xffffffff, 0, 0}},
      {{Bits(-2.5f), 0xfffffffe, 0xfffffffe, 0, 0}},
      {{0x4effffff, 0x7fffff80, 0x7fffff80, 0x7fffff80, 0x7fffff80}},
      {{0x4f000000, 0x7fffffff, 0x7fffffff, 0x80000000, 0x80000000}},
      {{0xcf000000, 0x80000000, 0x80000000, 0, 0}},
      {{0x7f800000, 0x7fffffff, 0x7fffffff, 0xffffffff, 0xffffffff}},
      {{0xff800000, 0x80000000, 0x80000000, 0, 0}},
      {{0x7fc01234, 0, 0, 0, 0}},
  }};
  const std::array<PcoOpcode, 4> conversions{{PcoOpcode::kFloatToInt32Rtne,
      PcoOpcode::kFloatToInt32Rtz, PcoOpcode::kFloatToUint32Rtne,
      PcoOpcode::kFloatToUint32Rtz}};
  PcoInstruction i;
  for (const auto &row : integers) for (unsigned op = 0; op < conversions.size(); ++op) {
    i.opcode = conversions[op];
    Check(EvaluatePcoAluInstruction(i, {row[0], 0, 0, 0}) == row[op + 1],
          "native float/integer conversion rounding or saturation changed");
  }
  const std::array<std::array<std::uint32_t, 3>, 15> halves{{
      {{0x00000000, 0x0000, 0x0000}}, // zero
      {{0x00000001, 0x0000, 0x0000}}, // binary32 subnormal
      {{0x33000000, 0x0000, 0x0000}}, // half minimum midpoint, tie to zero
      {{0x33000001, 0x0001, 0x0000}}, // just above that midpoint
      {{0x33800000, 0x0001, 0x0001}}, // exact minimum half subnormal
      {{0x387fc000, 0x03ff, 0x03ff}}, // exact maximum half subnormal
      {{0x387fe000, 0x0400, 0x03ff}}, // tie carries to minimum normal
      {{0x38800000, 0x0400, 0x0400}},
      {{0x3f801000, 0x3c00, 0x3c00}}, // even lower mantissa
      {{0x3f803000, 0x3c02, 0x3c01}}, // odd lower mantissa
      {{0x477fe000, 0x7bff, 0x7bff}}, // maximum finite half
      {{0x477ff000, 0x7c00, 0x7bff}}, // RTNE overflow midpoint
      {{0x47800000, 0x7c00, 0x7bff}}, // finite overflow
      {{0x7f800000, 0x7c00, 0x7c00}}, // infinity is not finite saturation
      {{0x7fc01234, 0x7e00, 0x7e00}}, // canonical quiet half NaN
  }};
  for (const auto &row : halves) for (unsigned negative = 0; negative < 2; ++negative)
    for (unsigned rtz = 0; rtz < 2; ++rtz) {
      i.opcode = rtz ? PcoOpcode::kFloatPackHalfRtz : PcoOpcode::kFloatPackHalfRtne;
      Check(EvaluatePcoAluInstruction(i, {row[0] | (negative << 31), 0, 0, 0}) ==
                (row[1 + rtz] | (negative << 15)),
            "native half pack lost rounding mode, sign, or boundary behavior");
    }
  i.opcode = PcoOpcode::kFloatUnpackHalf;
  for (std::uint32_t half = 0; half < 65536; ++half) {
    const auto exponent = (half >> 10) & 31;
    const auto fraction = half & 1023;
    const auto sign = (half & 0x8000) << 16;
    // Numerical powers-of-two reference for all finite halves, independent of
    // the implementation's normalization shifts; exceptional payload is raw.
    const auto expected = exponent == 31
        ? sign | 0x7f800000 | (fraction << 13)
        : Bits(std::ldexp(static_cast<float>(fraction + (exponent ? 1024 : 0)),
                          exponent ? static_cast<int>(exponent) - 25 : -24)) | sign;
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      const auto packed = repeat ? (half << 16) | 0x55aa : 0x55aa0000 | half;
      Check(EvaluatePcoAluInstruction(i, {packed, 0, 0, 0}, repeat) == expected,
            "native half unpack used the wrong repeat half or lost raw bits");
    }
  }
}

void UnsupportedIsExplicit() {
  for (auto opcode : {PcoOpcode::kBufferLoad, PcoOpcode::kBufferStore,
                     PcoOpcode::kTextureSample, PcoOpcode::kAtomicAdd,
                     PcoOpcode::kDerivativeX}) {
    PcoInstruction instruction;
    instruction.opcode = opcode;
    bool rejected = false;
    try { (void)EvaluatePcoAluInstruction(instruction, {}); }
    catch (const std::exception &error) { rejected = error.what()[0] != '\0'; }
    Check(rejected, "a pure ALU call must reject non-ALU execution");
  }
  PcoInstruction predicate;
  predicate.opcode = PcoOpcode::kBooleanCompare;
  predicate.comparison_test_type = 1; // U16 is outside the modeled TST subset.
  predicate.comparison_test_op = 4;
  bool rejected = false;
  try { (void)EvaluatePcoPredicate(predicate, {}); }
  catch (const std::exception &error) { rejected = error.what()[0] != '\0'; }
  Check(rejected, "unsupported TST type must not silently return false");
}
} // namespace

int main() {
  try {
    ConstantsAndIntegers();
    Bitfields();
    Comparisons();
    FloatModifiersAndSelect();
    NumericConversions();
    UnsupportedIsExplicit();
    std::cout << "pure compute ALU checks=" << checks << ": PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pure compute ALU check " << checks << ": " << error.what() << '\n';
    return 1;
  }
}
