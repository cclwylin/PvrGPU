/* SPDX-License-Identifier: MIT */
/* TST/MOVC source modifiers from the public Mesa PCO ISA, not shader results.
 * pco_isa.py I_SNGL_EXT: ABS=bit 0, NEG=bit 1. pco_map.py O_MIN/O_MAX:
 * P0->ft0, P1->ft1, test ft0/ft1, MOVC chooses ft0 on true and ft1 on false.
 * Captured MAX(abs(r2),abs(r1))->r0 group, unchanged from Mesa:
 *   78 d2 00 d0 3c fa 10 97 01 97 01 42 41 10 40 ff
 * Test mutations exercise every legal modifier and both independent inputs. */
#include "shader/pco_iss.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}

void Immediate(std::vector<std::uint8_t> &bytes, unsigned reg,
               std::uint32_t value) {
  bytes.insert(bytes.end(), {0x86, 0x92, 0x40, 0x13});
  for (unsigned i = 0; i < 4; ++i)
    bytes.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
  bytes.insert(bytes.end(), {0, 0, static_cast<std::uint8_t>(0x40 | reg), 0xff});
}

std::vector<std::uint8_t> Binary(ShaderStage stage, bool minimum,
                                unsigned left_modifier, unsigned right_modifier,
                                std::uint32_t left, std::uint32_t right) {
  std::vector<std::uint8_t> bytes;
  Immediate(bytes, 2, left);
  Immediate(bytes, 1, right);
  bytes.insert(bytes.end(), {
      0x78, 0xd2, 0, 0xd0, 0x3c,
      static_cast<std::uint8_t>(minimum ? 0xf0 : 0xfa),
      static_cast<std::uint8_t>(minimum ? 0x11 : 0x10),
      0x97, static_cast<std::uint8_t>(right_modifier),
      0x97, static_cast<std::uint8_t>(left_modifier),
      0x42, 0x41, 0x10, 0x40, 0xff});
  if (stage == ShaderStage::kFragment) {
    for (unsigned channel = 0; channel < 4; ++channel)
      bytes.insert(bytes.end(), {0x34, 0x8a,
          static_cast<std::uint8_t>(channel == 3 ? 0x80 : 0),
          0x87, 0x40, 0, 0, static_cast<std::uint8_t>(0x20 + channel)});
  } else {
    Immediate(bytes, 1, 0);
    Immediate(bytes, 2, 0);
    Immediate(bytes, 3, 0x3f800000);
    bytes.insert(bytes.end(), {0x55, 0xa0, 0x06, 0x08, 0, 0xc0, 0, 0, 0, 0x30,
                               0x44, 0xa0, 0x80, 0x05, 0, 0, 0, 0xff});
  }
  return bytes;
}

std::uint32_t Modified(std::uint32_t bits, unsigned modifier) {
  if (modifier & 1) bits &= 0x7fffffff;
  if (modifier & 2) bits ^= 0x80000000;
  return bits;
}

std::vector<std::uint8_t> UnaryBinary(ShaderStage stage, unsigned true_modifier,
                                     unsigned false_modifier,
                                     std::uint32_t condition) {
  std::vector<std::uint8_t> bytes;
  Immediate(bytes, 0, 0xc0400000);  // true value: -3
  Immediate(bytes, 13, 0xc0000000); // false value: -2
  Immediate(bytes, 25, condition);
  /* O_CSEL-style MOVC ft0/is4(ft1), U32 zero test via lower-source s1.
   * Unlike MIN/MAX the tested condition bypasses both MBYP phases. */
  bytes.insert(bytes.end(), {0x79, 0xd2, 0, 0xd0, 0x3c, 0xf0, 0xb0,
      0x97, static_cast<std::uint8_t>(false_modifier),
      0x97, static_cast<std::uint8_t>(true_modifier),
      0xc0, 0xf9, 0x80, 0x10, 0x4d, 0x11, 0x4d});
  if (stage == ShaderStage::kFragment) {
    for (unsigned channel = 0; channel < 4; ++channel)
      bytes.insert(bytes.end(), {0x34, 0x8a,
          static_cast<std::uint8_t>(channel == 3 ? 0x80 : 0),
          0x87, 0x4d, 0, 0, static_cast<std::uint8_t>(0x20 + channel)});
  } else {
    Immediate(bytes, 14, 0);
    Immediate(bytes, 15, 0);
    Immediate(bytes, 16, 0x3f800000);
    bytes.insert(bytes.end(), {0x55, 0xa0, 0x06, 0x08, 0, 0xcd, 0, 0, 0, 0x30,
                               0x44, 0xa0, 0x80, 0x05, 0, 0, 0, 0xff});
  }
  return bytes;
}

std::uint32_t Expected(std::uint32_t left, std::uint32_t right, bool minimum) {
  float left_float, right_float;
  std::memcpy(&left_float, &left, sizeof(left));
  std::memcpy(&right_float, &right, sizeof(right));
  /* Ordered compare and exact selected source, including NaN payload/-0. */
  const bool passed = minimum ? left_float < right_float : left_float > right_float;
  return passed ? left : right;
}

std::vector<std::uint8_t> IntegerBinary(ShaderStage stage, unsigned type,
                                       unsigned operation, std::uint32_t left,
                                       std::uint32_t right) {
  auto bytes = Binary(stage, false, 0, 0, left, right);
  /* Compact, unmodified MBYPs exercise the pair-select decoder as emitted
   * by Mesa O_MIN/O_MAX, rather than the extended-source dispatch. */
  bytes.erase(bytes.begin() + 24, bytes.begin() + 40);
  bytes.insert(bytes.begin() + 24, {0x77, 0xd2, 0, 0xd0, 0x3c,
      static_cast<std::uint8_t>(0xf0 | ((operation & 7u) << 1)),
      static_cast<std::uint8_t>((type << 5) | 0x10 | (operation >> 3)),
      0x87, 0x87, 0x42, 0x41, 0x10, 0x40, 0xff});
  return bytes;
}

template <typename Integer>
bool IntegerPredicate(unsigned operation, Integer left, Integer right) {
  switch (operation) {
  case 4: return left == right;
  case 5: return left > right;
  case 6: return left >= right;
  case 7: return left != right;
  case 8: return left < right;
  case 9: return left <= right;
  default: throw std::runtime_error("invalid integer test operation");
  }
}

void TestIntegerSelect(ShaderStage stage) {
  const std::array<std::uint32_t, 8> inputs = {
      0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff, 0x12345678};
  for (const unsigned type : {5u, 6u}) {
    for (unsigned operation = 4; operation <= 9; ++operation) {
      for (auto left : inputs) {
        for (auto right : inputs) {
          const auto decoded = Decode(stage, IntegerBinary(stage, type, operation,
                                                            left, right));
          std::int32_t signed_left, signed_right;
          std::memcpy(&signed_left, &left, sizeof(left));
          std::memcpy(&signed_right, &right, sizeof(right));
          const bool passed = type == 5
              ? IntegerPredicate(operation, left, right)
              : IntegerPredicate(operation, signed_left, signed_right);
          const auto expected = passed ? left : right;
          const auto actual = stage == ShaderStage::kVertex
              ? ExecuteVertex(decoded.summary, decoded.instructions, {}).outputs[0]
              : ExecuteFragment(decoded.summary, decoded.instructions).pixel_outputs[0];
          Check(actual == expected, "integer TST/MOVC lost signedness or raw selected bits");
        }
      }
    }
  }
}
} // namespace

int main() {
  try {
    const std::array<std::uint32_t, 9> inputs = {
      0xc0400000, 0x40000000, 0x80000000, 0,
      0xff800000, 0x7f800000, 0xffc12345, 0x7fc23456, 0x00000001};
    for (const auto stage : {ShaderStage::kVertex, ShaderStage::kFragment}) {
      TestIntegerSelect(stage);
      for (const bool minimum : {false, true}) {
        for (unsigned left_modifier = 0; left_modifier < 4; ++left_modifier) {
          for (unsigned right_modifier = 0; right_modifier < 4; ++right_modifier) {
            for (auto left : inputs) {
              for (auto right : inputs) {
                const auto decoded = Decode(stage, Binary(stage, minimum,
                    left_modifier, right_modifier, left, right));
                const auto &select = decoded.instructions.at(2);
                Check(select.phase_composed == 1 &&
                      select.phase0.source.index == 2 &&
                      select.phase1.source.index == 1,
                      "TST/MOVC retains phase source routing");
                const auto expected = Expected(Modified(left, left_modifier),
                                               Modified(right, right_modifier),
                                               minimum);
                const auto actual = stage == ShaderStage::kVertex
                    ? ExecuteVertex(decoded.summary, decoded.instructions, {}).outputs[0]
                    : ExecuteFragment(decoded.summary, decoded.instructions).pixel_outputs[0];
                Check(actual == expected, "modified TST/MOVC selected wrong source bits");
              }
            }
          }
        }
      }
      for (const unsigned at : {32u, 34u}) {
        for (const unsigned reserved_modifier : {4u, 8u, 0x80u}) {
          auto malformed = Binary(stage, false, 1, 1, 0xc0400000, 0x40000000);
          malformed[at] = static_cast<std::uint8_t>(reserved_modifier);
          bool rejected = false;
          try { (void)Decode(stage, malformed); }
          catch (const std::exception &) { rejected = true; }
          Check(rejected, "reserved MBYP modifier was accepted");
        }
      }
      for (unsigned true_modifier = 0; true_modifier < 4; ++true_modifier) {
        for (unsigned false_modifier = 0; false_modifier < 4; ++false_modifier) {
          for (const std::uint32_t condition : {0u, 1u, UINT32_MAX}) {
            const auto decoded = Decode(stage, UnaryBinary(stage, true_modifier,
                                                            false_modifier,
                                                            condition));
            const auto expected = condition == 0
                ? Modified(0xc0400000, true_modifier)
                : Modified(0xc0000000, false_modifier);
            const auto actual = stage == ShaderStage::kVertex
                ? ExecuteVertex(decoded.summary, decoded.instructions, {}).outputs[0]
                : ExecuteFragment(decoded.summary, decoded.instructions).pixel_outputs[0];
            Check(actual == expected, "unary TST lost true/false MBYP modifiers");
          }
        }
      }
    }
    std::cout << "TST/MOVC MBYP tests: " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TST/MOVC MBYP failure: " << error.what() << '\n';
    return 1;
  }
}
