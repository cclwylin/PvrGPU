// Independent integer reference for the public PCO DMA atomic operation field.
// Test arithmetic deliberately uses widened values rather than the model RMW.
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

inline constexpr std::array<unsigned, 10> kComputeAtomicNibbles{
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10};
inline constexpr std::array<std::uint32_t, 8> kComputeAtomicWords{
    0, 1, 0x7fffffffU, 0x80000000U, 0x80000001U, 0xffffffffU,
    0x55555555U, 0xaaaaaaaaU};

inline std::uint32_t ComputeAtomicReference(unsigned op, std::uint32_t old,
                                            std::uint32_t operand) {
  const std::uint64_t a = old, b = operand;
  const std::int64_t sa = a >= UINT64_C(0x80000000) ?
      static_cast<std::int64_t>(a) - INT64_C(0x100000000) : static_cast<std::int64_t>(a);
  const std::int64_t sb = b >= UINT64_C(0x80000000) ?
      static_cast<std::int64_t>(b) - INT64_C(0x100000000) : static_cast<std::int64_t>(b);
  switch (op) {
  case 0: return static_cast<std::uint32_t>((a + b) & UINT64_C(0xffffffff));
  case 1: return static_cast<std::uint32_t>((a + UINT64_C(0x100000000) - b) & UINT64_C(0xffffffff));
  case 2: return operand;
  case 4: return a <= b ? old : operand;
  case 5: return sa <= sb ? old : operand;
  case 6: return a >= b ? old : operand;
  case 7: return sa >= sb ? old : operand;
  case 8: return static_cast<std::uint32_t>(a & b);
  case 9: return static_cast<std::uint32_t>(a | b);
  case 10: return static_cast<std::uint32_t>(a ^ b);
  default: throw std::runtime_error("reserved PCO DMA atomic operation in test");
  }
}
