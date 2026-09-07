#pragma once

#include "pco_uniform_buffer_fixtures.h"
#include <algorithm>
#include <array>

namespace pvrgpu::stub::test {

// Actual Mesa pco_isa.py 1up_3b11i / 1lo_3b11i_2m emitter output; reproduce
// with tools/pco-fixtures/generate_temp256_operands.c. No model packing helper
// is used to derive the byte sequences being tested.
inline std::array<std::uint8_t, 3> MesaTemporaryOperand(unsigned index) {
  struct Entry { unsigned index; std::array<std::uint8_t, 3> bytes; };
  static constexpr Entry entries[] = {
      {0, {0xc0, 0x00, 0x00}}, {60, {0xfc, 0x00, 0x00}},
      {63, {0xff, 0x00, 0x00}}, {64, {0xc0, 0x01, 0x00}},
      {124, {0xfc, 0x01, 0x00}}, {127, {0xff, 0x01, 0x00}},
      {128, {0xc0, 0x02, 0x00}}, {188, {0xfc, 0x02, 0x00}},
      {191, {0xff, 0x02, 0x00}}, {192, {0xc0, 0x03, 0x00}},
      {240, {0xf0, 0x03, 0x00}}, {252, {0xfc, 0x03, 0x00}},
      {253, {0xfd, 0x03, 0x00}}, {254, {0xfe, 0x03, 0x00}},
      {255, {0xff, 0x03, 0x00}}, {256, {0xc0, 0x00, 0x01}},
  };
  for (const auto &entry : entries) {
    if (entry.index == index)
      return entry.bytes;
  }
  throw std::invalid_argument("TEMP operand fixture is absent");
}

inline std::array<std::uint8_t, 3> MesaTemporaryPair(unsigned index) {
  struct Entry { unsigned index; std::array<std::uint8_t, 3> bytes; };
  static constexpr Entry entries[] = {
      {63, {0xbf, 0xc0, 0x08}}, {64, {0xc0, 0xc1, 0x08}},
      {127, {0xff, 0xc0, 0x10}}, {128, {0x80, 0xc1, 0x11}},
      {191, {0xbf, 0xc0, 0x19}}, {192, {0xc0, 0xc1, 0x19}},
      {240, {0xf0, 0xf1, 0x19}}, {254, {0xfe, 0xff, 0x19}},
  };
  for (const auto &entry : entries) {
    if (entry.index == index)
      return entry.bytes;
  }
  throw std::invalid_argument("TEMP pair operand fixture is absent");
}

// Reuse the compiler's two signed ADD64_32 groups and scalar setup, then
// assemble LD/WDF and four raw exports using the public encodings. These are
// intentionally ISA-level test inputs, not claimed as unmodified GLSL output.
// The descriptor is SH0..3, and SH4 is the runtime byte offset. Every output
// exports the last DWORD of the loaded TEMP range, including TEMP255.
inline std::vector<std::uint8_t> TemporaryFileUniformBufferFixture(
    bool vertex, unsigned base, unsigned count) {
  if (count == 0 || count > 16)
    throw std::invalid_argument("invalid TEMP fixture LD width");
  const auto original = UniformBufferFixture(vertex, 4);
  std::vector<std::uint8_t> binary(original.begin(), original.begin() + 48);
  const auto destination = MesaTemporaryOperand(base);
  binary.insert(binary.end(), {0x66, 0xa0, 0x00, 0xf1,
      static_cast<std::uint8_t>((count & 7U) << 2U),
      static_cast<std::uint8_t>((count >> 3U) & 1U), 0x41});
  binary.insert(binary.end(), destination.begin(), destination.end());
  binary.insert(binary.end(), {0x00, 0xff, 0x02, 0x80, 0x6a, 0xff});
  const auto source = MesaTemporaryOperand(base + count - 1U);
  for (unsigned channel = 0; channel < 4; ++channel) {
    const std::uint8_t end = channel == 3 ? 0x80 : 0;
    if (vertex) {
      binary.insert(binary.end(), {0x55, 0xa0, end,
          static_cast<std::uint8_t>(channel == 3 ? 0x0e : 0x08),
          static_cast<std::uint8_t>(channel)});
      binary.insert(binary.end(), source.begin(), source.end());
      binary.insert(binary.end(), {0x00, 0x30});
    } else {
      binary.insert(binary.end(), {0x35, 0x8a, end, 0x87});
      binary.insert(binary.end(), source.begin(), source.end());
      binary.insert(binary.end(), {0x00, 0x00,
          static_cast<std::uint8_t>(0x20U + channel)});
    }
  }
  return binary;
}

inline std::vector<std::uint8_t> TemporaryAddressUniformBufferFixture(
    bool vertex, unsigned address_base, bool long_form) {
  auto binary = TemporaryFileUniformBufferFixture(vertex, 255, 1);
  const auto pair = MesaTemporaryPair(address_base);
  // The 3-byte pair takes the original dual destination's trailing pad byte.
  std::copy(pair.begin(), pair.end(), binary.begin() + 45);
  if (long_form) {
    binary[34] += 1;
    binary[47] |= 0x80; // I_TWO_3B11I_3B11I ext2, both high3 index fields zero.
    binary.insert(binary.begin() + 48, {0x00, 0xff});
  }
  const unsigned load = long_form ? 50 : 48;
  const auto address = MesaTemporaryOperand(address_base);
  binary[load] += 1;
  binary[load + 6] = address[0];
  binary.insert(binary.begin() + load + 7, {address[1], address[2]});
  return binary;
}

} // namespace pvrgpu::stub::test
