/*
 * Functional tests for the PowerVR PCO instruction-set simulator (ISS).
 *
 * PCO is Mesa's public PowerVR shader-backend name, ISS means Instruction Set
 * Simulator, and USC means Unified Shading Cluster.  These tests pin the exact
 * Mesa-generated Fill.Solid, attribute and varying bytes, execute raw USC
 * register/coefficient values, and prove that malformed opcode, register,
 * stage, padding, length, synchronization, and .end encodings fail closed
 * instead of falling back to shader-name-specific behavior.
 */
#include "shader/pco_iss.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using pvrgpu::stub::CountPcoInstructions;
using pvrgpu::stub::ConditionalsFragmentPcoBinary;
using pvrgpu::stub::ConditionalsVertexPcoBinary;
using pvrgpu::stub::Decode;
using pvrgpu::stub::ExecuteFragment;
using pvrgpu::stub::ExecuteVertex;
using pvrgpu::stub::ResumeFragment;
using pvrgpu::stub::ResumeVertex;
using pvrgpu::stub::AttributeFetchEightAttributeVertexPcoBinary;
using pvrgpu::stub::AttributeFetchFourAttributeVertexPcoBinary;
using pvrgpu::stub::AttributeFetchGrayFragmentPcoBinary;
using pvrgpu::stub::AttributeFetchTwoAttributeVertexPcoBinary;
using pvrgpu::stub::AttributeFetchVertexPcoBinary;
using pvrgpu::stub::FillSolidBlackFragmentPcoBinary;
using pvrgpu::stub::FillSolidFragmentPcoBinary;
using pvrgpu::stub::FillSolidGreenHalfAlphaFragmentPcoBinary;
using pvrgpu::stub::FillSolidRedHalfAlphaFragmentPcoBinary;
using pvrgpu::stub::FillSolidVertexPcoBinary;
using pvrgpu::stub::FillTexNearestFragmentPcoBinary;
using pvrgpu::stub::FillTexNearestVertexPcoBinary;
using pvrgpu::stub::PcoOpcode;
using pvrgpu::stub::PcoFragmentExecutionContext;
using pvrgpu::stub::PcoInstruction;
using pvrgpu::stub::PcoVertexExecutionContext;
using pvrgpu::stub::PcoIterationMode;
using pvrgpu::stub::PcoProgramSummary;
using pvrgpu::stub::PcoRegisterBank;
using pvrgpu::stub::PcoWriteTarget;
using pvrgpu::stub::ShaderStage;
using pvrgpu::stub::TriangleSetupCyanFragmentPcoBinary;
using pvrgpu::stub::TriangleSetupOrangeFragmentPcoBinary;
using pvrgpu::stub::VaryingsOneFragmentPcoBinary;
using pvrgpu::stub::VaryingsOneVertexPcoBinary;
using pvrgpu::stub::VaryingsTwoFragmentPcoBinary;
using pvrgpu::stub::VaryingsTwoVertexPcoBinary;
using pvrgpu::stub::VaryingsFourFragmentPcoBinary;
using pvrgpu::stub::VaryingsFourVertexPcoBinary;
using pvrgpu::stub::VaryingsEightFragmentPcoBinary;
using pvrgpu::stub::VaryingsEightVertexPcoBinary;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("test check failed: " + message);
}

template <typename Function>
void ExpectFailure(Function &&function, const std::string &description) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected failure: " + description);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t Fnv1a64(const std::vector<std::uint8_t> &bytes) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

std::vector<std::uint8_t> BytesFromHex(std::string_view text) {
  std::istringstream input{std::string(text)};
  input >> std::hex;
  std::vector<std::uint8_t> bytes;
  unsigned value = 0;
  while (input >> value) {
    if (value > 0xffU)
      throw std::runtime_error("hex byte is outside uint8 range");
    bytes.push_back(static_cast<std::uint8_t>(value));
  }
  if (!input.eof())
    throw std::runtime_error("invalid hexadecimal byte fixture");
  return bytes;
}

PcoFragmentExecutionContext MakeVaryingsOneContext() {
  PcoFragmentExecutionContext context;
  context.coefficient_count = 20;
  context.sample_x = UINT32_C(0x42020000); // 32.5
  context.sample_y = UINT32_C(0x423e0000); // 47.5
  /* Five A/B/C/PAD sets.  cf0..3 is the non-constant reciprocal-W plane
   * x/128 + 63/256, which evaluates to 0.5 at this sample but represents
   * unequal vertex W values.  The numerator planes produce x/128, y/128,
   * 1/8 and 1/2, exercising non-affine perspective division. */
  context.coefficients = {
      UINT32_C(0x3c000000), UINT32_C(0x00000000),
      UINT32_C(0x3e7c0000), UINT32_C(0xdeadbeef),
      UINT32_C(0x3c000000), UINT32_C(0x00000000),
      UINT32_C(0x00000000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x3c000000),
      UINT32_C(0x00000000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x00000000),
      UINT32_C(0x3e000000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x00000000),
      UINT32_C(0x3f000000), UINT32_C(0xdeadbeef),
  };
  return context;
}

PcoFragmentExecutionContext MakeVaryingsTwoContext() {
  PcoFragmentExecutionContext context;
  context.coefficient_count = 36;
  context.sample_x = UINT32_C(0x42020000); // 32.5
  context.sample_y = UINT32_C(0x423e0000); // 47.5
  /* Nine A/B/C/PAD sets. cf0..3 is the same non-constant reciprocal-W
   * plane as the case-1 test. Both linked vec4 varyings carry c/2, so each
   * numerator plane is half the case-1 plane and their public FADD sum must
   * reconstruct the exact case-1 result. */
  context.coefficients = {
      UINT32_C(0x3c000000), UINT32_C(0x00000000),
      UINT32_C(0x3e7c0000), UINT32_C(0xdeadbeef),
      UINT32_C(0x3b800000), UINT32_C(0x00000000),
      UINT32_C(0x00000000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x3b800000),
      UINT32_C(0x00000000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x00000000),
      UINT32_C(0x3d800000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x00000000),
      UINT32_C(0x3e800000), UINT32_C(0xdeadbeef),
      UINT32_C(0x3b800000), UINT32_C(0x00000000),
      UINT32_C(0x00000000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x3b800000),
      UINT32_C(0x00000000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x00000000),
      UINT32_C(0x3d800000), UINT32_C(0xdeadbeef),
      UINT32_C(0x00000000), UINT32_C(0x00000000),
      UINT32_C(0x3e800000), UINT32_C(0xdeadbeef),
  };
  return context;
}

PcoFragmentExecutionContext MakeVaryingsFourContext() {
  PcoFragmentExecutionContext context;
  context.coefficient_count = 68;
  context.sample_x = UINT32_C(0x42020000); // 32.5
  context.sample_y = UINT32_C(0x423e0000); // 47.5
  /* Seventeen A/B/C/PAD sets. cf0..3 is the same non-constant reciprocal-W
   * plane as cases 1/2. Each of four vec4 varyings carries c/4, so its
   * numerator planes are exactly one quarter of case 1. The public shader's
   * three ordered FADD layers must reconstruct the same final c value. */
  context.coefficients[0] = UINT32_C(0x3c000000);
  context.coefficients[1] = UINT32_C(0x00000000);
  context.coefficients[2] = UINT32_C(0x3e7c0000);
  context.coefficients[3] = UINT32_C(0xdeadbeef);
  for (std::size_t varying = 0; varying < 4; ++varying) {
    const std::size_t base = 4 + varying * 16;
    context.coefficients[base + 0] = UINT32_C(0x3b000000);
    context.coefficients[base + 1] = UINT32_C(0x00000000);
    context.coefficients[base + 2] = UINT32_C(0x00000000);
    context.coefficients[base + 3] = UINT32_C(0xdeadbeef);
    context.coefficients[base + 4] = UINT32_C(0x00000000);
    context.coefficients[base + 5] = UINT32_C(0x3b000000);
    context.coefficients[base + 6] = UINT32_C(0x00000000);
    context.coefficients[base + 7] = UINT32_C(0xdeadbeef);
    context.coefficients[base + 8] = UINT32_C(0x00000000);
    context.coefficients[base + 9] = UINT32_C(0x00000000);
    context.coefficients[base + 10] = UINT32_C(0x3d000000);
    context.coefficients[base + 11] = UINT32_C(0xdeadbeef);
    context.coefficients[base + 12] = UINT32_C(0x00000000);
    context.coefficients[base + 13] = UINT32_C(0x00000000);
    context.coefficients[base + 14] = UINT32_C(0x3e000000);
    context.coefficients[base + 15] = UINT32_C(0xdeadbeef);
  }
  return context;
}

PcoFragmentExecutionContext MakeVaryingsEightContext() {
  PcoFragmentExecutionContext context;
  context.coefficient_count = 132;
  context.sample_x = UINT32_C(0x42020000); // 32.5
  context.sample_y = UINT32_C(0x423e0000); // 47.5
  /* Thirty-three A/B/C/PAD sets. cf0..3 retains the unequal-W plane used
   * by cases 1/2/4. Each of the eight linked vec4 values carries c/8, so
   * seven ordered public FADD layers must reconstruct the exact case-1 c. */
  context.coefficients[0] = UINT32_C(0x3c000000);
  context.coefficients[1] = UINT32_C(0x00000000);
  context.coefficients[2] = UINT32_C(0x3e7c0000);
  context.coefficients[3] = UINT32_C(0xdeadbeef);
  for (std::size_t varying = 0; varying < 8; ++varying) {
    const std::size_t base = 4 + varying * 16;
    context.coefficients[base + 0] = UINT32_C(0x3a800000);
    context.coefficients[base + 1] = UINT32_C(0x00000000);
    context.coefficients[base + 2] = UINT32_C(0x00000000);
    context.coefficients[base + 3] = UINT32_C(0xdeadbeef);
    context.coefficients[base + 4] = UINT32_C(0x00000000);
    context.coefficients[base + 5] = UINT32_C(0x3a800000);
    context.coefficients[base + 6] = UINT32_C(0x00000000);
    context.coefficients[base + 7] = UINT32_C(0xdeadbeef);
    context.coefficients[base + 8] = UINT32_C(0x00000000);
    context.coefficients[base + 9] = UINT32_C(0x00000000);
    context.coefficients[base + 10] = UINT32_C(0x3c800000);
    context.coefficients[base + 11] = UINT32_C(0xdeadbeef);
    context.coefficients[base + 12] = UINT32_C(0x00000000);
    context.coefficients[base + 13] = UINT32_C(0x00000000);
    context.coefficients[base + 14] = UINT32_C(0x3d800000);
    context.coefficients[base + 15] = UINT32_C(0xdeadbeef);
  }
  return context;
}

PcoFragmentExecutionContext MakeFillTexNearestContext() {
  PcoFragmentExecutionContext context;
  context.coefficient_count = 12;
  context.shared_count = 20;
  context.sample_x = UINT32_C(0x42020000); // 32.5
  context.sample_y = UINT32_C(0x423e0000); // 47.5
  /* cf0..3 is a non-constant reciprocal-W plane which evaluates to 0.5.
   * Constant numerator planes 0.125 and 0.375 therefore interpolate to
   * exact normalized texture coordinates (0.25, 0.75). */
  context.coefficients[0] = UINT32_C(0x3c000000);
  context.coefficients[1] = UINT32_C(0x00000000);
  context.coefficients[2] = UINT32_C(0x3e7c0000);
  context.coefficients[3] = UINT32_C(0xdeadbeef);
  context.coefficients[4] = UINT32_C(0x00000000);
  context.coefficients[5] = UINT32_C(0x00000000);
  context.coefficients[6] = UINT32_C(0x3e000000);
  context.coefficients[7] = UINT32_C(0xdeadbeef);
  context.coefficients[8] = UINT32_C(0x00000000);
  context.coefficients[9] = UINT32_C(0x00000000);
  context.coefficients[10] = UINT32_C(0x3ec00000);
  context.coefficients[11] = UINT32_C(0xdeadbeef);
  for (std::size_t word = 0; word < context.shared_registers.size(); ++word)
    context.shared_registers[word] = UINT32_C(0x10000000) + word;
  return context;
}

void TestEmbeddedBinaries() {
  const std::vector<std::uint8_t> expected_vertex = {
      0x58, 0xa0, 0x04, 0x08, 0x00, 0x80, 0x04, 0x00, 0x00, 0x30, 0xf3,
      0xff, 0xff, 0xff, 0xff, 0xff, 0x58, 0xa0, 0x80, 0x0e, 0x03, 0x80,
      0x01, 0x00, 0x00, 0x30, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
  };
  const std::vector<std::uint8_t> expected_fragment = {
      0x35, 0x8a, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x20, 0x34, 0x8a,
      0x00, 0x87, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x00, 0x00,
      0x00, 0x22, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
      0x80, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
  };
  const std::vector<std::uint8_t> expected_red_half_alpha = {
      0x35, 0x8a, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x20, 0x34, 0x8a,
      0x00, 0x87, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x00, 0x00,
      0x00, 0x22, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
      0x8b, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
  };
  const std::vector<std::uint8_t> expected_green_half_alpha = {
      0x34, 0x8a, 0x00, 0x87, 0x00, 0x00, 0x00, 0x20, 0x35, 0x8a, 0x00, 0x87,
      0x80, 0x01, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x00, 0x00,
      0x00, 0x22, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
      0x8b, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
  };
  const std::vector<std::uint8_t> expected_triangle_setup_orange = {
      0x35, 0x8a, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x20, 0x35, 0x8a,
      0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x21, 0x36, 0x8a, 0x00, 0x87,
      0x00, 0x00, 0x00, 0x22, 0xf2, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
      0x80, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
  };
  const std::vector<std::uint8_t> expected_triangle_setup_half_culled_cyan = {
      0x34, 0x8a, 0x00, 0x87, 0x00, 0x00, 0x00, 0x20, 0x35, 0x8a, 0x00, 0x87,
      0x8b, 0x01, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x8b, 0x01,
      0x00, 0x00, 0x00, 0x22, 0xf2, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
      0x80, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
  };
  const std::vector<std::uint8_t> expected_attribute_fetch_vertex = {
      0x35, 0x82, 0x00, 0x87, 0x80, 0x04, 0x00, 0x00, 0x00, 0x40,
      0x35, 0x82, 0x00, 0x87, 0x81, 0x04, 0x00, 0x00, 0x00, 0x41,
      0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x42,
      0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x43,
      0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
      0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
  };
  const std::vector<std::uint8_t>
      expected_attribute_fetch_two_attribute_vertex = {
          0x35, 0x82, 0x00, 0x00, 0x80, 0xc2, 0x18, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0x81, 0xc3, 0x18, 0x00, 0x00, 0x41,
          0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x42,
          0x35, 0x82, 0x00, 0x87, 0x81, 0x01, 0x00, 0x00, 0x00, 0x43,
          0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
          0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
      };
  const std::vector<std::uint8_t>
      expected_attribute_fetch_four_attribute_vertex = {
          0x35, 0x82, 0x00, 0x00, 0x80, 0xc2, 0x18, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0x81, 0xc3, 0x18, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xc4, 0x08, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xc5, 0x08, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xc6, 0x08, 0x00, 0x00, 0x42,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xc7, 0x08, 0x00, 0x00, 0x43,
          0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x44,
          0x35, 0x82, 0x00, 0x87, 0x82, 0x01, 0x00, 0x00, 0x00, 0x45,
          0x55, 0xa0, 0x06, 0x08, 0x00, 0xc2, 0x00, 0x00, 0x00, 0x30,
          0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
      };
  const std::vector<std::uint8_t>
      expected_attribute_fetch_eight_attribute_vertex = {
          0x35, 0x82, 0x00, 0x00, 0x80, 0xc2, 0x18, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0x81, 0xc3, 0x18, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xc4, 0x08, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xc5, 0x08, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xc6, 0x08, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xc7, 0x08, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xc8, 0x08, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xc9, 0x08, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xca, 0x08, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xcb, 0x08, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xcc, 0x08, 0x00, 0x00, 0x40,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xcd, 0x08, 0x00, 0x00, 0x41,
          0x35, 0x82, 0x00, 0x00, 0xc0, 0xce, 0x08, 0x00, 0x00, 0x42,
          0x35, 0x82, 0x00, 0x00, 0xc1, 0xcf, 0x08, 0x00, 0x00, 0x43,
          0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x44,
          0x35, 0x82, 0x00, 0x87, 0x83, 0x01, 0x00, 0x00, 0x00, 0x45,
          0x55, 0xa0, 0x06, 0x08, 0x00, 0xc2, 0x00, 0x00, 0x00, 0x30,
          0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
      };
  const std::vector<std::uint8_t> expected_attribute_fetch_gray_fragment = {
      0x35, 0x8a, 0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x20,
      0x35, 0x8a, 0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x21,
      0x36, 0x8a, 0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x22,
      0xf1, 0xff,
      0x38, 0x8a, 0x80, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x23,
      0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
  };
  const std::vector<std::uint8_t> expected_varyings_one_vertex = {
      0x35, 0x82, 0x00, 0x87, 0x80, 0x04, 0x00, 0x00, 0x00, 0x40,
      0x35, 0x82, 0x00, 0x87, 0x81, 0x04, 0x00, 0x00, 0x00, 0x41,
      0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x42,
      0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x43,
      0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
      0x58, 0xa0, 0x06, 0x08, 0x04, 0xc0, 0x00, 0x00, 0x00, 0x30,
      0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
      0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
  };
  const std::vector<std::uint8_t> expected_varyings_one_fragment = {
      0x56, 0xa0, 0x00, 0xb0, 0x04, 0xc4, 0x40, 0x10, 0xc0, 0x40, 0x00, 0xff,
      0x02, 0x80, 0x6a, 0xff,
      0x34, 0x8a, 0x00, 0x87, 0x40, 0x00, 0x00, 0x20,
      0x34, 0x8a, 0x00, 0x87, 0x41, 0x00, 0x00, 0x21,
      0x34, 0x8a, 0x00, 0x87, 0x42, 0x00, 0x00, 0x22,
      0x34, 0x8a, 0x80, 0x87, 0x43, 0x00, 0x00, 0x23,
  };
  const std::vector<std::uint8_t> expected_varyings_two_vertex = {
      0x35, 0x82, 0x00, 0x87, 0x80, 0x04, 0x00, 0x00, 0x00, 0x40, 0x35, 0x82,
      0x00, 0x87, 0x81, 0x04, 0x00, 0x00, 0x00, 0x41, 0x34, 0x82, 0x00, 0x87,
      0x00, 0x00, 0x00, 0x42, 0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00,
      0x00, 0x43, 0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
      0x35, 0x82, 0x00, 0x40, 0x81, 0xcb, 0x12, 0x00, 0x00, 0x45, 0x35, 0x82,
      0x00, 0x40, 0x80, 0xcb, 0x12, 0x00, 0x00, 0x44, 0x34, 0x82, 0x00, 0x87,
      0x00, 0x00, 0x00, 0x46, 0x35, 0x82, 0x00, 0x87, 0x8b, 0x01, 0x00, 0x00,
      0x00, 0x47, 0x55, 0xa0, 0x06, 0x08, 0x04, 0xc4, 0x00, 0x00, 0x00, 0x30,
      0x58, 0xa0, 0x06, 0x08, 0x08, 0xc4, 0x00, 0x00, 0x00, 0x30, 0xf3, 0xff,
      0xff, 0xff, 0xff, 0xff, 0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
  };
  const std::vector<std::uint8_t> expected_varyings_two_fragment = {
      0x56, 0xa0, 0x00, 0xb0, 0x04, 0xd4, 0x40, 0x10, 0xc0, 0x40, 0x00, 0xff,
      0x02, 0x80, 0x6a, 0xff, 0x56, 0xa0, 0x00, 0xb0, 0x04, 0xc4, 0x40, 0x10,
      0xc0, 0x44, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff, 0x35, 0x82, 0x00, 0x00,
      0xc4, 0xa0, 0x00, 0x00, 0x48, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc5, 0xa1,
      0x00, 0x00, 0x49, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc6, 0xa2, 0x00, 0x00,
      0x4a, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc7, 0xa3, 0x00, 0x00, 0x40, 0xff,
      0x34, 0x8a, 0x00, 0x87, 0x48, 0x00, 0x00, 0x20, 0x34, 0x8a, 0x00, 0x87,
      0x49, 0x00, 0x00, 0x21, 0x34, 0x8a, 0x00, 0x87, 0x4a, 0x00, 0x00, 0x22,
      0x34, 0x8a, 0x80, 0x87, 0x40, 0x00, 0x00, 0x23,
  };
  const std::vector<std::uint8_t> expected_varyings_four_vertex = {
      0x35, 0x82, 0x00, 0x87, 0x80, 0x04, 0x00, 0x00, 0x00, 0x40, 0x35, 0x82,
      0x00, 0x87, 0x81, 0x04, 0x00, 0x00, 0x00, 0x41, 0x34, 0x82, 0x00, 0x87,
      0x00, 0x00, 0x00, 0x42, 0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00,
      0x00, 0x43, 0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
      0x35, 0x82, 0x00, 0x40, 0x81, 0xcc, 0x12, 0x00, 0x00, 0x45, 0x35, 0x82,
      0x00, 0x40, 0x80, 0xcc, 0x12, 0x00, 0x00, 0x44, 0x34, 0x82, 0x00, 0x87,
      0x00, 0x00, 0x00, 0x46, 0x35, 0x82, 0x00, 0x87, 0x8c, 0x01, 0x00, 0x00,
      0x00, 0x47, 0x55, 0xa0, 0x06, 0x08, 0x04, 0xc4, 0x00, 0x00, 0x00, 0x30,
      0x55, 0xa0, 0x06, 0x08, 0x08, 0xc4, 0x00, 0x00, 0x00, 0x30, 0x55, 0xa0,
      0x06, 0x08, 0x0c, 0xc4, 0x00, 0x00, 0x00, 0x30, 0x56, 0xa0, 0x06, 0x08,
      0x10, 0xc4, 0x00, 0x00, 0x00, 0x30, 0xf1, 0xff, 0x44, 0xa0, 0x80, 0x05,
      0x00, 0x00, 0x00, 0xff,
  };
  const std::vector<std::uint8_t> expected_varyings_four_fragment = {
      0x56, 0xa0, 0x00, 0xb0, 0x04, 0xd4, 0x40, 0x10, 0xc0, 0x40, 0x00, 0xff,
      0x02, 0x80, 0x6a, 0xff, 0x56, 0xa0, 0x00, 0xb0, 0x04, 0xc4, 0x40, 0x10,
      0xc0, 0x44, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff, 0x35, 0x82, 0x00, 0x00,
      0xc4, 0xa0, 0x00, 0x00, 0x48, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc5, 0xa1,
      0x00, 0x00, 0x49, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc6, 0xa2, 0x00, 0x00,
      0x4a, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc7, 0xa3, 0x00, 0x00, 0x40, 0xff,
      0x56, 0xa0, 0x00, 0xb0, 0x04, 0xe4, 0x40, 0x10, 0xc0, 0x41, 0x00, 0xff,
      0x02, 0x80, 0x6a, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc8, 0xa1, 0x00, 0x00,
      0x45, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc9, 0xa2, 0x00, 0x00, 0x46, 0xff,
      0x35, 0x82, 0x00, 0x00, 0xca, 0xa3, 0x00, 0x00, 0x47, 0xff, 0x35, 0x82,
      0x00, 0x00, 0xc0, 0xa4, 0x00, 0x00, 0x40, 0xff, 0x56, 0xa0, 0x00, 0xb0,
      0x04, 0xf4, 0x40, 0x10, 0xc0, 0x41, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff,
      0x35, 0x82, 0x00, 0x00, 0xc5, 0xa1, 0x00, 0x00, 0x45, 0xff, 0x35, 0x82,
      0x00, 0x00, 0xc6, 0xa2, 0x00, 0x00, 0x46, 0xff, 0x35, 0x82, 0x00, 0x00,
      0xc7, 0xa3, 0x00, 0x00, 0x47, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc0, 0xa4,
      0x00, 0x00, 0x40, 0xff, 0x34, 0x8a, 0x00, 0x87, 0x45, 0x00, 0x00, 0x20,
      0x34, 0x8a, 0x00, 0x87, 0x46, 0x00, 0x00, 0x21, 0x34, 0x8a, 0x00, 0x87,
      0x47, 0x00, 0x00, 0x22, 0x34, 0x8a, 0x80, 0x87, 0x40, 0x00, 0x00, 0x23,
  };
  Check(FillSolidVertexPcoBinary() == expected_vertex,
        "vertex binary must match Mesa VS_PASSTHROUGH_COMMON");
  Check(FillSolidFragmentPcoBinary() == expected_fragment,
        "fragment binary must match the Mesa PCO fixture generator");
  Check(FillSolidRedHalfAlphaFragmentPcoBinary() == expected_red_half_alpha,
        "red half-alpha binary must match the Mesa PCO fixture generator");
  Check(FillSolidGreenHalfAlphaFragmentPcoBinary() == expected_green_half_alpha,
        "green half-alpha binary must match the Mesa PCO fixture generator");
  Check(TriangleSetupOrangeFragmentPcoBinary() ==
            expected_triangle_setup_orange,
        "triangle_setup orange binary must match the Mesa PCO fixture "
        "generator");
  Check(TriangleSetupCyanFragmentPcoBinary() ==
            expected_triangle_setup_half_culled_cyan,
        "triangle_setup_half_culled cyan binary must match the Mesa PCO "
        "fixture generator");
  Check(AttributeFetchVertexPcoBinary() == expected_attribute_fetch_vertex,
        "attribute-fetch VS must match the exact Mesa PCO binary");
  Check(AttributeFetchTwoAttributeVertexPcoBinary() ==
            expected_attribute_fetch_two_attribute_vertex,
        "two-attribute VS must match the exact Mesa PCO binary");
  Check(AttributeFetchFourAttributeVertexPcoBinary() ==
            expected_attribute_fetch_four_attribute_vertex,
        "four-attribute VS must match the exact Mesa PCO binary");
  Check(AttributeFetchEightAttributeVertexPcoBinary() ==
            expected_attribute_fetch_eight_attribute_vertex,
        "eight-attribute VS must match the exact Mesa PCO binary");
  Check(AttributeFetchGrayFragmentPcoBinary() ==
            expected_attribute_fetch_gray_fragment,
        "attribute-fetch gray FS must match the exact Mesa PCO binary");
  Check(VaryingsOneVertexPcoBinary() == expected_varyings_one_vertex,
        "varyings_shader_1 VS must match the exact Mesa PCO binary");
  Check(VaryingsOneFragmentPcoBinary() == expected_varyings_one_fragment,
        "varyings_shader_1 FS must match the exact Mesa PCO binary");
  Check(VaryingsTwoVertexPcoBinary() == expected_varyings_two_vertex,
        "varyings_shader_2 VS must match the exact Mesa PCO binary");
  Check(VaryingsTwoFragmentPcoBinary() == expected_varyings_two_fragment,
        "varyings_shader_2 FS must match the exact Mesa PCO binary");
  Check(VaryingsFourVertexPcoBinary() == expected_varyings_four_vertex,
        "varyings_shader_4 VS must match the exact Mesa PCO binary");
  Check(VaryingsFourFragmentPcoBinary() == expected_varyings_four_fragment,
        "varyings_shader_4 FS must match the exact Mesa PCO binary");
  Check(VaryingsEightVertexPcoBinary().size() == 176 &&
            Fnv1a64(VaryingsEightVertexPcoBinary()) ==
                UINT64_C(0x3ea4e650a43484ce),
        "varyings_shader_8 VS must match the exact Mesa PCO binary hash");
  Check(VaryingsEightFragmentPcoBinary().size() == 440 &&
            Fnv1a64(VaryingsEightFragmentPcoBinary()) ==
                UINT64_C(0xb1f3b2aa7d58d59d),
        "varyings_shader_8 FS must match the exact Mesa PCO binary hash");
  Check(FillTexNearestVertexPcoBinary().size() == 80 &&
            Fnv1a64(FillTexNearestVertexPcoBinary()) ==
                UINT64_C(0x36c31424e4119557),
        "fill_tex_nearest VS must match the exact Mesa PCO binary hash");
  Check(FillTexNearestFragmentPcoBinary().size() == 184 &&
            Fnv1a64(FillTexNearestFragmentPcoBinary()) ==
                UINT64_C(0x0693891931816150),
        "fill_tex_nearest FS must match the exact Mesa PCO binary hash");
}

void TestDecodeAndExecuteVertex() {
  const auto decoded = Decode(ShaderStage::kVertex, FillSolidVertexPcoBinary());
  Check(decoded.summary.binary_size == 32, "vertex binary size");
  Check(decoded.summary.group_count == 2, "vertex group count");
  Check(decoded.summary.instruction_count == 2, "vertex instruction count");
  Check(decoded.summary.vertex_input_mask == UINT32_C(0x00000007),
        "passthrough VS reads vi0..vi2");
  Check(decoded.summary.vertex_output_mask == 0x000f,
        "vertex output write mask");
  Check(decoded.summary.pixel_output_mask == 0,
        "vertex program does not write pixel outputs");
  Check(decoded.summary.early_hsr_safe == 0,
        "early HSR is a fragment-program property");
  Check(decoded.summary.ends_task == 1, "UVSW ends the vertex task");

  Check(decoded.instructions[0].opcode == PcoOpcode::kUvsWrite,
        "first group is UVSW.write");
  Check(decoded.instructions[0].repeat_count == 3,
        "first group preserves PCO rpt=3");
  Check(decoded.instructions[1].opcode == PcoOpcode::kUvsWriteEmitEndTask,
        "final group is UVSW.write.emit.endtask");

  const auto static_counts = CountPcoInstructions(decoded.instructions, false);
  const auto per_invocation_counts =
      CountPcoInstructions(decoded.instructions, true);
  Check(static_counts.alu == 0 && static_counts.texture == 0 &&
            static_counts.memory == 2,
        "vertex program has two static UVSW memory/export instructions");
  Check(per_invocation_counts.alu == 0 && per_invocation_counts.texture == 0 &&
            per_invocation_counts.memory == 4,
        "vertex repeat expands to four memory/export operations per lane");

  const std::vector<std::uint32_t> vertex_inputs = {
      FloatBits(-0.75f), FloatBits(0.25f), FloatBits(0.5f)};
  const auto result =
      ExecuteVertex(decoded.summary, decoded.instructions, vertex_inputs);
  Check(result.written_mask == 0x000f, "four clip-position outputs written");
  Check(result.outputs[0] == vertex_inputs[0], "raw vi0 copied to output 0");
  Check(result.outputs[1] == vertex_inputs[1], "raw vi1 copied to output 1");
  Check(result.outputs[2] == vertex_inputs[2], "raw vi2 copied to output 2");
  Check(result.outputs[3] == UINT32_C(0x3f800000),
        "public sc64 supplies IEEE-754 1.0 to clip w");
  Check(result.emitted == 1 && result.ended_task == 1,
        "vertex is emitted and the task is ended");

  ExpectFailure(
      [&] {
        const std::vector<std::uint32_t> too_few_inputs = {vertex_inputs[0],
                                                           vertex_inputs[1]};
        (void)ExecuteVertex(decoded.summary, decoded.instructions,
                            too_few_inputs);
      },
      "missing vi2 register");
}

void TestVertexOutput64BitBoundary() {
  /* Keep the exact public passthrough instruction forms, but move the
   * repeat-three UVSW write to VTXOUT60..62 and its final write/emit to
   * VTXOUT63.  This proves the decoder, summary and executor do not retain
   * the earlier 16-register mask limit. */
  auto high_outputs = FillSolidVertexPcoBinary();
  high_outputs[4] = 60;
  high_outputs[20] = 63;
  const auto decoded = Decode(ShaderStage::kVertex, high_outputs);
  Check(decoded.summary.vertex_output_mask ==
            UINT64_C(0xf000000000000000),
        "VTXOUT60..63 use the high four bits of the 64-bit summary mask");

  const std::vector<std::uint32_t> inputs = {
      UINT32_C(0xbf000000), UINT32_C(0x3e800000), UINT32_C(0x3f000000)};
  const auto result =
      ExecuteVertex(decoded.summary, decoded.instructions, inputs);
  Check(result.written_mask == UINT64_C(0xf000000000000000),
        "VTXOUT60..63 use the high four bits of the execution mask");
  Check(result.outputs[60] == inputs[0] &&
            result.outputs[61] == inputs[1] &&
            result.outputs[62] == inputs[2] &&
            result.outputs[63] == UINT32_C(0x3f800000),
        "64-entry VTXOUT storage preserves registers 60 through 63");

  auto overflow = high_outputs;
  overflow[4] = 62;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, overflow); },
                "repeat-three UVSW cannot cross the 64-register boundary");
}

void TestDecodeAndExecuteAttributeFetch() {
  const auto vertex =
      Decode(ShaderStage::kVertex, AttributeFetchVertexPcoBinary());
  Check(vertex.summary.binary_size == 56, "attribute-fetch VS binary size");
  Check(vertex.summary.group_count == 6,
        "attribute-fetch VS has six exact PCO groups");
  Check(vertex.summary.instruction_count == 6,
        "attribute-fetch VS has six semantic instructions");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x00000003),
        "attribute-fetch VS reads exactly vi0 and vi1");
  Check(vertex.summary.vertex_output_mask == UINT16_C(0x000f),
        "attribute-fetch VS exports VTXOUT0..3");
  Check(vertex.summary.ends_task == 1,
        "standalone UVSW.emit.endtask terminates the VS");

  const std::uint32_t expected_offsets[] = {3, 13, 23, 31, 41, 51};
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == expected_offsets[index],
          "attribute-fetch instruction offset");
  }
  for (std::size_t index = 0; index < 4; ++index) {
    const auto &move = vertex.instructions[index];
    Check(move.opcode == PcoOpcode::kMoveBypass,
          "attribute-fetch setup group is MBYP");
    Check(move.target == PcoWriteTarget::kTemporary,
          "attribute-fetch MBYP targets TEMP");
    Check(move.output_index == index,
          "attribute-fetch MBYP writes temp0..temp3 in order");
  }
  Check(vertex.instructions[0].source.bank == PcoRegisterBank::kVertexInput &&
            vertex.instructions[0].source.index == 0,
        "first MBYP reads vi0");
  Check(vertex.instructions[1].source.bank == PcoRegisterBank::kVertexInput &&
            vertex.instructions[1].source.index == 1,
        "second MBYP reads vi1");
  Check(vertex.instructions[2].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[2].source.index == 0,
        "third MBYP reads public sc0");
  Check(vertex.instructions[3].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[3].source.index == 64,
        "fourth MBYP reads public sc64");

  const auto &write = vertex.instructions[4];
  Check(write.opcode == PcoOpcode::kUvsWrite &&
            write.target == PcoWriteTarget::kVertexOutput &&
            write.source.bank == PcoRegisterBank::kTemporary &&
            write.source.index == 0 && write.output_index == 0 &&
            write.repeat_count == 4,
        "UVSW.write repeats temp0..temp3 into VTXOUT0..3");
  const auto &emit = vertex.instructions[5];
  Check(emit.opcode == PcoOpcode::kUvsEmitEndTask &&
            emit.target == PcoWriteTarget::kNone && emit.repeat_count == 1 &&
            emit.end_group == 1,
        "final group is standalone UVSW.emit.endtask");

  const auto static_counts = CountPcoInstructions(vertex.instructions, false);
  const auto dynamic_counts = CountPcoInstructions(vertex.instructions, true);
  Check(static_counts.alu == 4 && static_counts.texture == 0 &&
            static_counts.memory == 2,
        "attribute-fetch VS static classes are ALU4/TEX0/MEM2");
  Check(dynamic_counts.alu == 4 && dynamic_counts.texture == 0 &&
            dynamic_counts.memory == 5,
        "attribute-fetch VS dynamic classes are ALU4/TEX0/MEM5");

  const std::vector<std::uint32_t> vertex_inputs = {
      FloatBits(-0.375f), FloatBits(0.625f)};
  const auto vertex_result =
      ExecuteVertex(vertex.summary, vertex.instructions, vertex_inputs);
  Check(vertex_result.written_mask == UINT16_C(0x000f),
        "attribute-fetch VS writes all clip components");
  Check(vertex_result.outputs[0] == vertex_inputs[0] &&
            vertex_result.outputs[1] == vertex_inputs[1],
        "attribute-fetch VS preserves raw x/y register bits");
  Check(vertex_result.outputs[2] == UINT32_C(0x00000000) &&
            vertex_result.outputs[3] == UINT32_C(0x3f800000),
        "attribute-fetch VS materializes GLES z=0,w=1 defaults");
  Check(vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "standalone UVSW emits the vertex and ends the task");

  ExpectFailure(
      [&] {
        const std::vector<std::uint32_t> missing_vi1 = {vertex_inputs[0]};
        (void)ExecuteVertex(vertex.summary, vertex.instructions, missing_vi1);
      },
      "attribute-fetch lane missing required vi1");

  const auto fragment =
      Decode(ShaderStage::kFragment, AttributeFetchGrayFragmentPcoBinary());
  Check(fragment.summary.binary_size == 48 &&
            fragment.summary.group_count == 4 &&
            fragment.summary.vertex_input_mask == 0 &&
            fragment.summary.pixel_output_mask == 0x0f,
        "attribute-fetch gray FS summary");
  const auto fragment_result =
      ExecuteFragment(fragment.summary, fragment.instructions);
  Check(fragment_result.written_mask == 0x0f,
        "attribute-fetch gray FS writes RGBA");
  for (std::uint32_t component : fragment_result.pixel_outputs) {
    Check(component == UINT32_C(0x3f000000),
          "attribute-fetch gray FS writes exact IEEE-754 0.5");
  }
}

void TestDecodeAndExecuteTwoAttributeFetch() {
  const auto vertex = Decode(ShaderStage::kVertex,
                             AttributeFetchTwoAttributeVertexPcoBinary());
  Check(vertex.summary.binary_size == 56,
        "two-attribute VS binary size is exact");
  Check(vertex.summary.group_count == 6 &&
            vertex.summary.instruction_count == 6,
        "two-attribute VS has six exact PCO groups");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x0000000f),
        "two-attribute VS reads exactly vi0..vi3");
  Check(vertex.summary.vertex_output_mask == UINT16_C(0x000f) &&
            vertex.summary.ends_task == 1,
        "two-attribute VS exports VTXOUT0..3 and ends the task");

  const std::uint32_t expected_offsets[] = {3, 13, 23, 31, 41, 51};
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == expected_offsets[index],
          "two-attribute instruction offset");
  }

  const auto &add_x = vertex.instructions[0];
  Check(add_x.opcode == PcoOpcode::kFloatAdd &&
            add_x.target == PcoWriteTarget::kTemporary &&
            add_x.source_count == 2 &&
            add_x.source.bank == PcoRegisterBank::kVertexInput &&
            add_x.source.index == 0 &&
            add_x.source1.bank == PcoRegisterBank::kVertexInput &&
            add_x.source1.index == 2 && add_x.output_index == 0,
        "first FADD is vi0 + vi2 -> temp0");
  const auto &add_y = vertex.instructions[1];
  Check(add_y.opcode == PcoOpcode::kFloatAdd &&
            add_y.target == PcoWriteTarget::kTemporary &&
            add_y.source_count == 2 &&
            add_y.source.bank == PcoRegisterBank::kVertexInput &&
            add_y.source.index == 1 &&
            add_y.source1.bank == PcoRegisterBank::kVertexInput &&
            add_y.source1.index == 3 && add_y.output_index == 1,
        "second FADD is vi1 + vi3 -> temp1");

  const auto &move_z = vertex.instructions[2];
  const auto &move_w = vertex.instructions[3];
  Check(move_z.opcode == PcoOpcode::kMoveBypass &&
            move_z.source_count == 1 &&
            move_z.source.bank == PcoRegisterBank::kSpecial &&
            move_z.source.index == 0 && move_z.output_index == 2,
        "two-attribute VS moves public sc0 to temp2");
  Check(move_w.opcode == PcoOpcode::kMoveBypass &&
            move_w.source_count == 1 &&
            move_w.source.bank == PcoRegisterBank::kSpecial &&
            move_w.source.index == 65 && move_w.output_index == 3,
        "two-attribute VS moves public sc65 (2.0) to temp3");

  const auto &write = vertex.instructions[4];
  Check(write.opcode == PcoOpcode::kUvsWrite &&
            write.source_count == 1 &&
            write.source.bank == PcoRegisterBank::kTemporary &&
            write.source.index == 0 && write.output_index == 0 &&
            write.repeat_count == 4,
        "two-attribute VS repeats temp0..3 into VTXOUT0..3");
  const auto &emit = vertex.instructions[5];
  Check(emit.opcode == PcoOpcode::kUvsEmitEndTask &&
            emit.source_count == 0 && emit.end_group == 1,
        "two-attribute VS has standalone UVSW emit/endtask");

  const auto static_counts = CountPcoInstructions(vertex.instructions, false);
  const auto dynamic_counts = CountPcoInstructions(vertex.instructions, true);
  Check(static_counts.alu == 4 && static_counts.texture == 0 &&
            static_counts.memory == 2,
        "two-attribute VS static classes are ALU4/TEX0/MEM2");
  Check(dynamic_counts.alu == 4 && dynamic_counts.texture == 0 &&
            dynamic_counts.memory == 5,
        "two-attribute VS dynamic classes are ALU4/TEX0/MEM5");

  const std::vector<std::uint32_t> vertex_inputs = {
      FloatBits(0.25f), FloatBits(-0.125f), FloatBits(0.5f),
      FloatBits(0.375f)};
  const auto result =
      ExecuteVertex(vertex.summary, vertex.instructions, vertex_inputs);
  Check(result.written_mask == UINT16_C(0x000f),
        "two-attribute VS writes all clip components");
  Check(result.outputs[0] == FloatBits(0.75f) &&
            result.outputs[1] == FloatBits(0.25f),
        "PCO FADD executes both attribute sums");
  Check(result.outputs[2] == UINT32_C(0x00000000) &&
            result.outputs[3] == UINT32_C(0x40000000),
        "two-attribute VS exports z=0 and w=2 from sc0/sc65");
  Check(result.emitted == 1 && result.ended_task == 1,
        "two-attribute VS emits and ends the task");

  ExpectFailure(
      [&] {
        const std::vector<std::uint32_t> missing_vi3(vertex_inputs.begin(),
                                                     vertex_inputs.end() - 1);
        (void)ExecuteVertex(vertex.summary, vertex.instructions, missing_vi3);
      },
      "two-attribute lane missing required vi3");
}

void TestDecodeAndExecuteFourAttributeFetch() {
  const auto vertex = Decode(ShaderStage::kVertex,
                             AttributeFetchFourAttributeVertexPcoBinary());
  Check(vertex.summary.binary_size == 96,
        "four-attribute VS binary size is exact");
  Check(vertex.summary.group_count == 10 &&
            vertex.summary.instruction_count == 10,
        "four-attribute VS has ten exact PCO groups");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x000000ff),
        "four-attribute VS reads exactly vi0..vi7");
  Check(vertex.summary.vertex_output_mask == UINT16_C(0x000f) &&
            vertex.summary.ends_task == 1,
        "four-attribute VS exports VTXOUT0..3 and ends the task");

  const std::uint32_t expected_offsets[] = {3, 13, 23, 33, 43,
                                             53, 63, 71, 81, 91};
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == expected_offsets[index],
          "four-attribute instruction offset");
  }

  struct ExpectedFadd {
    PcoRegisterBank source0_bank;
    std::uint16_t source0_index;
    std::uint16_t source1_index;
    std::uint16_t destination_index;
  };
  const ExpectedFadd expected_fadds[] = {
      {PcoRegisterBank::kVertexInput, 0, 2, 0},
      {PcoRegisterBank::kVertexInput, 1, 3, 1},
      {PcoRegisterBank::kTemporary, 0, 4, 0},
      {PcoRegisterBank::kTemporary, 1, 5, 1},
      {PcoRegisterBank::kTemporary, 0, 6, 2},
      {PcoRegisterBank::kTemporary, 1, 7, 3},
  };
  for (std::size_t index = 0; index < std::size(expected_fadds); ++index) {
    const auto &instruction = vertex.instructions[index];
    const auto &expected = expected_fadds[index];
    Check(instruction.opcode == PcoOpcode::kFloatAdd &&
              instruction.target == PcoWriteTarget::kTemporary &&
              instruction.source_count == 2 &&
              instruction.source.bank == expected.source0_bank &&
              instruction.source.index == expected.source0_index &&
              instruction.source1.bank == PcoRegisterBank::kVertexInput &&
              instruction.source1.index == expected.source1_index &&
              instruction.output_index == expected.destination_index,
          "four-attribute FADD dataflow is exact");
  }

  Check(vertex.instructions[6].opcode == PcoOpcode::kMoveBypass &&
            vertex.instructions[6].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[6].source.index == 0 &&
            vertex.instructions[6].output_index == 4,
        "four-attribute VS moves sc0 to temp4");
  Check(vertex.instructions[7].opcode == PcoOpcode::kMoveBypass &&
            vertex.instructions[7].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[7].source.index == 66 &&
            vertex.instructions[7].output_index == 5,
        "four-attribute VS moves public sc66 (4.0) to temp5");
  Check(vertex.instructions[8].opcode == PcoOpcode::kUvsWrite &&
            vertex.instructions[8].source.bank == PcoRegisterBank::kTemporary &&
            vertex.instructions[8].source.index == 2 &&
            vertex.instructions[8].output_index == 0 &&
            vertex.instructions[8].repeat_count == 4,
        "four-attribute VS exports temp2..temp5");
  Check(vertex.instructions[9].opcode == PcoOpcode::kUvsEmitEndTask &&
            vertex.instructions[9].source_count == 0 &&
            vertex.instructions[9].end_group == 1,
        "four-attribute VS has standalone UVSW emit/endtask");

  const auto static_counts = CountPcoInstructions(vertex.instructions, false);
  const auto dynamic_counts = CountPcoInstructions(vertex.instructions, true);
  Check(static_counts.alu == 8 && static_counts.texture == 0 &&
            static_counts.memory == 2,
        "four-attribute VS static classes are ALU8/TEX0/MEM2");
  Check(dynamic_counts.alu == 8 && dynamic_counts.texture == 0 &&
            dynamic_counts.memory == 5,
        "four-attribute VS dynamic classes are ALU8/TEX0/MEM5");

  const std::vector<std::uint32_t> vertex_inputs = {
      FloatBits(0.25f),  FloatBits(-0.125f), FloatBits(0.5f),
      FloatBits(0.375f), FloatBits(-0.125f), FloatBits(0.25f),
      FloatBits(0.375f), FloatBits(-0.5f),
  };
  const auto result =
      ExecuteVertex(vertex.summary, vertex.instructions, vertex_inputs);
  Check(result.written_mask == UINT16_C(0x000f),
        "four-attribute VS writes all clip components");
  Check(result.outputs[0] == FloatBits(1.0f) &&
            result.outputs[1] == FloatBits(0.0f),
        "four-attribute TEMP RMW FADD chain executes in order");
  Check(result.outputs[2] == UINT32_C(0x00000000) &&
            result.outputs[3] == UINT32_C(0x40800000),
        "four-attribute VS exports z=0 and w=4 from sc0/sc66");
  Check(result.emitted == 1 && result.ended_task == 1,
        "four-attribute VS emits and ends the task");

  ExpectFailure(
      [&] {
        const std::vector<std::uint32_t> missing_vi7(vertex_inputs.begin(),
                                                     vertex_inputs.end() - 1);
        (void)ExecuteVertex(vertex.summary, vertex.instructions, missing_vi7);
      },
      "four-attribute lane missing required vi7");
}

void TestDecodeAndExecuteEightAttributeFetch() {
  const auto vertex = Decode(ShaderStage::kVertex,
                             AttributeFetchEightAttributeVertexPcoBinary());
  Check(vertex.summary.binary_size == 176,
        "eight-attribute VS binary size is exact");
  Check(vertex.summary.group_count == 18 &&
            vertex.summary.instruction_count == 18,
        "eight-attribute VS has eighteen exact PCO groups");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x0000ffff),
        "eight-attribute VS reads exactly vi0..vi15");
  Check(vertex.summary.vertex_output_mask == UINT16_C(0x000f) &&
            vertex.summary.ends_task == 1,
        "eight-attribute VS exports VTXOUT0..3 and ends the task");

  const std::uint32_t expected_offsets[] = {
      3,   13,  23,  33,  43,  53,  63,  73,  83,
      93,  103, 113, 123, 133, 143, 151, 161, 171,
  };
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == expected_offsets[index],
          "eight-attribute instruction offset");
  }

  for (std::size_t index = 0; index < 14; ++index) {
    const auto &instruction = vertex.instructions[index];
    const bool first_pair = index < 2;
    const std::uint16_t source0_index =
        first_pair ? static_cast<std::uint16_t>(index)
                   : static_cast<std::uint16_t>(index & 1U);
    const std::uint16_t destination_index =
        index < 12 ? static_cast<std::uint16_t>(index & 1U)
                   : static_cast<std::uint16_t>(index - 10U);
    Check(instruction.opcode == PcoOpcode::kFloatAdd &&
              instruction.target == PcoWriteTarget::kTemporary &&
              instruction.source_count == 2 &&
              instruction.source.bank ==
                  (first_pair ? PcoRegisterBank::kVertexInput
                              : PcoRegisterBank::kTemporary) &&
              instruction.source.index == source0_index &&
              instruction.source1.bank == PcoRegisterBank::kVertexInput &&
              instruction.source1.index == index + 2U &&
              instruction.output_index == destination_index,
          "eight-attribute FADD TEMP chain is exact");
  }

  Check(vertex.instructions[14].opcode == PcoOpcode::kMoveBypass &&
            vertex.instructions[14].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[14].source.index == 0 &&
            vertex.instructions[14].output_index == 4,
        "eight-attribute VS moves sc0 to temp4");
  Check(vertex.instructions[15].opcode == PcoOpcode::kMoveBypass &&
            vertex.instructions[15].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[15].source.index == 67 &&
            vertex.instructions[15].output_index == 5,
        "eight-attribute VS moves public sc67 (8.0) to temp5");
  Check(vertex.instructions[16].opcode == PcoOpcode::kUvsWrite &&
            vertex.instructions[16].source.bank ==
                PcoRegisterBank::kTemporary &&
            vertex.instructions[16].source.index == 2 &&
            vertex.instructions[16].output_index == 0 &&
            vertex.instructions[16].repeat_count == 4,
        "eight-attribute VS exports temp2..temp5");
  Check(vertex.instructions[17].opcode == PcoOpcode::kUvsEmitEndTask &&
            vertex.instructions[17].source_count == 0 &&
            vertex.instructions[17].end_group == 1,
        "eight-attribute VS has standalone UVSW emit/endtask");

  const auto static_counts = CountPcoInstructions(vertex.instructions, false);
  const auto dynamic_counts = CountPcoInstructions(vertex.instructions, true);
  Check(static_counts.alu == 16 && static_counts.texture == 0 &&
            static_counts.memory == 2,
        "eight-attribute VS static classes are ALU16/TEX0/MEM2");
  Check(dynamic_counts.alu == 16 && dynamic_counts.texture == 0 &&
            dynamic_counts.memory == 5,
        "eight-attribute VS dynamic classes are ALU16/TEX0/MEM5");

  const std::vector<std::uint32_t> vertex_inputs = {
      FloatBits(0.25f),   FloatBits(-0.125f), FloatBits(0.5f),
      FloatBits(0.375f),  FloatBits(-0.125f), FloatBits(0.25f),
      FloatBits(0.375f),  FloatBits(-0.5f),   FloatBits(0.25f),
      FloatBits(0.125f),  FloatBits(-0.5f),   FloatBits(0.125f),
      FloatBits(0.125f),  FloatBits(0.25f),   FloatBits(0.125f),
      FloatBits(-0.5f),
  };
  const auto result =
      ExecuteVertex(vertex.summary, vertex.instructions, vertex_inputs);
  Check(result.written_mask == UINT16_C(0x000f),
        "eight-attribute VS writes all clip components");
  Check(result.outputs[0] == FloatBits(1.0f) &&
            result.outputs[1] == FloatBits(0.0f),
        "eight-attribute TEMP RMW FADD chain executes in order");
  Check(result.outputs[2] == UINT32_C(0x00000000) &&
            result.outputs[3] == UINT32_C(0x41000000),
        "eight-attribute VS exports z=0 and w=8 from sc0/sc67");
  Check(result.emitted == 1 && result.ended_task == 1,
        "eight-attribute VS emits and ends the task");

  ExpectFailure(
      [&] {
        const std::vector<std::uint32_t> missing_vi15(vertex_inputs.begin(),
                                                      vertex_inputs.end() - 1);
        (void)ExecuteVertex(vertex.summary, vertex.instructions, missing_vi15);
      },
      "eight-attribute lane missing required vi15");
}

void TestIntegerFloatAddRoundingAndClasses() {
  const auto decoded = Decode(ShaderStage::kVertex,
                              AttributeFetchTwoAttributeVertexPcoBinary());
  const auto execute = [&](std::uint32_t x0, std::uint32_t y0,
                           std::uint32_t x1, std::uint32_t y1) {
    return ExecuteVertex(decoded.summary, decoded.instructions,
                         {x0, y0, x1, y1});
  };

  /* 1.0 + 2^-24 is exactly halfway. The even 1.0 mantissa stays down. */
  auto result = execute(UINT32_C(0x3f800000), UINT32_C(0x00000000),
                        UINT32_C(0x33800000), UINT32_C(0x00000000));
  Check(result.outputs[0] == UINT32_C(0x3f800000),
        "FADD RNE tie keeps an even low significand");

  /* The adjacent odd mantissa rounds the same half-way increment upward. */
  result = execute(UINT32_C(0x3f800001), UINT32_C(0x00000000),
                   UINT32_C(0x33800000), UINT32_C(0x00000000));
  Check(result.outputs[0] == UINT32_C(0x3f800002),
        "FADD RNE tie increments an odd low significand");

  result = execute(UINT32_C(0x3fc00000), UINT32_C(0x3fc00000),
                   UINT32_C(0xbf000000), UINT32_C(0xbfc00000));
  Check(result.outputs[0] == UINT32_C(0x3f800000),
        "FADD opposite-sign subtraction produces exact 1.0");
  Check(result.outputs[1] == UINT32_C(0x00000000),
        "FADD exact normal cancellation produces positive zero");

  result = execute(UINT32_C(0x80000000), UINT32_C(0x00000000),
                   UINT32_C(0x80000000), UINT32_C(0x80000000));
  Check(result.outputs[0] == UINT32_C(0x80000000),
        "FADD preserves -0 only for -0 plus -0");
  Check(result.outputs[1] == UINT32_C(0x00000000),
        "FADD opposite-signed zeros produce positive zero under RNE");

  for (const auto &[input, expected] : {
           std::pair{UINT32_C(0x7fc00000), UINT32_C(0x7fc00000)},
           std::pair{UINT32_C(0x7f800000), UINT32_C(0x7f800000)},
           std::pair{UINT32_C(0xff800000), UINT32_C(0xff800000)},
           std::pair{UINT32_C(0x00000001), UINT32_C(0x00000001)},
           std::pair{UINT32_C(0x807fffff), UINT32_C(0x807fffff)},
       }) {
    result = execute(input, UINT32_C(0x00000000),
                     UINT32_C(0x00000000), UINT32_C(0x00000000));
    Check(result.outputs[0] == expected,
          "FADD implements IEEE special/subnormal input semantics");
  }

  result = execute(UINT32_C(0x7f7fffff), UINT32_C(0x00000000),
                   UINT32_C(0x7f7fffff), UINT32_C(0x00000000));
  Check(result.outputs[0] == UINT32_C(0x7f800000),
        "FADD finite overflow produces positive infinity");
  result = execute(UINT32_C(0x00800001), UINT32_C(0x00000000),
                   UINT32_C(0x80800000), UINT32_C(0x00000000));
  Check(result.outputs[0] == UINT32_C(0x00000001),
        "FADD finite cancellation produces a subnormal result");
}

void TestDecodeAndExecuteFragment() {
  const auto decoded =
      Decode(ShaderStage::kFragment, FillSolidFragmentPcoBinary());
  Check(decoded.summary.binary_size == 48, "fragment binary size");
  Check(decoded.summary.group_count == 4, "fragment group count");
  Check(decoded.summary.instruction_count == 4, "fragment instruction count");
  Check(decoded.summary.vertex_output_mask == 0,
        "fragment program does not write vertex outputs");
  Check(decoded.summary.pixel_output_mask == 0x0f,
        "fragment program writes pixout0..3");
  Check(decoded.summary.early_hsr_safe == 1,
        "MBYP-only shader cannot discard or write depth");
  Check(decoded.summary.ends_task == 0,
        "fragment program does not contain UVSW endtask");

  const auto static_counts = CountPcoInstructions(decoded.instructions, false);
  const auto per_invocation_counts =
      CountPcoInstructions(decoded.instructions, true);
  Check(static_counts.alu == 4 && static_counts.texture == 0 &&
            static_counts.memory == 0,
        "fragment program has four static MBYP ALU/move instructions");
  Check(per_invocation_counts.alu == 4 && per_invocation_counts.texture == 0 &&
            per_invocation_counts.memory == 0,
        "fragment program executes four ALU/move operations per lane");

  const auto result = ExecuteFragment(decoded.summary, decoded.instructions);
  Check(result.written_mask == 0x0f, "all RGBA pixel outputs written");
  Check(result.pixel_outputs[0] == UINT32_C(0x3f800000), "R = 1.0");
  Check(result.pixel_outputs[1] == UINT32_C(0x00000000), "G = 0.0");
  Check(result.pixel_outputs[2] == UINT32_C(0x00000000), "B = 0.0");
  Check(result.pixel_outputs[3] == UINT32_C(0x3f800000), "A = 1.0");
}

void TestDecodeAndExecuteHalfAlphaFragments() {
  const auto black =
      Decode(ShaderStage::kFragment, FillSolidBlackFragmentPcoBinary());
  Check(black.summary.binary_size == 46 && black.summary.group_count == 4,
        "black opaque fixture has four exact PCO groups");
  const auto black_result = ExecuteFragment(black.summary, black.instructions);
  Check(black_result.written_mask == 0x0f,
        "black opaque fixture writes all RGBA outputs");
  Check(black_result.pixel_outputs[0] == UINT32_C(0x00000000),
        "black opaque R = 0.0");
  Check(black_result.pixel_outputs[1] == UINT32_C(0x00000000),
        "black opaque G = 0.0");
  Check(black_result.pixel_outputs[2] == UINT32_C(0x00000000),
        "black opaque B = 0.0");
  Check(black_result.pixel_outputs[3] == UINT32_C(0x3f800000),
        "black opaque A = 1.0");

  const auto red =
      Decode(ShaderStage::kFragment, FillSolidRedHalfAlphaFragmentPcoBinary());
  Check(red.summary.binary_size == 48 && red.summary.group_count == 4,
        "red half-alpha fixture has four exact PCO groups");
  Check(red.instructions[3].source.index == 75,
        "red half-alpha fixture decodes Mesa public sc75");
  const auto red_result = ExecuteFragment(red.summary, red.instructions);
  Check(red_result.written_mask == 0x0f,
        "red half-alpha fixture writes all RGBA outputs");
  Check(red_result.pixel_outputs[0] == UINT32_C(0x3f800000),
        "red half-alpha R = 1.0");
  Check(red_result.pixel_outputs[1] == UINT32_C(0x00000000),
        "red half-alpha G = 0.0");
  Check(red_result.pixel_outputs[2] == UINT32_C(0x00000000),
        "red half-alpha B = 0.0");
  Check(red_result.pixel_outputs[3] == UINT32_C(0x3f000000),
        "red half-alpha A = exact IEEE-754 0.5 from sc75");

  const auto green = Decode(ShaderStage::kFragment,
                            FillSolidGreenHalfAlphaFragmentPcoBinary());
  Check(green.summary.binary_size == 48 && green.summary.group_count == 4,
        "green half-alpha fixture has four exact PCO groups");
  Check(green.instructions[3].source.index == 75,
        "green half-alpha fixture decodes Mesa public sc75");
  const auto green_result = ExecuteFragment(green.summary, green.instructions);
  Check(green_result.written_mask == 0x0f,
        "green half-alpha fixture writes all RGBA outputs");
  Check(green_result.pixel_outputs[0] == UINT32_C(0x00000000),
        "green half-alpha R = 0.0");
  Check(green_result.pixel_outputs[1] == UINT32_C(0x3f800000),
        "green half-alpha G = 1.0");
  Check(green_result.pixel_outputs[2] == UINT32_C(0x00000000),
        "green half-alpha B = 0.0");
  Check(green_result.pixel_outputs[3] == UINT32_C(0x3f000000),
        "green half-alpha A = exact IEEE-754 0.5 from sc75");
}

void TestDecodeAndExecuteTriangleSetupOrange() {
  const auto decoded =
      Decode(ShaderStage::kFragment, TriangleSetupOrangeFragmentPcoBinary());
  Check(decoded.summary.binary_size == 48, "triangle_setup orange binary size");
  Check(decoded.summary.group_count == 4, "triangle_setup orange group count");
  Check(decoded.summary.instruction_count == 4,
        "triangle_setup orange instruction count");
  Check(decoded.summary.pixel_output_mask == 0x0f,
        "triangle_setup orange writes pixout0..3");
  Check(decoded.instructions[1].source.index == 75,
        "triangle_setup orange G decodes Mesa public sc75");

  const auto result = ExecuteFragment(decoded.summary, decoded.instructions);
  Check(result.written_mask == 0x0f,
        "triangle_setup orange writes all RGBA outputs");
  Check(result.pixel_outputs[0] == UINT32_C(0x3f800000),
        "triangle_setup orange R = exact IEEE-754 1.0");
  Check(result.pixel_outputs[1] == UINT32_C(0x3f000000),
        "triangle_setup orange G = exact IEEE-754 0.5 from sc75");
  Check(result.pixel_outputs[2] == UINT32_C(0x00000000),
        "triangle_setup orange B = exact IEEE-754 0.0");
  Check(result.pixel_outputs[3] == UINT32_C(0x3f800000),
        "triangle_setup orange A = exact IEEE-754 1.0");
}

void TestDecodeAndExecuteTriangleSetupHalfCulledCyan() {
  const auto decoded =
      Decode(ShaderStage::kFragment, TriangleSetupCyanFragmentPcoBinary());
  Check(decoded.summary.binary_size == 48,
        "triangle_setup_half_culled cyan binary size");
  Check(decoded.summary.group_count == 4,
        "triangle_setup_half_culled cyan group count");
  Check(decoded.summary.instruction_count == 4,
        "triangle_setup_half_culled cyan instruction count");
  Check(decoded.summary.pixel_output_mask == 0x0f,
        "triangle_setup_half_culled cyan writes pixout0..3");
  Check(decoded.instructions[1].source.index == 75 &&
            decoded.instructions[2].source.index == 75,
        "triangle_setup_half_culled cyan G/B decode Mesa public sc75");

  const auto result = ExecuteFragment(decoded.summary, decoded.instructions);
  Check(result.written_mask == 0x0f,
        "triangle_setup_half_culled cyan writes all RGBA outputs");
  Check(result.pixel_outputs[0] == UINT32_C(0x00000000),
        "triangle_setup_half_culled cyan R = exact IEEE-754 0.0");
  Check(result.pixel_outputs[1] == UINT32_C(0x3f000000),
        "triangle_setup_half_culled cyan G = exact IEEE-754 0.5 from sc75");
  Check(result.pixel_outputs[2] == UINT32_C(0x3f000000),
        "triangle_setup_half_culled cyan B = exact IEEE-754 0.5 from sc75");
  Check(result.pixel_outputs[3] == UINT32_C(0x3f800000),
        "triangle_setup_half_culled cyan A = exact IEEE-754 1.0");
}

void TestDecodeFailsClosed() {
  ExpectFailure(
      [] { (void)Decode(ShaderStage::kFragment, FillSolidVertexPcoBinary()); },
      "vertex bytes decoded as fragment stage");
  ExpectFailure(
      [] { (void)Decode(ShaderStage::kVertex, FillSolidFragmentPcoBinary()); },
      "fragment bytes decoded as vertex stage");

  auto bad_opcode = FillSolidFragmentPcoBinary();
  bad_opcode[3] = 0x86;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_opcode); },
                "unknown main opcode");

  auto bad_destination = FillSolidFragmentPcoBinary();
  bad_destination[9] = 0x24;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_destination); },
                "unsupported pixout4 destination");

  auto bad_constant = FillSolidFragmentPcoBinary();
  bad_constant[5] = 0x02; // sc128 rather than the supported public sc64.
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_constant); },
                "unsupported special constant");

  auto nearby_half_constant = FillSolidRedHalfAlphaFragmentPcoBinary();
  nearby_half_constant[36] = 0x8a; // sc74 is not part of the exact subset.
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, nearby_half_constant); },
      "nearby but unsupported special constant must fail closed");

  auto bad_orange_constant = TriangleSetupOrangeFragmentPcoBinary();
  bad_orange_constant[14] = 0x8a; // sc74 is not part of the exact subset.
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_orange_constant); },
      "triangle_setup orange nearby special constant must fail closed");

  auto bad_cyan_constant = TriangleSetupCyanFragmentPcoBinary();
  bad_cyan_constant[12] = 0x8a; // sc74 is not part of the exact subset.
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_cyan_constant); },
      "triangle_setup_half_culled cyan nearby special constant must fail "
      "closed");

  auto bad_iss = FillSolidVertexPcoBinary();
  bad_iss[9] = 0x31;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, bad_iss); },
                "unsupported internal-source selection");

  auto bad_padding = FillSolidFragmentPcoBinary();
  bad_padding[26] = 0xf2;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_padding); },
                "incorrect alignment-padding marker");

  auto no_end = FillSolidFragmentPcoBinary();
  no_end[34] = 0x00;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, no_end); },
                "missing final .end bit");

  auto truncated = FillSolidVertexPcoBinary();
  truncated.pop_back();
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, truncated); },
                "truncated final group");

  auto trailing = FillSolidFragmentPcoBinary();
  trailing.push_back(0xff);
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, trailing); },
                "bytes after the final group");
}

void TestAttributeFetchFailsClosed() {
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kFragment,
                     AttributeFetchVertexPcoBinary());
      },
      "attribute-fetch VS decoded as fragment stage");

  auto bad_move_header = AttributeFetchVertexPcoBinary();
  bad_move_header[1] = 0x8a;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_move_header); },
      "attribute-fetch MBYP output-load-check mutation");

  auto bad_move_opcode = AttributeFetchVertexPcoBinary();
  bad_move_opcode[3] = 0x86;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_move_opcode); },
      "attribute-fetch unknown main opcode");

  auto bad_move_source_bank = AttributeFetchVertexPcoBinary();
  bad_move_source_bank[4] = 0xc0;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_move_source_bank); },
      "attribute-fetch unsupported coefficient source bank");

  auto bad_move_upper = AttributeFetchVertexPcoBinary();
  bad_move_upper[7] = 0x01;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_move_upper); },
      "attribute-fetch unsupported MBYP upper source");

  auto bad_move_iss = AttributeFetchVertexPcoBinary();
  bad_move_iss[8] = 0x01;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, bad_move_iss); },
                "attribute-fetch unsupported MBYP ISS selection");

  auto bad_temp_destination = AttributeFetchVertexPcoBinary();
  bad_temp_destination[9] = 0x44;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_temp_destination); },
      "attribute-fetch MBYP destination outside temp0..temp3");

  auto bad_uvsw_repeat = AttributeFetchVertexPcoBinary();
  bad_uvsw_repeat[40] = 0x04;
  const auto scalar_uvsw = Decode(ShaderStage::kVertex, bad_uvsw_repeat);
  Check(scalar_uvsw.instructions[4].repeat_count == 3,
        "generic UVSW accepts a bounded short temporary export");

  auto bad_temp_source = AttributeFetchVertexPcoBinary();
  bad_temp_source[43] = 0xc1;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_temp_source); },
      "UVSW repeat cannot read beyond initialized temporary registers");

  auto bad_emit_opcode = AttributeFetchVertexPcoBinary();
  bad_emit_opcode[51] = 0x04;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_emit_opcode); },
      "attribute-fetch unknown standalone UVSW opcode");

  for (std::size_t offset : {std::size_t{52}, std::size_t{53},
                             std::size_t{54}}) {
    auto bad_blank_field = AttributeFetchVertexPcoBinary();
    bad_blank_field[offset] = 0x01;
    ExpectFailure(
        [&] { (void)Decode(ShaderStage::kVertex, bad_blank_field); },
        "attribute-fetch standalone UVSW non-zero placeholder field");
  }

  auto bad_word_padding = AttributeFetchVertexPcoBinary();
  bad_word_padding[55] = 0xfe;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_word_padding); },
      "attribute-fetch standalone UVSW word padding");

  auto truncated = AttributeFetchVertexPcoBinary();
  truncated.pop_back();
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, truncated); },
                "truncated attribute-fetch VS");

  auto decoded =
      Decode(ShaderStage::kVertex, AttributeFetchVertexPcoBinary());
  decoded.instructions[4].source.index = 4;
  const std::vector<std::uint32_t> inputs = {FloatBits(0.0f), FloatBits(0.0f)};
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "UVSW read of an unwritten temporary register");

  decoded = Decode(ShaderStage::kVertex, AttributeFetchVertexPcoBinary());
  decoded.summary.vertex_input_mask = UINT32_C(0x00000001);
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "tampered vertex-input summary mask");
}

void TestDecodeAndExecuteScalarSource0Floor() {
  /* Terrain D1 FS group 10 is the exact public scalar encoding
   *   fadd ft0, r5.flr, sc0; w0=r7
   * surrounded here by canonical immediate/PIXOUT groups so the real group
   * can be decoded and executed without carrying the 14,872-byte shader. */
  const auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 a0 bf 00 00 45 ff
35 82 00 01 c5 80 00 00 47 ff
34 8a 00 87 47 00 00 20
34 8a 00 87 47 00 00 21
34 8a 00 87 47 00 00 22
34 8a 80 87 47 00 00 23
)hex");
  const auto decoded = Decode(ShaderStage::kFragment, binary);
  Check(decoded.summary.binary_size == 54 &&
            decoded.summary.group_count == 6 &&
            decoded.summary.pixel_output_mask == 0x0f,
        "Terrain scalar-floor canonical fragment envelope");
  const auto &floor = decoded.instructions[1];
  Check(floor.binary_offset == 15 && floor.opcode == PcoOpcode::kFloatAdd &&
            floor.source0_floor == 1 && floor.source_count == 2 &&
            floor.source.bank == PcoRegisterBank::kTemporary &&
            floor.source.index == 5 &&
            floor.source1.bank == PcoRegisterBank::kSpecial &&
            floor.source1.index == 0 && floor.output_index == 7,
        "Terrain D1 exact 35 82 00 01 c5 80 00 00 47 group semantics");
  const auto execution = ExecuteFragment(decoded.summary, decoded.instructions);
  Check(execution.written_mask == 0x0f,
        "scalar-floor fragment writes the complete color");
  for (const std::uint32_t component : execution.pixel_outputs) {
    Check(component == FloatBits(-2.0F),
          "fragment FADD applies floor(source0) before adding sc0");
  }

  auto unsupported_modifier = binary;
  unsupported_modifier[15] = 0x03;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, unsupported_modifier); },
      "unverified scalar source-modifier combination");

  auto tampered = decoded;
  tampered.instructions[1].source0_floor = 2;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(tampered.summary, tampered.instructions);
      },
      "non-Boolean decoded source0-floor modifier");
}

void TestDecodeAndExecuteSpecialConstant153() {
  /* Pinned Mesa pco_const_imms.c maps sc153 to exact binary32 0x3e2aaaab.
   * The FMUL is Terrain D1 FS group 18 verbatim; the later FADD group is also
   * capture-derived and proves that the constant gate is source/op neutral. */
  const auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 c0 40 00 00 4c ff
36 82 00 40 99 ec 80 02 00 00 4c ff
86 92 40 13 00 00 80 3f 00 00 48 ff
36 82 00 00 c8 d9 80 01 00 00 50 ff
34 8a 00 87 4c 00 00 20
34 8a 00 87 4c 00 00 21
34 8a 00 87 50 00 00 22
34 8a 80 87 50 00 00 23
)hex");
  const auto decoded = Decode(ShaderStage::kFragment, binary);
  Check(decoded.summary.binary_size == 80 &&
            decoded.summary.group_count == 8 &&
            decoded.summary.pixel_output_mask == 0x0f,
        "Terrain sc153 canonical fragment envelope");
  const auto &multiply = decoded.instructions[1];
  Check(multiply.binary_offset == 15 &&
            multiply.opcode == PcoOpcode::kFloatMultiply &&
            multiply.source.bank == PcoRegisterBank::kSpecial &&
            multiply.source.index == 153 &&
            multiply.source1.bank == PcoRegisterBank::kTemporary &&
            multiply.source1.index == 12 && multiply.output_index == 12,
        "Terrain exact 36 82 00 40 99 ec 80 02 00 00 4c group semantics");
  const auto &add = decoded.instructions[3];
  Check(add.opcode == PcoOpcode::kFloatAdd &&
            add.source.bank == PcoRegisterBank::kTemporary &&
            add.source.index == 8 &&
            add.source1.bank == PcoRegisterBank::kSpecial &&
            add.source1.index == 153 && add.output_index == 16,
        "capture-derived FADD accepts sc153 in the other source slot");

  const auto execution = ExecuteFragment(decoded.summary, decoded.instructions);
  Check(execution.pixel_outputs[0] == UINT32_C(0x3f800000) &&
            execution.pixel_outputs[1] == UINT32_C(0x3f800000) &&
            execution.pixel_outputs[2] == UINT32_C(0x3f955555) &&
            execution.pixel_outputs[3] == UINT32_C(0x3f955555),
        "sc153 executes as exact bits 0x3e2aaaab for FMUL and FADD");

  auto unsupported_source0 = binary;
  unsupported_source0[16] = 0x9a;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, unsupported_source0); },
      "neighboring special constant sc154 in source0");
  auto unsupported_source1 = binary;
  unsupported_source1[41] = 0xda;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, unsupported_source1); },
      "neighboring special constant sc154 in source1");
}

void TestTwoAttributeFetchFailsClosed() {
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kFragment,
                     AttributeFetchTwoAttributeVertexPcoBinary());
      },
      "two-attribute VS decoded as fragment stage");

  for (std::uint8_t main_mutation : {std::uint8_t{0x02},
                                     std::uint8_t{0x20},
                                     std::uint8_t{0x43},
                                     std::uint8_t{0x80}}) {
    auto binary = AttributeFetchTwoAttributeVertexPcoBinary();
    binary[3] = main_mutation;
    ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, binary); },
                  "modified/later FADD main encoding");
  }

  auto saturate_fadd = AttributeFetchTwoAttributeVertexPcoBinary();
  saturate_fadd[3] = 0x10;
  const auto saturated = Decode(ShaderStage::kVertex, saturate_fadd);
  Check(saturated.instructions[0].opcode == PcoOpcode::kFloatAdd &&
            saturated.instructions[0].saturate == 1,
        "public FADD saturate modifier remains orthogonal to FADD");

  auto source0_floor_fadd = AttributeFetchTwoAttributeVertexPcoBinary();
  source0_floor_fadd[3] = 0x01;
  const auto floor_fadd = Decode(ShaderStage::kVertex, source0_floor_fadd);
  Check(floor_fadd.instructions[0].opcode == PcoOpcode::kFloatAdd &&
            floor_fadd.instructions[0].source0_floor == 1,
        "public FADD source0-floor modifier remains orthogonal to FADD");
  const std::vector<std::uint32_t> floor_inputs = {
      FloatBits(1.75F), FloatBits(-0.125F), FloatBits(0.5F),
      FloatBits(0.375F)};
  Check(ExecuteVertex(floor_fadd.summary, floor_fadd.instructions, floor_inputs)
                .outputs[0] == FloatBits(1.5F),
        "vertex FADD applies floor(source0) before the binary add");

  auto source0_absolute_fadd = AttributeFetchTwoAttributeVertexPcoBinary();
  source0_absolute_fadd[3] = 0x04;
  const auto absolute_fadd =
      Decode(ShaderStage::kVertex, source0_absolute_fadd);
  Check(absolute_fadd.instructions[0].opcode == PcoOpcode::kFloatAdd &&
            absolute_fadd.instructions[0].source0_absolute == 1 &&
            absolute_fadd.instructions[0].source1_absolute == 0,
        "public FADD source0-absolute modifier remains orthogonal to FADD");
  const std::vector<std::uint32_t> absolute_fadd_inputs = {
      FloatBits(-0.75F), FloatBits(-0.125F), FloatBits(0.5F),
      FloatBits(0.375F)};
  Check(ExecuteVertex(absolute_fadd.summary, absolute_fadd.instructions,
                      absolute_fadd_inputs)
                .outputs[0] == FloatBits(1.25F),
        "vertex FADD clears source0's sign before the binary add");

  auto source0_negated_fadd = AttributeFetchTwoAttributeVertexPcoBinary();
  source0_negated_fadd[3] = 0x08;
  Check(Decode(ShaderStage::kVertex, source0_negated_fadd)
                .instructions[0]
                .opcode == PcoOpcode::kFloatAddNegateSource0,
        "public FADD source0-negate modifier has a distinct semantic opcode");

  auto generic_fmul = AttributeFetchTwoAttributeVertexPcoBinary();
  generic_fmul[3] = 0x40;
  Check(Decode(ShaderStage::kVertex, generic_fmul).instructions[0].opcode ==
            PcoOpcode::kFloatMultiply,
        "the same canonical two-lower-source group decodes public FMUL");

  auto source1_absolute_fmul = AttributeFetchTwoAttributeVertexPcoBinary();
  source1_absolute_fmul[3] = 0x42;
  const auto absolute_fmul =
      Decode(ShaderStage::kVertex, source1_absolute_fmul);
  Check(absolute_fmul.instructions[0].opcode == PcoOpcode::kFloatMultiply &&
            absolute_fmul.instructions[0].source0_absolute == 0 &&
            absolute_fmul.instructions[0].source1_absolute == 1,
        "public FMUL source1-absolute modifier remains orthogonal to FMUL");
  const std::vector<std::uint32_t> absolute_fmul_inputs = {
      FloatBits(2.0F), FloatBits(-0.125F), FloatBits(-0.5F),
      FloatBits(0.375F)};
  Check(ExecuteVertex(absolute_fmul.summary, absolute_fmul.instructions,
                      absolute_fmul_inputs)
                .outputs[0] == FloatBits(1.0F),
        "vertex FMUL clears source1's sign before the binary multiply");

  const auto expect_byte_failure = [](std::size_t offset,
                                      std::uint8_t value,
                                      const std::string &description) {
    auto binary = AttributeFetchTwoAttributeVertexPcoBinary();
    binary[offset] = value;
    ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, binary); },
                  description);
  };
  expect_byte_failure(4, 0x00, "FADD ext0 cleared");
  expect_byte_failure(5, 0x42, "FADD selector cleared");
  expect_byte_failure(5, 0x82, "FADD ext1 cleared");
  expect_byte_failure(6, 0x98, "FADD ext2 enabled");
  expect_byte_failure(6, 0x38, "FADD lower-source mux changed");
  expect_byte_failure(4, 0xbf, "FADD source0 exceeds vi31");
  expect_byte_failure(7, 0x01, "FADD upper-source encoding changed");
  expect_byte_failure(8, 0x01, "FADD ISS selection changed");
  expect_byte_failure(9, 0x44, "FADD destination exceeds temp3");
  auto generic_sc66 = AttributeFetchTwoAttributeVertexPcoBinary();
  generic_sc66[32] = 0x82;
  Check(Decode(ShaderStage::kVertex, generic_sc66).instructions[3].source.index ==
            66,
        "generic ALU accepts the public negative-one special constant");

  auto fragment_sc65 = AttributeFetchGrayFragmentPcoBinary();
  fragment_sc65[4] = 0x81;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, fragment_sc65); },
      "sc65 remains outside the exact fragment-shader gate");

  auto truncated = AttributeFetchTwoAttributeVertexPcoBinary();
  truncated.pop_back();
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, truncated); },
                "truncated two-attribute VS");
  auto trailing = AttributeFetchTwoAttributeVertexPcoBinary();
  trailing.push_back(0xff);
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, trailing); },
                "bytes after two-attribute VS final group");

  const std::vector<std::uint32_t> inputs = {
      FloatBits(0.25f), FloatBits(-0.125f), FloatBits(0.5f),
      FloatBits(0.375f)};
  auto decoded = Decode(ShaderStage::kVertex,
                        AttributeFetchTwoAttributeVertexPcoBinary());
  decoded.instructions[0].source1.bank = PcoRegisterBank::kTemporary;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "tampered FADD source1 register bank");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchTwoAttributeVertexPcoBinary());
  decoded.instructions[0].source_count = 1;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "tampered FADD source count");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchTwoAttributeVertexPcoBinary());
  decoded.instructions[1].output_index = 0;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "two FADD groups write the same temporary");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchTwoAttributeVertexPcoBinary());
  decoded.summary.vertex_input_mask = UINT32_C(0x00000007);
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "tampered two-attribute input mask");
}

void TestFourAttributeFetchFailsClosed() {
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kFragment,
                     AttributeFetchFourAttributeVertexPcoBinary());
      },
      "four-attribute VS decoded as fragment stage");

  const auto expect_byte_failure = [](std::size_t offset,
                                      std::uint8_t value,
                                      const std::string &description) {
    auto binary = AttributeFetchFourAttributeVertexPcoBinary();
    binary[offset] = value;
    ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, binary); },
                  description);
  };
  auto floor_temp_vi_fadd = AttributeFetchFourAttributeVertexPcoBinary();
  floor_temp_vi_fadd[23] = 0x01;
  const auto floor_temp_vi =
      Decode(ShaderStage::kVertex, floor_temp_vi_fadd);
  Check(floor_temp_vi.instructions[2].opcode == PcoOpcode::kFloatAdd &&
            floor_temp_vi.instructions[2].source0_floor == 1,
        "TEMP+VI FADD decodes the public source0-floor modifier");
  expect_byte_failure(24, 0xc2,
                      "TEMP+VI FADD reads temp2 before it is written");

  auto fragment_sc66 = AttributeFetchGrayFragmentPcoBinary();
  fragment_sc66[4] = 0x82;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, fragment_sc66); },
      "sc66 remains outside the exact fragment-shader gate");

  auto truncated = AttributeFetchFourAttributeVertexPcoBinary();
  truncated.pop_back();
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, truncated); },
                "truncated four-attribute VS");
  auto trailing = AttributeFetchFourAttributeVertexPcoBinary();
  trailing.push_back(0xff);
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, trailing); },
                "bytes after four-attribute VS final group");

  const std::vector<std::uint32_t> inputs(8, FloatBits(0.25f));
  auto decoded = Decode(ShaderStage::kVertex,
                        AttributeFetchFourAttributeVertexPcoBinary());
  decoded.instructions[2].source.index = 2;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "serialized TEMP source reads temp2 before write");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchFourAttributeVertexPcoBinary());
  decoded.summary.vertex_input_mask = UINT32_C(0x0000007f);
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "tampered four-attribute input mask");
}

void TestEightAttributeFetchFailsClosed() {
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kFragment,
                     AttributeFetchEightAttributeVertexPcoBinary());
      },
      "eight-attribute VS decoded as fragment stage");

  const auto expect_byte_failure = [](std::size_t offset,
                                      std::uint8_t value,
                                      const std::string &description) {
    auto binary = AttributeFetchEightAttributeVertexPcoBinary();
    binary[offset] = value;
    ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, binary); },
                  description);
  };
  expect_byte_failure(24, 0xc2,
                      "eight-attribute TEMP chain reads temp2 before write");
  auto floor_later_fadd = AttributeFetchEightAttributeVertexPcoBinary();
  floor_later_fadd[103] = 0x01;
  const auto floor_later = Decode(ShaderStage::kVertex, floor_later_fadd);
  bool found_floor_later = false;
  for (const auto &instruction : floor_later.instructions) {
    found_floor_later |= instruction.binary_offset == 103 &&
                         instruction.opcode == PcoOpcode::kFloatAdd &&
                         instruction.source0_floor == 1;
  }
  Check(found_floor_later,
        "eight-attribute later FADD decodes source0-floor");
  expect_byte_failure(152, 0x84,
                      "eight-attribute sc67 changed to unknown sc68");

  auto fragment_sc67 = AttributeFetchGrayFragmentPcoBinary();
  fragment_sc67[4] = 0x83;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, fragment_sc67); },
      "sc67 remains outside the exact fragment-shader gate");

  auto truncated = AttributeFetchEightAttributeVertexPcoBinary();
  truncated.pop_back();
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, truncated); },
                "truncated eight-attribute VS");
  auto trailing = AttributeFetchEightAttributeVertexPcoBinary();
  trailing.push_back(0xff);
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, trailing); },
                "bytes after eight-attribute VS final group");

  const std::vector<std::uint32_t> inputs(16, FloatBits(0.25f));
  auto decoded = Decode(ShaderStage::kVertex,
                        AttributeFetchEightAttributeVertexPcoBinary());
  decoded.instructions[2].source.index = 2;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "serialized eight-attribute TEMP source reads temp2 before write");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchEightAttributeVertexPcoBinary());
  decoded.instructions[6].source1.index = 7;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "serialized eight-attribute vi8 dependency changed");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchEightAttributeVertexPcoBinary());
  decoded.instructions[12].output_index = 0;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "serialized eight-attribute final TEMP destination changed");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchEightAttributeVertexPcoBinary());
  decoded.summary.vertex_input_mask = UINT32_C(0x00007fff);
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "tampered eight-attribute input mask");
}

void TestDecodeAndExecuteVaryingsOne() {
  const auto vertex =
      Decode(ShaderStage::kVertex, VaryingsOneVertexPcoBinary());
  Check(vertex.summary.binary_size == 72,
        "varyings_shader_1 VS binary size");
  Check(vertex.summary.group_count == 7 &&
            vertex.summary.instruction_count == 7,
        "varyings_shader_1 VS has seven exact PCO groups");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x00000003),
        "varyings_shader_1 VS reads vi0 and vi1");
  Check(vertex.summary.vertex_output_mask == UINT64_C(0x00000000000000ff),
        "varyings_shader_1 VS exports position and v1");
  Check(vertex.summary.pixel_output_mask == 0 &&
            vertex.summary.early_hsr_safe == 0 &&
            vertex.summary.ends_task == 1,
        "varyings_shader_1 VS summary flags");

  const std::uint32_t vertex_offsets[] = {3, 13, 23, 31, 41, 51, 67};
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == vertex_offsets[index],
          "varyings_shader_1 VS instruction offset");
  }
  for (std::size_t index = 0; index < 4; ++index) {
    Check(vertex.instructions[index].opcode == PcoOpcode::kMoveBypass &&
              vertex.instructions[index].target ==
                  PcoWriteTarget::kTemporary &&
              vertex.instructions[index].output_index == index,
          "varyings_shader_1 VS uses four exact MBYP setup groups");
  }
  Check(vertex.instructions[0].source.bank ==
                PcoRegisterBank::kVertexInput &&
            vertex.instructions[0].source.index == 0 &&
            vertex.instructions[1].source.bank ==
                PcoRegisterBank::kVertexInput &&
            vertex.instructions[1].source.index == 1 &&
            vertex.instructions[2].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[2].source.index == 0 &&
            vertex.instructions[3].source.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[3].source.index == 64,
        "varyings_shader_1 VS reconstructs x/y/0/1");
  for (std::size_t export_index = 0; export_index < 2; ++export_index) {
    const auto &instruction = vertex.instructions[4 + export_index];
    Check(instruction.opcode == PcoOpcode::kUvsWrite &&
              instruction.source.bank == PcoRegisterBank::kTemporary &&
              instruction.source.index == 0 &&
              instruction.output_index == export_index * 4 &&
              instruction.repeat_count == 4,
          "varyings_shader_1 VS has two repeat-four UVSW exports");
  }
  Check(vertex.instructions[6].opcode == PcoOpcode::kUvsEmitEndTask,
        "varyings_shader_1 VS ends with standalone UVSW emit");

  const auto vertex_static =
      CountPcoInstructions(vertex.instructions, false);
  const auto vertex_dynamic =
      CountPcoInstructions(vertex.instructions, true);
  Check(vertex_static.alu == 4 && vertex_static.texture == 0 &&
            vertex_static.memory == 3,
        "varyings_shader_1 VS static instruction taxonomy");
  Check(vertex_dynamic.alu == 4 && vertex_dynamic.texture == 0 &&
            vertex_dynamic.memory == 9,
        "varyings_shader_1 VS repeat-expanded instruction taxonomy");

  const std::vector<std::uint32_t> inputs = {
      UINT32_C(0xbf400000), // -0.75
      UINT32_C(0x3e800000), // 0.25
  };
  const auto vertex_result =
      ExecuteVertex(vertex.summary, vertex.instructions, inputs);
  Check(vertex_result.written_mask == UINT64_C(0x00000000000000ff),
        "varyings_shader_1 VS writes all eight VTXOUT registers");
  const std::uint32_t expected_vertex_outputs[] = {
      inputs[0], inputs[1], UINT32_C(0x00000000), UINT32_C(0x3f800000),
  };
  for (std::size_t component = 0; component < 4; ++component) {
    Check(vertex_result.outputs[component] ==
                  expected_vertex_outputs[component] &&
              vertex_result.outputs[component + 4] ==
                  expected_vertex_outputs[component],
          "varyings_shader_1 VS duplicates position into smooth v1");
  }
  Check(vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "varyings_shader_1 VS emits and ends the vertex task");

  const auto fragment =
      Decode(ShaderStage::kFragment, VaryingsOneFragmentPcoBinary());
  Check(fragment.summary.binary_size == 48,
        "varyings_shader_1 FS binary size");
  Check(fragment.summary.group_count == 6 &&
            fragment.summary.instruction_count == 6,
        "varyings_shader_1 FS has FITRP, WDF and four MBYP groups");
  Check(fragment.summary.vertex_input_mask == 0 &&
            fragment.summary.vertex_output_mask == 0 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.summary.early_hsr_safe == 1 &&
            fragment.summary.ends_task == 0,
        "varyings_shader_1 FS summary flags");
  const std::uint32_t fragment_offsets[] = {3, 14, 19, 27, 35, 43};
  for (std::size_t index = 0; index < fragment.instructions.size(); ++index) {
    Check(fragment.instructions[index].binary_offset == fragment_offsets[index],
          "varyings_shader_1 FS instruction offset");
  }

  const auto &fitrp = fragment.instructions[0];
  Check(fitrp.opcode == PcoOpcode::kFloatInterpolatePerspective &&
            fitrp.target == PcoWriteTarget::kTemporary &&
            fitrp.source.bank == PcoRegisterBank::kCoefficient &&
            fitrp.source.index == 4 &&
            fitrp.source1.bank == PcoRegisterBank::kCoefficient &&
            fitrp.source1.index == 0 && fitrp.output_index == 0 &&
            fitrp.component_count == 4 && fitrp.data_request == 0 &&
            fitrp.iteration_mode == PcoIterationMode::kPixel &&
            fitrp.perspective == 1 && fitrp.saturate == 0 &&
            fitrp.source_count == 2 && fitrp.repeat_count == 1,
        "varyings_shader_1 FS decodes exact FITRP.PIXEL vec4/drc0");
  const auto &wdf = fragment.instructions[1];
  Check(wdf.opcode == PcoOpcode::kWaitDataFence &&
            wdf.target == PcoWriteTarget::kNone &&
            wdf.data_request == 0 && wdf.source_count == 0,
        "varyings_shader_1 FS decodes exact WDF drc0");
  for (std::size_t component = 0; component < 4; ++component) {
    const auto &move = fragment.instructions[component + 2];
    Check(move.opcode == PcoOpcode::kMoveBypass &&
              move.target == PcoWriteTarget::kPixelOutput &&
              move.source.bank == PcoRegisterBank::kTemporary &&
              move.source.index == component &&
              move.output_index == component,
          "varyings_shader_1 FS exports temp0..3 to PIXOUT0..3");
  }

  const auto fragment_static =
      CountPcoInstructions(fragment.instructions, false);
  const auto fragment_dynamic =
      CountPcoInstructions(fragment.instructions, true);
  Check(fragment_static.alu == 5 && fragment_static.texture == 0 &&
            fragment_static.memory == 0,
        "FITRP plus four MBYP groups are five static ALU instructions");
  Check(fragment_dynamic.alu == 5 && fragment_dynamic.texture == 0 &&
            fragment_dynamic.memory == 0,
        "vec4 FITRP remains one issued dynamic PCO instruction");

  const auto fragment_result = ExecuteFragment(
      fragment.summary, fragment.instructions, MakeVaryingsOneContext());
  Check(fragment_result.written_mask == 0x0f,
        "varyings_shader_1 FS writes all RGBA outputs");
  Check(fragment_result.pixel_outputs[0] == UINT32_C(0x3f020000),
        "perspective interpolation computes R = (x/128)/0.5");
  Check(fragment_result.pixel_outputs[1] == UINT32_C(0x3f3e0000),
        "perspective interpolation computes G = (y/128)/0.5");
  Check(fragment_result.pixel_outputs[2] == UINT32_C(0x3e800000),
        "perspective interpolation computes B = (1/8)/0.5");
  Check(fragment_result.pixel_outputs[3] == UINT32_C(0x3f800000),
        "perspective interpolation computes A = 0.5/0.5");
}

void TestDecodeAndExecuteVaryingsTwo() {
  const auto vertex =
      Decode(ShaderStage::kVertex, VaryingsTwoVertexPcoBinary());
  Check(vertex.summary.binary_size == 120 &&
            vertex.summary.group_count == 12 &&
            vertex.summary.instruction_count == 12,
        "varyings_shader_2 VS binary/group counts");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x00000003) &&
            vertex.summary.vertex_output_mask == UINT64_C(0x0000000000000fff) &&
            vertex.summary.pixel_output_mask == 0 &&
            vertex.summary.early_hsr_safe == 0 &&
            vertex.summary.ends_task == 1,
        "varyings_shader_2 VS exact summary");
  const std::uint32_t vertex_offsets[] = {
      3, 13, 23, 31, 41, 51, 61, 71, 79, 89, 99, 115,
  };
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == vertex_offsets[index],
          "varyings_shader_2 VS instruction offset");
  }
  Check(vertex.instructions[5].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[5].source.bank ==
                PcoRegisterBank::kVertexInput &&
            vertex.instructions[5].source.index == 1 &&
            vertex.instructions[5].source1.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[5].source1.index == 75 &&
            vertex.instructions[5].output_index == 5 &&
            vertex.instructions[6].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[6].source.index == 0 &&
            vertex.instructions[6].source1.index == 75 &&
            vertex.instructions[6].output_index == 4,
        "varyings_shader_2 VS decodes two public vi*sc75 FMUL groups");
  Check(vertex.instructions[9].opcode == PcoOpcode::kUvsWrite &&
            vertex.instructions[9].source.index == 4 &&
            vertex.instructions[9].output_index == 4 &&
            vertex.instructions[9].repeat_count == 4 &&
            vertex.instructions[10].opcode == PcoOpcode::kUvsWrite &&
            vertex.instructions[10].source.index == 4 &&
            vertex.instructions[10].output_index == 8 &&
            vertex.instructions[10].repeat_count == 4,
        "varyings_shader_2 VS exports c/2 to v1 and v2");

  const auto vertex_static = CountPcoInstructions(vertex.instructions, false);
  const auto vertex_dynamic = CountPcoInstructions(vertex.instructions, true);
  Check(vertex_static.alu == 8 && vertex_static.texture == 0 &&
            vertex_static.memory == 4,
        "varyings_shader_2 VS static instruction taxonomy");
  Check(vertex_dynamic.alu == 8 && vertex_dynamic.texture == 0 &&
            vertex_dynamic.memory == 13,
        "varyings_shader_2 VS repeat-expanded instruction taxonomy");

  const std::vector<std::uint32_t> inputs = {
      UINT32_C(0xbf400000), // -0.75
      UINT32_C(0x3e800000), // 0.25
  };
  const auto vertex_result =
      ExecuteVertex(vertex.summary, vertex.instructions, inputs);
  Check(vertex_result.written_mask == UINT64_C(0x0000000000000fff) &&
            vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "varyings_shader_2 VS writes twelve outputs and ends the task");
  const std::uint32_t position[] = {
      UINT32_C(0xbf400000), UINT32_C(0x3e800000), UINT32_C(0x00000000),
      UINT32_C(0x3f800000),
  };
  const std::uint32_t half_position[] = {
      UINT32_C(0xbec00000), UINT32_C(0x3e000000), UINT32_C(0x00000000),
      UINT32_C(0x3f000000),
  };
  for (std::size_t component = 0; component < 4; ++component) {
    Check(vertex_result.outputs[component] == position[component] &&
              vertex_result.outputs[component + 4] == half_position[component] &&
              vertex_result.outputs[component + 8] == half_position[component],
          "varyings_shader_2 VS executes c, c/2, c/2 outputs");
  }

  const auto fragment =
      Decode(ShaderStage::kFragment, VaryingsTwoFragmentPcoBinary());
  Check(fragment.summary.binary_size == 104 &&
            fragment.summary.group_count == 12 &&
            fragment.summary.instruction_count == 12 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.summary.early_hsr_safe == 1,
        "varyings_shader_2 FS exact summary");
  const std::uint32_t fragment_offsets[] = {
      3, 14, 19, 30, 35, 45, 55, 65, 75, 83, 91, 99,
  };
  for (std::size_t index = 0; index < fragment.instructions.size(); ++index) {
    Check(fragment.instructions[index].binary_offset == fragment_offsets[index],
          "varyings_shader_2 FS instruction offset");
  }
  Check(fragment.instructions[0].opcode ==
                PcoOpcode::kFloatInterpolatePerspective &&
            fragment.instructions[0].source.index == 20 &&
            fragment.instructions[0].output_index == 0 &&
            fragment.instructions[1].opcode == PcoOpcode::kWaitDataFence &&
            fragment.instructions[2].opcode ==
                PcoOpcode::kFloatInterpolatePerspective &&
            fragment.instructions[2].source.index == 4 &&
            fragment.instructions[2].output_index == 4 &&
            fragment.instructions[3].opcode == PcoOpcode::kWaitDataFence,
        "varyings_shader_2 FS decodes two ordered FITRP/WDF requests");
  for (std::size_t component = 0; component < 4; ++component) {
    const auto &add = fragment.instructions[component + 4];
    Check(add.opcode == PcoOpcode::kFloatAdd &&
              add.source.bank == PcoRegisterBank::kTemporary &&
              add.source.index == component + 4 &&
              add.source1.bank == PcoRegisterBank::kTemporary &&
              add.source1.index == component,
          "varyings_shader_2 FS decodes public TEMP+TEMP FADD");
  }
  constexpr std::uint16_t kExportSources[] = {8, 9, 10, 0};
  for (std::size_t component = 0; component < 4; ++component) {
    const auto &move = fragment.instructions[component + 8];
    Check(move.opcode == PcoOpcode::kMoveBypass &&
              move.target == PcoWriteTarget::kPixelOutput &&
              move.source.index == kExportSources[component] &&
              move.output_index == component,
          "varyings_shader_2 FS exports FADD results to RGBA");
  }

  const auto fragment_static =
      CountPcoInstructions(fragment.instructions, false);
  const auto fragment_dynamic =
      CountPcoInstructions(fragment.instructions, true);
  Check(fragment_static.alu == 10 && fragment_static.texture == 0 &&
            fragment_static.memory == 0 && fragment_dynamic.alu == 10 &&
            fragment_dynamic.texture == 0 && fragment_dynamic.memory == 0,
        "varyings_shader_2 FS exact instruction taxonomy");

  const auto result = ExecuteFragment(
      fragment.summary, fragment.instructions, MakeVaryingsTwoContext());
  Check(result.written_mask == 0x0f &&
            result.pixel_outputs[0] == UINT32_C(0x3f020000) &&
            result.pixel_outputs[1] == UINT32_C(0x3f3e0000) &&
            result.pixel_outputs[2] == UINT32_C(0x3e800000) &&
            result.pixel_outputs[3] == UINT32_C(0x3f800000),
        "varyings_shader_2 FS executes two perspective varyings plus FADD");
}

void TestVaryingsTwoFailsClosed() {
  ExpectFailure(
      [] { (void)Decode(ShaderStage::kFragment, VaryingsTwoVertexPcoBinary()); },
      "varyings_shader_2 VS decoded as fragment stage");
  ExpectFailure(
      [] { (void)Decode(ShaderStage::kVertex, VaryingsTwoFragmentPcoBinary()); },
      "varyings_shader_2 FS decoded as vertex stage");

  auto floor_fmul_opcode = VaryingsTwoVertexPcoBinary();
  floor_fmul_opcode[51] = 0x41;
  const auto floor_fmul = Decode(ShaderStage::kVertex, floor_fmul_opcode);
  Check(floor_fmul.instructions[5].opcode == PcoOpcode::kFloatMultiply &&
            floor_fmul.instructions[5].source0_floor == 1,
        "public FMUL shares the scalar source0-floor modifier contract");

  auto bad_fmul_constant = VaryingsTwoVertexPcoBinary();
  bad_fmul_constant[53] = 0xca;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_fmul_constant); },
      "varyings_shader_2 FMUL sc75 changed");

  auto bad_uvsw_temp4 = VaryingsTwoVertexPcoBinary();
  bad_uvsw_temp4[91] = 0xc5;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_uvsw_temp4); },
      "varyings_shader_2 non-canonical extended TEMP5 UVSW source");

  auto bad_v2_address = VaryingsTwoVertexPcoBinary();
  bad_v2_address[100] = 0x07;
  const auto shifted_v2 = Decode(ShaderStage::kVertex, bad_v2_address);
  Check(shifted_v2.instructions[10].output_index == 7,
        "generic UVSW accepts an in-range varying output base");

  auto bad_first_coeff = VaryingsTwoFragmentPcoBinary();
  bad_first_coeff[5] = 0xc4;
  Check(Decode(ShaderStage::kFragment, bad_first_coeff)
                .instructions[0]
                .source.index == 4,
        "generic FITRP accepts an aligned in-range coefficient base");

  auto bad_second_destination = VaryingsTwoFragmentPcoBinary();
  bad_second_destination[25] = 0x40;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_second_destination); },
      "varyings_shader_2 second FITRP destination changed");

  auto noncanonical_compact_sources = VaryingsTwoFragmentPcoBinary();
  noncanonical_compact_sources[37] = 0xe0;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment,
                         noncanonical_compact_sources); },
      "varyings_shader_2 TEMP+TEMP ext1 changed from canonical compact form");

  auto bad_fadd_iss = VaryingsTwoFragmentPcoBinary();
  bad_fadd_iss[39] = 0x01;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fadd_iss); },
      "varyings_shader_2 FADD ISS selection changed");

  auto bad_fadd_destination = VaryingsTwoFragmentPcoBinary();
  bad_fadd_destination[40] = 0x49;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fadd_destination); },
      "varyings_shader_2 first FADD destination changed");

  auto bad_export_source = VaryingsTwoFragmentPcoBinary();
  bad_export_source[76] = 0x40;
  Check(Decode(ShaderStage::kFragment, bad_export_source)
                .instructions[8]
                .source.index == 0,
        "generic PIXOUT MBYP accepts an initialized temporary source");

  const auto fragment =
      Decode(ShaderStage::kFragment, VaryingsTwoFragmentPcoBinary());
  auto short_context = MakeVaryingsTwoContext();
  short_context.coefficient_count = 35;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              short_context);
      },
      "varyings_shader_2 truncated coefficient span");

  auto wrong_profile_context = MakeVaryingsTwoContext();
  wrong_profile_context.coefficient_count = 20;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              wrong_profile_context);
      },
      "varyings_shader_2 cannot execute with case-1 coefficient count");

  const auto varying_one =
      Decode(ShaderStage::kFragment, VaryingsOneFragmentPcoBinary());
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(varying_one.summary, varying_one.instructions,
                              MakeVaryingsTwoContext());
      },
      "varyings_shader_1 exact profile rejects case-2 coefficient count");
}

void TestDecodeAndExecuteVaryingsFour() {
  const auto vertex =
      Decode(ShaderStage::kVertex, VaryingsFourVertexPcoBinary());
  Check(vertex.summary.binary_size == 136 &&
            vertex.summary.group_count == 14 &&
            vertex.summary.instruction_count == 14,
        "varyings_shader_4 VS binary/group counts");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x00000003) &&
            vertex.summary.vertex_output_mask == UINT64_C(0x00000000000fffff) &&
            vertex.summary.pixel_output_mask == 0 &&
            vertex.summary.early_hsr_safe == 0 &&
            vertex.summary.ends_task == 1,
        "varyings_shader_4 VS exact summary");
  const std::uint32_t vertex_offsets[] = {
      3, 13, 23, 31, 41, 51, 61, 71, 79, 89, 99, 109, 119, 131,
  };
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == vertex_offsets[index],
          "varyings_shader_4 VS instruction offset");
  }
  Check(vertex.instructions[5].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[5].source.index == 1 &&
            vertex.instructions[5].source1.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[5].source1.index == 76 &&
            vertex.instructions[5].output_index == 5 &&
            vertex.instructions[6].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[6].source.index == 0 &&
            vertex.instructions[6].source1.index == 76 &&
            vertex.instructions[6].output_index == 4,
        "varyings_shader_4 VS decodes two public vi*sc76 FMUL groups");
  for (std::size_t varying = 0; varying < 4; ++varying) {
    const auto &write = vertex.instructions[varying + 9];
    Check(write.opcode == PcoOpcode::kUvsWrite &&
              write.source.bank == PcoRegisterBank::kTemporary &&
              write.source.index == 4 && write.output_index == 4 + varying * 4 &&
              write.repeat_count == 4,
          "varyings_shader_4 VS exports c/4 to each varying");
  }

  const auto vertex_static = CountPcoInstructions(vertex.instructions, false);
  const auto vertex_dynamic = CountPcoInstructions(vertex.instructions, true);
  Check(vertex_static.alu == 8 && vertex_static.texture == 0 &&
            vertex_static.memory == 6,
        "varyings_shader_4 VS static instruction taxonomy");
  Check(vertex_dynamic.alu == 8 && vertex_dynamic.texture == 0 &&
            vertex_dynamic.memory == 21,
        "varyings_shader_4 VS repeat-expanded instruction taxonomy");

  const std::vector<std::uint32_t> inputs = {
      UINT32_C(0xbf400000), // -0.75
      UINT32_C(0x3e800000), // 0.25
  };
  const auto vertex_result =
      ExecuteVertex(vertex.summary, vertex.instructions, inputs);
  Check(vertex_result.written_mask == UINT64_C(0x00000000000fffff) &&
            vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "varyings_shader_4 VS writes twenty outputs and ends the task");
  const std::uint32_t position[] = {
      UINT32_C(0xbf400000), UINT32_C(0x3e800000), UINT32_C(0x00000000),
      UINT32_C(0x3f800000),
  };
  const std::uint32_t quarter_position[] = {
      UINT32_C(0xbe400000), UINT32_C(0x3d800000), UINT32_C(0x00000000),
      UINT32_C(0x3e800000),
  };
  for (std::size_t component = 0; component < 4; ++component) {
    Check(vertex_result.outputs[component] == position[component],
          "varyings_shader_4 VS preserves clip position");
    for (std::size_t varying = 0; varying < 4; ++varying) {
      Check(vertex_result.outputs[4 + varying * 4 + component] ==
                quarter_position[component],
            "varyings_shader_4 VS executes four c/4 exports");
    }
  }

  const auto fragment =
      Decode(ShaderStage::kFragment, VaryingsFourFragmentPcoBinary());
  Check(fragment.summary.binary_size == 216 &&
            fragment.summary.group_count == 24 &&
            fragment.summary.instruction_count == 24 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.summary.early_hsr_safe == 1,
        "varyings_shader_4 FS exact summary");
  const std::uint32_t fragment_offsets[] = {
      3,   14,  19,  30,  35,  45,  55,  65,  75,  86,  91,  101,
      111, 121, 131, 142, 147, 157, 167, 177, 187, 195, 203, 211,
  };
  for (std::size_t index = 0; index < fragment.instructions.size(); ++index) {
    Check(fragment.instructions[index].binary_offset == fragment_offsets[index],
          "varyings_shader_4 FS instruction offset");
  }
  constexpr std::uint16_t kFitrpCoefficients[] = {20, 4, 36, 52};
  constexpr std::uint16_t kFitrpDestinations[] = {0, 4, 1, 1};
  constexpr std::size_t kFitrpGroups[] = {0, 2, 8, 14};
  for (std::size_t varying = 0; varying < 4; ++varying) {
    const std::size_t group = kFitrpGroups[varying];
    const auto &fitrp = fragment.instructions[group];
    const auto &wdf = fragment.instructions[group + 1];
    Check(fitrp.opcode == PcoOpcode::kFloatInterpolatePerspective &&
              fitrp.source.index == kFitrpCoefficients[varying] &&
              fitrp.output_index == kFitrpDestinations[varying] &&
              wdf.opcode == PcoOpcode::kWaitDataFence,
          "varyings_shader_4 FS decodes ordered FITRP/WDF pairs");
  }
  for (std::size_t group : {std::size_t{4}, std::size_t{5}, std::size_t{6},
                            std::size_t{7}, std::size_t{10}, std::size_t{11},
                            std::size_t{12}, std::size_t{13},
                            std::size_t{16}, std::size_t{17},
                            std::size_t{18}, std::size_t{19}}) {
    Check(fragment.instructions[group].opcode == PcoOpcode::kFloatAdd,
          "varyings_shader_4 FS keeps twelve public FADD groups");
  }
  constexpr std::uint16_t kExportSources[] = {5, 6, 7, 0};
  for (std::size_t component = 0; component < 4; ++component) {
    const auto &move = fragment.instructions[component + 20];
    Check(move.opcode == PcoOpcode::kMoveBypass &&
              move.target == PcoWriteTarget::kPixelOutput &&
              move.source.index == kExportSources[component] &&
              move.output_index == component,
          "varyings_shader_4 FS exports left-associated FADD results");
  }

  const auto fragment_static =
      CountPcoInstructions(fragment.instructions, false);
  const auto fragment_dynamic =
      CountPcoInstructions(fragment.instructions, true);
  Check(fragment_static.alu == 20 && fragment_static.texture == 0 &&
            fragment_static.memory == 0 && fragment_dynamic.alu == 20 &&
            fragment_dynamic.texture == 0 && fragment_dynamic.memory == 0,
        "varyings_shader_4 FS exact instruction taxonomy");

  const auto result = ExecuteFragment(
      fragment.summary, fragment.instructions, MakeVaryingsFourContext());
  Check(result.written_mask == 0x0f &&
            result.pixel_outputs[0] == UINT32_C(0x3f020000) &&
            result.pixel_outputs[1] == UINT32_C(0x3f3e0000) &&
            result.pixel_outputs[2] == UINT32_C(0x3e800000) &&
            result.pixel_outputs[3] == UINT32_C(0x3f800000),
        "varyings_shader_4 FS executes four perspective varyings and FADDs");
}

void TestVaryingsFourFailsClosed() {
  ExpectFailure(
      [] { (void)Decode(ShaderStage::kFragment, VaryingsFourVertexPcoBinary()); },
      "varyings_shader_4 VS decoded as fragment stage");
  ExpectFailure(
      [] { (void)Decode(ShaderStage::kVertex, VaryingsFourFragmentPcoBinary()); },
      "varyings_shader_4 FS decoded as vertex stage");

  auto bad_quarter_constant = VaryingsFourVertexPcoBinary();
  bad_quarter_constant[53] = 0xcb;
  Check(Decode(ShaderStage::kVertex, bad_quarter_constant)
                .instructions[5]
                .source1.index == 75,
        "generic FMUL accepts the public one-half special constant");

  auto bad_v4_address = VaryingsFourVertexPcoBinary();
  bad_v4_address[120] = 0x0c;
  Check(Decode(ShaderStage::kVertex, bad_v4_address)
                .instructions[12]
                .output_index == 12,
        "generic UVSW accepts an in-range fourth varying output base");

  auto bad_uvsw_padding = VaryingsFourVertexPcoBinary();
  bad_uvsw_padding[126] = 0xf2;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_uvsw_padding); },
      "varyings_shader_4 final UVSW alignment padding changed");

  auto bad_third_coefficient = VaryingsFourFragmentPcoBinary();
  bad_third_coefficient[77] = 0xd4;
  Check(Decode(ShaderStage::kFragment, bad_third_coefficient)
                .instructions[8]
                .source.index == 20,
        "generic FITRP accepts a reused aligned coefficient base");

  auto bad_third_destination = VaryingsFourFragmentPcoBinary();
  bad_third_destination[81] = 0x40;
  Check(Decode(ShaderStage::kFragment, bad_third_destination)
                .instructions[8]
                .output_index == 0,
        "generic FITRP accepts an in-range temporary destination");

  auto bad_fourth_coefficient = VaryingsFourFragmentPcoBinary();
  bad_fourth_coefficient[133] = 0xe4;
  Check(Decode(ShaderStage::kFragment, bad_fourth_coefficient)
                .instructions[14]
                .source.index == 36,
        "generic FITRP accepts another aligned coefficient base");

  auto bad_second_layer_fadd = VaryingsFourFragmentPcoBinary();
  bad_second_layer_fadd[92] = 0xc9;
  Check(Decode(ShaderStage::kFragment, bad_second_layer_fadd)
                .instructions[10]
                .source.index == 9,
        "generic FADD accepts another initialized temporary source");

  auto bad_export_source = VaryingsFourFragmentPcoBinary();
  bad_export_source[188] = 0x44;
  Check(Decode(ShaderStage::kFragment, bad_export_source)
                .instructions[20]
                .source.index == 4,
        "generic PIXOUT MBYP accepts another initialized temporary source");

  const auto fragment =
      Decode(ShaderStage::kFragment, VaryingsFourFragmentPcoBinary());
  auto short_context = MakeVaryingsFourContext();
  short_context.coefficient_count = 67;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              short_context);
      },
      "varyings_shader_4 truncated coefficient span");

  auto wrong_profile_context = MakeVaryingsFourContext();
  wrong_profile_context.coefficient_count = 36;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              wrong_profile_context);
      },
      "varyings_shader_4 cannot execute with case-2 coefficient count");

  const auto varying_two =
      Decode(ShaderStage::kFragment, VaryingsTwoFragmentPcoBinary());
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(varying_two.summary, varying_two.instructions,
                              MakeVaryingsFourContext());
      },
      "varyings_shader_2 cannot execute with case-4 coefficient count");
}

void TestDecodeAndExecuteVaryingsEight() {
  const auto vertex =
      Decode(ShaderStage::kVertex, VaryingsEightVertexPcoBinary());
  Check(vertex.summary.binary_size == 176 &&
            vertex.summary.group_count == 18 &&
            vertex.summary.instruction_count == 18,
        "varyings_shader_8 VS binary/group counts");
  Check(vertex.summary.vertex_input_mask == UINT32_C(0x00000003) &&
            vertex.summary.vertex_output_mask ==
                UINT64_C(0x0000000fffffffff) &&
            vertex.summary.pixel_output_mask == 0 &&
            vertex.summary.early_hsr_safe == 0 &&
            vertex.summary.ends_task == 1,
        "varyings_shader_8 VS exact summary");
  const std::uint32_t vertex_offsets[] = {
      3,  13, 23, 31, 41,  51,  61,  71,  79,
      89, 99, 109, 119, 129, 139, 149, 159, 171,
  };
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == vertex_offsets[index],
          "varyings_shader_8 VS instruction offset");
  }
  Check(vertex.instructions[5].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[5].source.index == 1 &&
            vertex.instructions[5].source1.bank == PcoRegisterBank::kSpecial &&
            vertex.instructions[5].source1.index == 77 &&
            vertex.instructions[5].output_index == 5 &&
            vertex.instructions[6].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[6].source.index == 0 &&
            vertex.instructions[6].source1.index == 77 &&
            vertex.instructions[6].output_index == 4,
        "varyings_shader_8 VS decodes two public vi*sc77 FMUL groups");
  for (std::size_t varying = 0; varying < 8; ++varying) {
    const auto &write = vertex.instructions[varying + 9];
    Check(write.opcode == PcoOpcode::kUvsWrite &&
              write.source.bank == PcoRegisterBank::kTemporary &&
              write.source.index == 4 && write.output_index == 4 + varying * 4 &&
              write.repeat_count == 4,
          "varyings_shader_8 VS exports c/8 to each varying");
  }
  std::size_t vertex_mbyp = 0;
  std::size_t vertex_fmul = 0;
  std::size_t vertex_uvsw = 0;
  std::size_t vertex_emit = 0;
  for (const auto &instruction : vertex.instructions) {
    vertex_mbyp += instruction.opcode == PcoOpcode::kMoveBypass;
    vertex_fmul += instruction.opcode == PcoOpcode::kFloatMultiply;
    vertex_uvsw += instruction.opcode == PcoOpcode::kUvsWrite;
    vertex_emit += instruction.opcode == PcoOpcode::kUvsEmitEndTask;
  }
  Check(vertex_mbyp == 6 && vertex_fmul == 2 && vertex_uvsw == 9 &&
            vertex_emit == 1,
        "varyings_shader_8 VS exact opcode histogram");

  const auto vertex_static = CountPcoInstructions(vertex.instructions, false);
  const auto vertex_dynamic = CountPcoInstructions(vertex.instructions, true);
  Check(vertex_static.alu == 8 && vertex_static.texture == 0 &&
            vertex_static.memory == 10,
        "varyings_shader_8 VS static instruction taxonomy");
  Check(vertex_dynamic.alu == 8 && vertex_dynamic.texture == 0 &&
            vertex_dynamic.memory == 37,
        "varyings_shader_8 VS repeat-expanded instruction taxonomy");

  const std::vector<std::uint32_t> inputs = {
      UINT32_C(0xbf400000), // -0.75
      UINT32_C(0x3e800000), // 0.25
  };
  const auto vertex_result =
      ExecuteVertex(vertex.summary, vertex.instructions, inputs);
  Check(vertex_result.written_mask == UINT64_C(0x0000000fffffffff) &&
            vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "varyings_shader_8 VS writes thirty-six outputs and ends the task");
  const std::uint32_t position[] = {
      UINT32_C(0xbf400000), UINT32_C(0x3e800000), UINT32_C(0x00000000),
      UINT32_C(0x3f800000),
  };
  const std::uint32_t eighth_position[] = {
      UINT32_C(0xbdc00000), UINT32_C(0x3d000000), UINT32_C(0x00000000),
      UINT32_C(0x3e000000),
  };
  for (std::size_t component = 0; component < 4; ++component) {
    Check(vertex_result.outputs[component] == position[component],
          "varyings_shader_8 VS preserves clip position");
    for (std::size_t varying = 0; varying < 8; ++varying) {
      Check(vertex_result.outputs[4 + varying * 4 + component] ==
                eighth_position[component],
            "varyings_shader_8 VS executes eight c/8 exports");
    }
  }

  const auto fragment =
      Decode(ShaderStage::kFragment, VaryingsEightFragmentPcoBinary());
  Check(fragment.summary.binary_size == 440 &&
            fragment.summary.group_count == 48 &&
            fragment.summary.instruction_count == 48 &&
            fragment.summary.vertex_input_mask == 0 &&
            fragment.summary.vertex_output_mask == 0 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.summary.early_hsr_safe == 1 &&
            fragment.summary.ends_task == 0,
        "varyings_shader_8 FS exact summary");
  const std::uint32_t fragment_offsets[] = {
      3,   14,  19,  30,  35,  45,  55,  65,  75,  86,  91,  101,
      111, 121, 131, 142, 147, 157, 167, 177, 187, 198, 203, 213,
      223, 233, 243, 254, 259, 269, 279, 289, 299, 310, 315, 325,
      335, 345, 355, 366, 371, 381, 391, 401, 411, 419, 427, 435,
  };
  for (std::size_t index = 0; index < fragment.instructions.size(); ++index) {
    Check(fragment.instructions[index].binary_offset == fragment_offsets[index],
          "varyings_shader_8 FS instruction offset");
  }
  constexpr std::uint16_t kFitrpCoefficients[] = {
      20, 4, 36, 52, 68, 84, 100, 116,
  };
  constexpr std::uint16_t kFitrpDestinations[] = {0, 4, 1, 1, 1, 1, 1, 1};
  constexpr std::size_t kFitrpGroups[] = {0, 2, 8, 14, 20, 26, 32, 38};
  for (std::size_t varying = 0; varying < 8; ++varying) {
    const std::size_t group = kFitrpGroups[varying];
    const auto &fitrp = fragment.instructions[group];
    const auto &wdf = fragment.instructions[group + 1];
    Check(fitrp.opcode == PcoOpcode::kFloatInterpolatePerspective &&
              fitrp.source.index == kFitrpCoefficients[varying] &&
              fitrp.output_index == kFitrpDestinations[varying] &&
              wdf.opcode == PcoOpcode::kWaitDataFence,
          "varyings_shader_8 FS decodes ordered FITRP/WDF pairs");
  }
  std::size_t fitrp_count = 0;
  std::size_t wdf_count = 0;
  std::size_t fadd_count = 0;
  std::size_t mbyp_count = 0;
  for (const auto &instruction : fragment.instructions) {
    fitrp_count +=
        instruction.opcode == PcoOpcode::kFloatInterpolatePerspective;
    wdf_count += instruction.opcode == PcoOpcode::kWaitDataFence;
    fadd_count += instruction.opcode == PcoOpcode::kFloatAdd;
    mbyp_count += instruction.opcode == PcoOpcode::kMoveBypass;
  }
  Check(fitrp_count == 8 && wdf_count == 8 && fadd_count == 28 &&
            mbyp_count == 4,
        "varyings_shader_8 FS exact opcode histogram");
  constexpr std::uint16_t kExportSources[] = {5, 6, 7, 0};
  for (std::size_t component = 0; component < 4; ++component) {
    const auto &move = fragment.instructions[component + 44];
    Check(move.opcode == PcoOpcode::kMoveBypass &&
              move.target == PcoWriteTarget::kPixelOutput &&
              move.source.index == kExportSources[component] &&
              move.output_index == component,
          "varyings_shader_8 FS exports left-associated FADD results");
  }

  const auto fragment_static =
      CountPcoInstructions(fragment.instructions, false);
  const auto fragment_dynamic =
      CountPcoInstructions(fragment.instructions, true);
  Check(fragment_static.alu == 40 && fragment_static.texture == 0 &&
            fragment_static.memory == 0 && fragment_dynamic.alu == 40 &&
            fragment_dynamic.texture == 0 && fragment_dynamic.memory == 0,
        "varyings_shader_8 FS exact instruction taxonomy");

  const auto result = ExecuteFragment(
      fragment.summary, fragment.instructions, MakeVaryingsEightContext());
  Check(result.written_mask == 0x0f &&
            result.pixel_outputs[0] == UINT32_C(0x3f020000) &&
            result.pixel_outputs[1] == UINT32_C(0x3f3e0000) &&
            result.pixel_outputs[2] == UINT32_C(0x3e800000) &&
            result.pixel_outputs[3] == UINT32_C(0x3f800000),
        "varyings_shader_8 FS executes eight perspective varyings and FADDs");
}

void TestVaryingsEightFailsClosed() {
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kFragment, VaryingsEightVertexPcoBinary());
      },
      "varyings_shader_8 VS decoded as fragment stage");
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kVertex, VaryingsEightFragmentPcoBinary());
      },
      "varyings_shader_8 FS decoded as vertex stage");

  auto bad_eighth_constant = VaryingsEightVertexPcoBinary();
  bad_eighth_constant[53] = 0xcc;
  Check(Decode(ShaderStage::kVertex, bad_eighth_constant)
                .instructions[5]
                .source1.index == 76,
        "generic FMUL accepts the public one-quarter special constant");

  auto bad_eighth_address = VaryingsEightVertexPcoBinary();
  bad_eighth_address[160] = 0x1c;
  Check(Decode(ShaderStage::kVertex, bad_eighth_address)
                .instructions[16]
                .output_index == 28,
        "generic UVSW accepts an in-range eighth varying output base");

  auto bad_uvsw_padding = VaryingsEightVertexPcoBinary();
  bad_uvsw_padding[166] = 0xf2;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_uvsw_padding); },
      "varyings_shader_8 final UVSW alignment padding changed");

  auto bad_fifth_coefficient_high = VaryingsEightFragmentPcoBinary();
  bad_fifth_coefficient_high[191] = 0x10;
  Check(Decode(ShaderStage::kFragment, bad_fifth_coefficient_high)
                .instructions[20]
                .source.index == 4,
        "generic FITRP accepts a low-bank aligned coefficient base");

  auto bad_fifth_coefficient_low = VaryingsEightFragmentPcoBinary();
  bad_fifth_coefficient_low[189] = 0xd4;
  Check(Decode(ShaderStage::kFragment, bad_fifth_coefficient_low)
                .instructions[20]
                .source.index == 84,
        "generic FITRP accepts a high-bank aligned coefficient base");

  auto bad_eighth_coefficient = VaryingsEightFragmentPcoBinary();
  bad_eighth_coefficient[359] = 0x10;
  Check(Decode(ShaderStage::kFragment, bad_eighth_coefficient)
                .instructions[38]
                .source.index == 52,
        "generic FITRP accepts a reused final coefficient base");

  auto bad_final_layer_fadd = VaryingsEightFragmentPcoBinary();
  bad_final_layer_fadd[372] = 0xc6;
  Check(Decode(ShaderStage::kFragment, bad_final_layer_fadd)
                .instructions[40]
                .source.index == 6,
        "generic FADD accepts an initialized final-layer source");

  auto bad_export_source = VaryingsEightFragmentPcoBinary();
  bad_export_source[412] = 0x44;
  Check(Decode(ShaderStage::kFragment, bad_export_source)
                .instructions[44]
                .source.index == 4,
        "generic PIXOUT MBYP accepts a final initialized temporary");

  auto truncated_fragment = VaryingsEightFragmentPcoBinary();
  truncated_fragment.pop_back();
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, truncated_fragment); },
      "varyings_shader_8 truncated final export");

  const auto fragment =
      Decode(ShaderStage::kFragment, VaryingsEightFragmentPcoBinary());
  auto short_context = MakeVaryingsEightContext();
  short_context.coefficient_count = 131;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              short_context);
      },
      "varyings_shader_8 truncated coefficient span");

  auto wrong_profile_context = MakeVaryingsEightContext();
  wrong_profile_context.coefficient_count = 68;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              wrong_profile_context);
      },
      "varyings_shader_8 cannot execute with case-4 coefficient count");

  const auto varying_four =
      Decode(ShaderStage::kFragment, VaryingsFourFragmentPcoBinary());
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(varying_four.summary, varying_four.instructions,
                              MakeVaryingsEightContext());
      },
      "varyings_shader_4 cannot execute with case-8 coefficient count");
}

void TestVaryingsOneFailsClosed() {
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kFragment, VaryingsOneVertexPcoBinary());
      },
      "varyings_shader_1 VS decoded as fragment stage");
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kVertex, VaryingsOneFragmentPcoBinary());
      },
      "varyings_shader_1 FS decoded as vertex stage");

  auto bad_vertex_repeat = VaryingsOneVertexPcoBinary();
  bad_vertex_repeat[50] = 0x04;
  const auto short_varying = Decode(ShaderStage::kVertex, bad_vertex_repeat);
  Check(short_varying.instructions[5].repeat_count == 3,
        "generic UVSW accepts a bounded vec3 varying export");

  auto bad_vertex_address = VaryingsOneVertexPcoBinary();
  bad_vertex_address[52] = 0x03;
  const auto overlapping_varying =
      Decode(ShaderStage::kVertex, bad_vertex_address);
  Check(overlapping_varying.instructions[5].output_index == 3,
        "generic UVSW preserves an in-range encoded output base");

  auto bad_fitrp_perspective = VaryingsOneFragmentPcoBinary();
  bad_fitrp_perspective[3] = 0xa0;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fitrp_perspective); },
      "FITRP perspective bit cleared");

  auto bad_fitrp_drc = VaryingsOneFragmentPcoBinary();
  bad_fitrp_drc[3] = 0xb8;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_fitrp_drc); },
                "FITRP changed from drc0 to drc1");

  auto bad_fitrp_iteration = VaryingsOneFragmentPcoBinary();
  bad_fitrp_iteration[3] = 0xb1;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fitrp_iteration); },
      "FITRP PIXEL iteration mode changed");

  auto bad_fitrp_count = VaryingsOneFragmentPcoBinary();
  bad_fitrp_count[4] = 0x03;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_fitrp_count); },
                "vec3 FITRP leaves the later TEMP3 read uninitialized");

  auto bad_fitrp_saturate = VaryingsOneFragmentPcoBinary();
  bad_fitrp_saturate[4] = 0x14;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fitrp_saturate); },
      "FITRP saturate bit enabled");

  auto bad_fitrp_repeat = VaryingsOneFragmentPcoBinary();
  bad_fitrp_repeat[2] = 0x02;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_fitrp_repeat); },
                "FITRP group repeat changed");

  for (std::size_t offset = 5; offset <= 10; ++offset) {
    auto bad_source = VaryingsOneFragmentPcoBinary();
    bad_source[offset] ^= 0x01;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_source); },
                  "FITRP coefficient/temp source encoding changed");
  }

  auto bad_wdf_drc = VaryingsOneFragmentPcoBinary();
  bad_wdf_drc[14] = 0xea;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, bad_wdf_drc); },
                "WDF changed from drc0 to drc1");

  const auto decoded =
      Decode(ShaderStage::kFragment, VaryingsOneFragmentPcoBinary());
  ExpectFailure(
      [&] { (void)ExecuteFragment(decoded.summary, decoded.instructions); },
      "FITRP execution without a coefficient context");

  auto short_context = MakeVaryingsOneContext();
  short_context.coefficient_count = 19;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, decoded.instructions,
                              short_context);
      },
      "FITRP truncated coefficient span");

  auto zero_w_context = MakeVaryingsOneContext();
  zero_w_context.coefficients[0] = UINT32_C(0x00000000);
  zero_w_context.coefficients[2] = UINT32_C(0x00000000);
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, decoded.instructions,
                              zero_w_context);
      },
      "FITRP perspective divide by zero");

  auto semantic = decoded;
  semantic.instructions[0].source.index = 5;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsOneContext());
      },
      "FITRP coefficient base mutation");

  semantic = decoded;
  semantic.instructions[0].repeat_count = 2;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsOneContext());
      },
      "FITRP semantic repeat mutation");

  semantic = decoded;
  semantic.instructions[0].iteration_mode = PcoIterationMode::kSample;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsOneContext());
      },
      "FITRP PIXEL mode mutation");

  semantic = decoded;
  const auto wdf = semantic.instructions[1];
  semantic.instructions[1] = semantic.instructions[2];
  semantic.instructions[1].group_index = 1;
  semantic.instructions[1].binary_offset = 14;
  semantic.instructions[2] = wdf;
  semantic.instructions[2].group_index = 2;
  semantic.instructions[2].binary_offset = 19;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsOneContext());
      },
      "TEMP result read before WDF completion");

  semantic = decoded;
  semantic.instructions[1].data_request = 1;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsOneContext());
      },
      "WDF DRC mutation in serialized semantic instruction");
}

void TestDecodeAndExecuteFillTexNearest() {
  const auto vertex =
      Decode(ShaderStage::kVertex, FillTexNearestVertexPcoBinary());
  Check(vertex.summary.binary_size == 80 &&
            vertex.summary.group_count == 8 &&
            vertex.summary.instruction_count == 8 &&
            vertex.summary.vertex_input_mask == UINT32_C(0x0000000f) &&
            vertex.summary.vertex_output_mask == UINT64_C(0x0000003f) &&
            vertex.summary.ends_task == 1,
        "fill_tex_nearest VS exact summary");
  const std::uint32_t vertex_offsets[] = {3, 13, 23, 33, 41, 51, 61, 75};
  for (std::size_t index = 0; index < vertex.instructions.size(); ++index) {
    Check(vertex.instructions[index].binary_offset == vertex_offsets[index],
          "fill_tex_nearest VS instruction offset");
  }
  Check(vertex.instructions[0].opcode == PcoOpcode::kMoveBypass &&
            vertex.instructions[0].source.bank == PcoRegisterBank::kShared &&
            vertex.instructions[0].source.index == 0 &&
            vertex.instructions[0].output_index == 4,
        "fill_tex_nearest VS loads scale from SH0");
  Check(vertex.instructions[1].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[1].source.index == 0 &&
            vertex.instructions[1].source1.bank ==
                PcoRegisterBank::kTemporary &&
            vertex.instructions[1].source1.index == 4 &&
            vertex.instructions[1].output_index == 0 &&
            vertex.instructions[2].opcode == PcoOpcode::kFloatMultiply &&
            vertex.instructions[2].source.index == 1 &&
            vertex.instructions[2].source1.index == 4 &&
            vertex.instructions[2].output_index == 1,
        "fill_tex_nearest VS performs position.xy times scale");
  Check(vertex.instructions[5].opcode == PcoOpcode::kUvsWrite &&
            vertex.instructions[5].source.index == 0 &&
            vertex.instructions[5].output_index == 0 &&
            vertex.instructions[5].repeat_count == 4 &&
            vertex.instructions[6].opcode == PcoOpcode::kUvsWrite &&
            vertex.instructions[6].source.bank == PcoRegisterBank::kVertexInput &&
            vertex.instructions[6].source.index == 2 &&
            vertex.instructions[6].output_index == 4 &&
            vertex.instructions[6].repeat_count == 2 &&
            vertex.instructions[7].opcode == PcoOpcode::kUvsEmitEndTask,
        "fill_tex_nearest VS exports clip position and linked texcoord");

  const auto vertex_static = CountPcoInstructions(vertex.instructions, false);
  const auto vertex_dynamic = CountPcoInstructions(vertex.instructions, true);
  Check(vertex_static.alu == 5 && vertex_static.texture == 0 &&
            vertex_static.memory == 3 && vertex_dynamic.alu == 5 &&
            vertex_dynamic.texture == 0 && vertex_dynamic.memory == 7,
        "fill_tex_nearest VS exact instruction taxonomy");

  PcoVertexExecutionContext vertex_context;
  vertex_context.shared_count = 1;
  vertex_context.shared_registers[0] = FloatBits(0.5F);
  const std::vector<std::uint32_t> vertex_inputs = {
      FloatBits(-0.75F), FloatBits(0.25F), FloatBits(0.125F),
      FloatBits(0.875F),
  };
  const auto vertex_result = ExecuteVertex(
      vertex.summary, vertex.instructions, vertex_inputs, vertex_context);
  Check(vertex_result.written_mask == UINT64_C(0x3f) &&
            vertex_result.outputs[0] == FloatBits(-0.375F) &&
            vertex_result.outputs[1] == FloatBits(0.125F) &&
            vertex_result.outputs[2] == FloatBits(0.0F) &&
            vertex_result.outputs[3] == FloatBits(1.0F) &&
            vertex_result.outputs[4] == vertex_inputs[2] &&
            vertex_result.outputs[5] == vertex_inputs[3] &&
            vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "fill_tex_nearest VS executes live scale and texcoord values");

  const auto fragment =
      Decode(ShaderStage::kFragment, FillTexNearestFragmentPcoBinary());
  Check(fragment.summary.binary_size == 184 &&
            fragment.summary.group_count == 22 &&
            fragment.summary.instruction_count == 22 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.summary.early_hsr_safe == 1,
        "fill_tex_nearest FS exact summary");
  const std::uint32_t fragment_offsets[] = {
      3,  14, 19, 27, 35, 43, 51, 59, 67, 75, 83,
      91, 99, 107, 115, 123, 131, 144, 149, 157, 165, 179,
  };
  for (std::size_t index = 0; index < fragment.instructions.size(); ++index) {
    Check(fragment.instructions[index].binary_offset ==
              fragment_offsets[index],
          "fill_tex_nearest FS instruction offset");
  }
  const auto &fitrp = fragment.instructions[0];
  Check(fitrp.opcode == PcoOpcode::kFloatInterpolatePerspective &&
            fitrp.source.index == 4 && fitrp.source1.index == 0 &&
            fitrp.output_index == 16 && fitrp.component_count == 2 &&
            fitrp.iteration_mode == PcoIterationMode::kPixel &&
            fitrp.perspective == 1 &&
            fragment.instructions[1].opcode == PcoOpcode::kWaitDataFence,
        "fill_tex_nearest FS decodes public two-component FITRP/WDF");
  for (std::size_t index = 0; index < 14; ++index) {
    const auto &move = fragment.instructions[index + 2];
    Check(move.opcode == PcoOpcode::kMoveBypass &&
              move.target == PcoWriteTarget::kTemporary &&
              move.source.bank == PcoRegisterBank::kSpecial &&
              move.source.index == 0 && move.output_index == 18 + index,
          "fill_tex_nearest FS initializes TEMP18..31 from special zero");
  }
  const auto &sample = fragment.instructions[16];
  Check(sample.opcode == PcoOpcode::kTextureSample &&
            sample.target == PcoWriteTarget::kTemporary &&
            sample.source.bank == PcoRegisterBank::kTemporary &&
            sample.source.index == 16 &&
            sample.source1.bank == PcoRegisterBank::kShared &&
            sample.source1.index == 0 &&
            sample.source2.bank == PcoRegisterBank::kShared &&
            sample.source2.index == 8 && sample.output_index == 0 &&
            sample.component_count == 4 && sample.data_request == 0 &&
            fragment.instructions[17].opcode == PcoOpcode::kWaitDataFence,
        "fill_tex_nearest FS decodes public SMP.2D.FCNORM/WDF");
  for (std::size_t component = 0; component < 4; ++component) {
    const auto &move = fragment.instructions[component + 18];
    Check(move.opcode == PcoOpcode::kMoveBypass &&
              move.target == PcoWriteTarget::kPixelOutput &&
              move.source.bank == PcoRegisterBank::kTemporary &&
              move.source.index == component &&
              move.output_index == component,
          "fill_tex_nearest FS exports sampled TEMP0..3");
  }

  const auto fragment_static =
      CountPcoInstructions(fragment.instructions, false);
  const auto fragment_dynamic =
      CountPcoInstructions(fragment.instructions, true);
  Check(fragment_static.alu == 19 && fragment_static.texture == 1 &&
            fragment_static.memory == 0 && fragment_dynamic.alu == 19 &&
            fragment_dynamic.texture == 1 && fragment_dynamic.memory == 0,
        "fill_tex_nearest FS exact instruction taxonomy");

  const auto suspended = ExecuteFragment(
      fragment.summary, fragment.instructions, MakeFillTexNearestContext());
  Check(suspended.suspended == 1 &&
            suspended.texture_request_valid == 1 &&
            suspended.continuation.valid == 1 &&
            suspended.written_mask == 0 &&
            suspended.executed_instruction_count == 17,
        "fill_tex_nearest FS suspends exactly once after executing SMP");
  Check(suspended.texture_request.coordinates[0] == FloatBits(0.25F) &&
            suspended.texture_request.coordinates[1] == FloatBits(0.75F) &&
            suspended.texture_request.coordinate_count == 2 &&
            suspended.texture_request.component_count == 4 &&
            suspended.texture_request.descriptor_set == 0 &&
            suspended.texture_request.binding == 0 &&
            suspended.texture_request.dimension == 2 &&
            suspended.texture_request.normalized == 1 &&
            suspended.texture_request.data_request == 0,
        "fill_tex_nearest FS emits exact live normalized coordinates");
  for (std::size_t word = 0; word < 4; ++word) {
    Check(suspended.texture_request.texture_state[word] ==
                  UINT32_C(0x10000000) + word &&
              suspended.texture_request.sampler_state[word] ==
                  UINT32_C(0x10000008) + word,
          "fill_tex_nearest FS emits live texture/sampler descriptor words");
  }
  Check(suspended.continuation.resume_instruction_index == 17 &&
            suspended.continuation.temporary_written_mask ==
                UINT32_C(0xffff0000) &&
            suspended.continuation.program_binary_size == 184 &&
            suspended.continuation.program_instruction_count == 22 &&
            suspended.continuation.pending_output_index == 0 &&
            suspended.continuation.pending_component_count == 4 &&
            suspended.continuation.data_request == 0,
        "fill_tex_nearest FS captures canonical lane continuation");

  const std::array<std::uint32_t, 4> texture_response = {
      FloatBits(0.5F), FloatBits(0.25F), FloatBits(0.125F), FloatBits(1.0F),
  };
  const auto completed = ResumeFragment(
      fragment.summary, fragment.instructions, suspended.continuation,
      texture_response);
  Check(completed.suspended == 0 &&
            completed.texture_request_valid == 0 &&
            completed.continuation.valid == 0 &&
            completed.executed_instruction_count == 5 &&
            completed.written_mask == 0x0f &&
            completed.pixel_outputs == texture_response,
        "fill_tex_nearest FS resumes at WDF and exports the live response");
  Check(suspended.executed_instruction_count +
                completed.executed_instruction_count ==
            fragment.summary.instruction_count,
        "fill_tex_nearest suspend/resume executes every group exactly once");

  const std::array<std::uint32_t, 4> changed_response = {
      UINT32_C(0x01020304), UINT32_C(0x11223344),
      UINT32_C(0x55667788), UINT32_C(0xaabbccdd),
  };
  const auto changed = ResumeFragment(
      fragment.summary, fragment.instructions, suspended.continuation,
      changed_response);
  Check(changed.pixel_outputs == changed_response,
        "fill_tex_nearest output is supplied by the texture response");
}

void TestFillTexNearestFailsClosed() {
  ExpectFailure(
      [&] {
        (void)Decode(ShaderStage::kFragment,
                     FillTexNearestVertexPcoBinary());
      },
      "fill_tex_nearest VS decoded as fragment");
  ExpectFailure(
      [&] {
        (void)Decode(ShaderStage::kVertex,
                     FillTexNearestFragmentPcoBinary());
      },
      "fill_tex_nearest FS decoded as vertex");

  auto truncated_vertex = FillTexNearestVertexPcoBinary();
  truncated_vertex.pop_back();
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, truncated_vertex); },
      "fill_tex_nearest VS truncation");
  auto truncated_fragment = FillTexNearestFragmentPcoBinary();
  truncated_fragment.pop_back();
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, truncated_fragment); },
      "fill_tex_nearest FS truncation");

  const auto vertex =
      Decode(ShaderStage::kVertex, FillTexNearestVertexPcoBinary());
  const std::vector<std::uint32_t> inputs = {
      FloatBits(-1.0F), FloatBits(1.0F), FloatBits(0.0F), FloatBits(1.0F),
  };
  ExpectFailure(
      [&] { (void)ExecuteVertex(vertex.summary, vertex.instructions, inputs); },
      "fill_tex_nearest VS missing SH0 context");
  PcoVertexExecutionContext vertex_context;
  vertex_context.shared_count = 0;
  vertex_context.shared_registers[0] = FloatBits(1.0F);
  ExpectFailure(
      [&] {
        (void)ExecuteVertex(vertex.summary, vertex.instructions, inputs,
                            vertex_context);
      },
      "fill_tex_nearest VS truncated shared span");

  const auto fragment =
      Decode(ShaderStage::kFragment, FillTexNearestFragmentPcoBinary());
  ExpectFailure(
      [&] { (void)ExecuteFragment(fragment.summary, fragment.instructions); },
      "fill_tex_nearest FS missing coefficient/shared context");
  auto short_coefficients = MakeFillTexNearestContext();
  short_coefficients.coefficient_count = 11;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              short_coefficients);
      },
      "fill_tex_nearest FS truncated coefficient span");
  auto short_shared = MakeFillTexNearestContext();
  short_shared.shared_count = 19;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              short_shared);
      },
      "fill_tex_nearest FS truncated shared span");
  auto replay_response = MakeFillTexNearestContext();
  replay_response.texture_response_valid = 1;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              replay_response);
      },
      "texture response cannot replay a fresh shader from pc zero");

  const auto suspended = ExecuteFragment(
      fragment.summary, fragment.instructions, MakeFillTexNearestContext());
  const std::array<std::uint32_t, 4> response = {
      UINT32_C(1), UINT32_C(2), UINT32_C(3), UINT32_C(4),
  };
  auto continuation = suspended.continuation;
  ++continuation.program_binary_size;
  ExpectFailure(
      [&] {
        (void)ResumeFragment(fragment.summary, fragment.instructions,
                             continuation, response);
      },
      "fragment continuation binary identity mutation");
  continuation = suspended.continuation;
  --continuation.program_instruction_count;
  ExpectFailure(
      [&] {
        (void)ResumeFragment(fragment.summary, fragment.instructions,
                             continuation, response);
      },
      "fragment continuation instruction identity mutation");
  continuation = suspended.continuation;
  continuation.resume_instruction_index = 16;
  ExpectFailure(
      [&] {
        (void)ResumeFragment(fragment.summary, fragment.instructions,
                             continuation, response);
      },
      "fragment continuation resume-PC mutation");
  continuation = suspended.continuation;
  continuation.temporary_written_mask ^= UINT32_C(1);
  ExpectFailure(
      [&] {
        (void)ResumeFragment(fragment.summary, fragment.instructions,
                             continuation, response);
      },
      "fragment continuation temporary mask mutation");
  continuation = suspended.continuation;
  continuation.pending_component_count = 3;
  ExpectFailure(
      [&] {
        (void)ResumeFragment(fragment.summary, fragment.instructions,
                             continuation, response);
      },
      "fragment continuation response width mutation");
  continuation = suspended.continuation;
  continuation.valid = 0;
  ExpectFailure(
      [&] {
        (void)ResumeFragment(fragment.summary, fragment.instructions,
                             continuation, response);
      },
      "invalid fragment continuation cannot resume");

  auto semantic = fragment;
  semantic.instructions[16].source.index = 17;
  const auto reallocated = ExecuteFragment(
      semantic.summary, semantic.instructions, MakeFillTexNearestContext());
  Check(reallocated.texture_request.coordinates[0] == FloatBits(0.75F) &&
            reallocated.texture_request.coordinates[1] == FloatBits(0.0F),
        "generic SMP accepts another initialized two-TEMP coordinate base");
  semantic.instructions[16].source.index = 31;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeFillTexNearestContext());
      },
      "serialized SMP coordinate range exceeds TEMP31");
}

void TestDecodeAndExecuteGlmarkTexture() {
  const auto fragment_binary = BytesFromHex(R"hex(
56 a0 00 b0 03 c4 40 10 c0 40 00 ff 02 80 6a ff
34 82 00 87 41 00 00 43 34 82 00 87 42 00 00 44
57 a0 00 f4 4c 80 63 80 08 88 80 a3 00 ff 02 80
6a ff 35 82 00 40 c3 a0 00 00 47 ff 35 82 00 40
c4 a0 00 00 48 ff 35 82 00 40 c5 a0 00 00 40 ff
34 8a 00 87 47 00 00 20 34 8a 00 87 48 00 00 21
34 8a 00 87 40 00 00 22 34 8a 80 87 46 00 00 23
)hex");
  Check(fragment_binary.size() == 112 &&
            Fnv1a64(fragment_binary) == UINT64_C(0x35b7502216621b6d),
        "glmark texture FS exact bytes and FNV");

  const auto fragment = Decode(ShaderStage::kFragment, fragment_binary);
  Check(fragment.summary.group_count == 13 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.instructions[4].opcode == PcoOpcode::kTextureSample &&
            fragment.instructions[4].source.bank ==
                PcoRegisterBank::kTemporary &&
            fragment.instructions[4].source.index == 3 &&
            fragment.instructions[4].source1.bank ==
                PcoRegisterBank::kShared &&
            fragment.instructions[4].source1.index == 0 &&
            fragment.instructions[4].source2.bank ==
                PcoRegisterBank::kShared &&
            fragment.instructions[4].source2.index == 8 &&
            fragment.instructions[4].output_index == 3 &&
            fragment.instructions[4].component_count == 4,
        "glmark texture FS decodes allocated SMP registers");
  const auto static_counts =
      CountPcoInstructions(fragment.instructions, false);
  const auto dynamic_counts =
      CountPcoInstructions(fragment.instructions, true);
  Check(static_counts.alu == 10 && static_counts.texture == 1 &&
            static_counts.memory == 0 && dynamic_counts.alu == 10 &&
            dynamic_counts.texture == 1 && dynamic_counts.memory == 0,
        "glmark texture FS instruction classes");

  PcoFragmentExecutionContext context;
  context.coefficient_count = 16;
  context.coefficients[2] = FloatBits(1.0F);
  context.coefficients[6] = FloatBits(0.5F);
  /* U deliberately differs by one ULP if the two FITRP plane FMAs are
   * reassociated.  The public llvmpipe ABI is
   * fma(B, y, fma(A, x, C)), which produces 0x3ee73108 here; reversing the
   * nesting produces 0x3ee73107. */
  context.coefficients[8] = UINT32_C(0xbc369490);
  context.coefficients[9] = UINT32_C(0x3cbf4254);
  context.coefficients[10] = UINT32_C(0x3f2e8cc9);
  context.coefficients[14] = FloatBits(0.75F);
  context.sample_x = UINT32_C(0x42049eba);
  context.sample_y = UINT32_C(0x40bec4a6);
  context.shared_count = 20;
  for (std::size_t index = 0; index < context.shared_count; ++index)
    context.shared_registers[index] = static_cast<std::uint32_t>(0x100 + index);

  const auto suspended =
      ExecuteFragment(fragment.summary, fragment.instructions, context);
  Check(suspended.suspended == 1 && suspended.texture_request_valid == 1 &&
            suspended.executed_instruction_count == 5 &&
            suspended.texture_request.coordinates[0] ==
                UINT32_C(0x3ee73108) &&
            suspended.texture_request.coordinates[1] == FloatBits(0.75F) &&
            suspended.texture_request.texture_state[0] == UINT32_C(0x100) &&
            suspended.texture_request.sampler_state[0] == UINT32_C(0x108) &&
            suspended.continuation.temporary_written_mask == UINT32_C(0x1f) &&
            suspended.continuation.resume_instruction_index == 5 &&
            suspended.continuation.pending_output_index == 3 &&
            suspended.continuation.pending_component_count == 4,
        "glmark texture FS suspends once with allocated coordinate state");

  const auto resumed = ResumeFragment(
      fragment.summary, fragment.instructions, suspended.continuation,
      {FloatBits(0.8F), FloatBits(0.6F), FloatBits(0.4F), FloatBits(1.0F)});
  Check(resumed.suspended == 0 && resumed.written_mask == 0x0f &&
            resumed.executed_instruction_count == 8 &&
            resumed.pixel_outputs[0] == FloatBits(0.4F) &&
            resumed.pixel_outputs[1] == FloatBits(0.3F) &&
            resumed.pixel_outputs[2] == FloatBits(0.2F) &&
            resumed.pixel_outputs[3] == FloatBits(1.0F),
        "glmark texture FS resumes at WDF without replaying SMP");

  /* llvmpipe performs perspective correction as two rounded binary32
   * operations, rcp(w) followed by numerator * rcp(w).  For this actual
   * glmark W/numerator pair, replacing those operations with one FDIV moves
   * U from 0x3c4ccb96 to adjacent 0x3c4ccb97. */
  PcoFragmentExecutionContext perspective_context;
  perspective_context.coefficient_count = 16;
  perspective_context.coefficients[2] = UINT32_C(0x3e9e69d0);
  perspective_context.coefficients[10] = UINT32_C(0x3b7d749a);
  perspective_context.shared_count = 20;
  const auto perspective_suspended = ExecuteFragment(
      fragment.summary, fragment.instructions, perspective_context);
  Check(perspective_suspended.suspended == 1 &&
            perspective_suspended.texture_request_valid == 1 &&
            perspective_suspended.texture_request.coordinates[0] ==
                UINT32_C(0x3c4ccb96) &&
            perspective_suspended.texture_request.coordinates[1] == 0,
        "FITRP preserves llvmpipe reciprocal-then-multiply rounding");

  auto tampered_continuation = suspended.continuation;
  tampered_continuation.temporary_written_mask |= UINT32_C(1) << 5U;
  ExpectFailure(
      [&] {
        (void)ResumeFragment(
            fragment.summary, fragment.instructions, tampered_continuation,
            {FloatBits(0.8F), FloatBits(0.6F), FloatBits(0.4F),
             FloatBits(1.0F)});
      },
      "glmark texture continuation has an invented pre-SMP TEMP write");

  auto response_overflow = fragment_binary;
  response_overflow[0x2b] = 0xbf;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, response_overflow); },
      "glmark texture SMP response exceeds TEMP31");
}

void TestDecodeAndExecuteGlmarkTextureMediump() {
  /* Complete 304-byte FS emitted after the texture mediump lowering.  SHA-256:
   * 2341ae9f5061213fd9a9436fab22981efb92e8e8747d2efd2c11cac87354c325. */
  const auto fragment_binary = BytesFromHex(R"hex(
56 a0 00 b0 03 c4 40 10 c0 40 00 ff 02 80 6a ff
34 82 00 87 41 00 00 43 34 82 00 87 42 00 00 44
57 a0 00 f4 4c 80 63 80 08 88 80 a3 00 ff 02 80
6a ff 57 92 00 9c 0e 80 40 a0 40 10 00 2c 40 ff
57 92 00 9c 4e 80 40 a0 43 10 00 2c 41 ff 57 92
00 9c 4e 80 40 a0 44 10 00 2c 42 ff 57 92 00 9c
4e 80 40 a0 45 10 00 2c 47 ff 57 92 00 9c 4e 80
40 a0 46 10 00 2c 43 ff 35 82 00 9c 0e 41 00 00
41 ff 35 82 00 9c 0e 40 00 00 40 ff 35 82 00 40
c1 a0 00 00 41 ff 57 92 00 9c 0e 80 40 a0 41 10
00 2c 41 ff 35 82 00 9c 0e 42 00 00 42 ff 35 82
00 40 c2 a0 00 00 42 ff 57 92 00 9c 0e 80 40 a0
42 10 00 2c 42 ff 35 82 00 9c 0e 47 00 00 44 ff
35 82 00 40 c4 a0 00 00 40 ff 57 92 00 9c 0e 80
40 a0 40 10 00 2c 40 ff 35 82 00 9c 0e 41 00 00
41 ff 34 8a 00 87 41 00 00 20 35 82 00 9c 0e 42
00 00 41 ff 34 8a 00 87 41 00 00 21 35 82 00 9c
0e 40 00 00 40 ff 34 8a 00 87 40 00 00 22 35 82
00 9c 0e 43 00 00 40 ff 34 8a 80 87 40 00 00 23
)hex");
  Check(fragment_binary.size() == 304 &&
            Fnv1a64(fragment_binary) == UINT64_C(0x1e2d215432179c29),
        "glmark mediump texture FS exact bytes and FNV");

  const auto fragment = Decode(ShaderStage::kFragment, fragment_binary);
  Check(fragment.summary.group_count == 29 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.instructions[4].opcode == PcoOpcode::kTextureSample,
        "glmark mediump texture FS decodes the complete real program");
  std::size_t rtne_count = 0;
  std::size_t rtz_count = 0;
  std::size_t unpack_count = 0;
  for (const auto &instruction : fragment.instructions) {
    rtne_count +=
        instruction.opcode == PcoOpcode::kFloatPackHalfRtne ? 1U : 0U;
    rtz_count +=
        instruction.opcode == PcoOpcode::kFloatPackHalfRtz ? 1U : 0U;
    unpack_count +=
        instruction.opcode == PcoOpcode::kFloatUnpackHalf ? 1U : 0U;
  }
  Check(rtne_count == 4 && rtz_count == 4 && unpack_count == 8 &&
            fragment.instructions[6].opcode ==
                PcoOpcode::kFloatPackHalfRtne &&
            fragment.instructions[7].opcode ==
                PcoOpcode::kFloatPackHalfRtz,
        "glmark mediump graph preserves RTNE light/results and RTZ sample RGBA");

  PcoFragmentExecutionContext context;
  context.coefficient_count = 16;
  context.coefficients[2] = FloatBits(1.0F);
  context.coefficients[6] = FloatBits(0.5F);
  context.coefficients[10] = FloatBits(0.25F);
  context.coefficients[14] = FloatBits(0.75F);
  context.shared_count = 20;
  for (std::size_t index = 0; index < context.shared_count; ++index)
    context.shared_registers[index] = static_cast<std::uint32_t>(0x200 + index);

  const auto suspended =
      ExecuteFragment(fragment.summary, fragment.instructions, context);
  Check(suspended.suspended == 1 && suspended.executed_instruction_count == 5 &&
            suspended.texture_request.coordinates[0] == FloatBits(0.25F) &&
            suspended.texture_request.coordinates[1] == FloatBits(0.75F),
        "glmark mediump FS reaches its real SMP before conversion");
  const auto resumed = ResumeFragment(
      fragment.summary, fragment.instructions, suspended.continuation,
      {FloatBits(0.8F), FloatBits(0.6F), FloatBits(0.4F), FloatBits(1.0F)});
  Check(resumed.suspended == 0 && resumed.written_mask == 0x0f &&
            resumed.executed_instruction_count == 24 &&
            resumed.pixel_outputs[0] == FloatBits(0.39990234375F) &&
            resumed.pixel_outputs[1] == FloatBits(0.2998046875F) &&
            resumed.pixel_outputs[2] == FloatBits(0.199951171875F) &&
            resumed.pixel_outputs[3] == FloatBits(1.0F),
        "glmark mediump FS executes RTZ sample conversion then RTNE products");

  auto rtne_sample = fragment_binary;
  rtne_sample[0x52] = 0x0e;
  const auto rtne_fragment = Decode(ShaderStage::kFragment, rtne_sample);
  const auto rtne_suspended = ExecuteFragment(
      rtne_fragment.summary, rtne_fragment.instructions, context);
  const auto rtne_resumed = ResumeFragment(
      rtne_fragment.summary, rtne_fragment.instructions,
      rtne_suspended.continuation,
      {FloatBits(0.8F), FloatBits(0.6F), FloatBits(0.4F), FloatBits(1.0F)});
  Check(rtne_fragment.instructions[8].opcode ==
                PcoOpcode::kFloatPackHalfRtne &&
            rtne_resumed.pixel_outputs[1] != resumed.pixel_outputs[1],
        "changing the real sample pack from RTZ to RTNE changes arithmetic");

  auto invalid_rounding = fragment_binary;
  invalid_rounding[0x52] = 0x2e;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, invalid_rounding); },
      "glmark mediump PCK rejects an unknown rounding encoding");
}

void TestExecuteFailsClosed() {
  auto decoded = Decode(ShaderStage::kFragment, FillSolidFragmentPcoBinary());
  decoded.instructions[0].opcode = PcoOpcode::kUvsWrite;
  ExpectFailure(
      [&] { (void)ExecuteFragment(decoded.summary, decoded.instructions); },
      "tampered serialized decoded opcode");

  decoded = Decode(ShaderStage::kFragment, FillSolidFragmentPcoBinary());
  --decoded.summary.instruction_count;
  ExpectFailure(
      [&] { (void)ExecuteFragment(decoded.summary, decoded.instructions); },
      "tampered serialized program summary");

  decoded =
      Decode(ShaderStage::kFragment, FillSolidRedHalfAlphaFragmentPcoBinary());
  decoded.instructions[3].source.index = 74;
  ExpectFailure(
      [&] { (void)ExecuteFragment(decoded.summary, decoded.instructions); },
      "tampered decoded special constant must fail closed at execution");
}

void TestDecodeAndExecuteConditionals() {
  const auto &vertex_binary = ConditionalsVertexPcoBinary();
  const auto vertex = Decode(ShaderStage::kVertex, vertex_binary);
  Check(vertex_binary.size() == 520,
        "conditionals VS preserves the generated PCO byte count");
  Check(vertex.summary.group_count == 48 &&
            vertex.summary.instruction_count == 48 &&
            vertex.instructions.size() == 48,
        "conditionals VS decodes all 48 real PCO groups");
  Check(vertex.summary.vertex_input_mask == 0x7 &&
            vertex.summary.vertex_output_mask == UINT64_C(0xf) &&
            vertex.summary.ends_task == 1,
        "conditionals VS exposes the exact input/output contract");
  const auto vertex_static =
      CountPcoInstructions(vertex.instructions, false);
  const auto vertex_dynamic =
      CountPcoInstructions(vertex.instructions, true);
  Check(vertex_static.alu == 46 && vertex_static.texture == 0 &&
            vertex_static.memory == 2,
        "conditionals VS static instruction classes");
  Check(vertex_dynamic.alu == 46 && vertex_dynamic.texture == 0 &&
            vertex_dynamic.memory == 5,
        "conditionals VS expands the four-component UVSW");

  PcoVertexExecutionContext vertex_context;
  vertex_context.shared_count = 16;
  vertex_context.shared_registers[0] = FloatBits(1.0F);
  vertex_context.shared_registers[5] = FloatBits(1.0F);
  vertex_context.shared_registers[10] = FloatBits(1.0F);
  vertex_context.shared_registers[15] = FloatBits(1.0F);
  const auto below = ExecuteVertex(
      vertex.summary, vertex.instructions,
      {FloatBits(1.25F), FloatBits(2.0F), FloatBits(0.5F)}, vertex_context);
  Check(below.written_mask == UINT64_C(0xf) && below.emitted == 1 &&
            below.ended_task == 1,
        "conditionals VS exports one complete position");
  Check(below.outputs[0] == FloatBits(1.25F) &&
            below.outputs[1] == FloatBits(2.01875F) &&
            below.outputs[2] == FloatBits(0.5F) &&
            below.outputs[3] == FloatBits(1.0F),
        "conditionals VS executes the false ternary branch and matrix path");
  const auto above = ExecuteVertex(
      vertex.summary, vertex.instructions,
      {FloatBits(1.75F), FloatBits(2.0F), FloatBits(0.5F)}, vertex_context);
  Check(above.outputs[0] == FloatBits(1.75F) &&
            above.outputs[1] == FloatBits(2.0375F) &&
            above.outputs[2] == FloatBits(0.5F) &&
            above.outputs[3] == FloatBits(1.0F),
        "conditionals VS executes the true ternary branch and matrix path");

  const auto &fragment_binary = ConditionalsFragmentPcoBinary();
  const auto fragment = Decode(ShaderStage::kFragment, fragment_binary);
  Check(fragment_binary.size() == 520,
        "conditionals FS preserves the generated PCO byte count");
  Check(fragment.summary.group_count == 43 &&
            fragment.summary.instruction_count == 43 &&
            fragment.instructions.size() == 43,
        "conditionals FS decodes all 43 real PCO groups");
  Check(fragment.summary.pixel_output_mask == 0x0f &&
            fragment.summary.early_hsr_safe == 1,
        "conditionals FS exposes RGBA output and early-HSR safety");
  const auto fragment_static =
      CountPcoInstructions(fragment.instructions, false);
  Check(fragment_static.alu == 43 && fragment_static.texture == 0 &&
            fragment_static.memory == 0,
        "conditionals FS instruction classes");
  std::size_t pack_count = 0;
  std::size_t unpack_count = 0;
  for (const auto &instruction : fragment.instructions) {
    pack_count +=
        instruction.opcode == PcoOpcode::kFloatPackHalfRtne ? 1U : 0U;
    unpack_count +=
        instruction.opcode == PcoOpcode::kFloatUnpackHalf ? 1U : 0U;
  }
  Check(pack_count == 5 && unpack_count == 5,
        "conditionals FS preserves all scalar binary16 round trips");

  PcoFragmentExecutionContext fragment_context;
  fragment_context.shared_count = 4;
  fragment_context.shared_registers = {
      FloatBits(1.0F), FloatBits(0.0F), FloatBits(-1.0F), FloatBits(60.0F)};
  fragment_context.sample_x = FloatBits(75.0F);
  fragment_context.sample_y = FloatBits(75.0F);
  const auto pixel = ExecuteFragment(fragment.summary, fragment.instructions,
                                     fragment_context);
  Check(pixel.written_mask == 0x0f && pixel.discarded == false,
        "conditionals FS writes one RGBA pixel");
  Check(pixel.pixel_outputs[0] == FloatBits(0.125F) &&
            pixel.pixel_outputs[1] == FloatBits(0.125F) &&
            pixel.pixel_outputs[2] == FloatBits(0.125F) &&
            pixel.pixel_outputs[3] == FloatBits(1.0F),
        "conditionals FS executes frag-coord math and the true ternary branch");
  fragment_context.sample_x = FloatBits(100.0F);
  fragment_context.sample_y = FloatBits(25.0F);
  const auto alternate_pixel = ExecuteFragment(
      fragment.summary, fragment.instructions, fragment_context);
  Check(alternate_pixel.pixel_outputs[0] == FloatBits(0.75F) &&
            alternate_pixel.pixel_outputs[1] == FloatBits(0.75F) &&
            alternate_pixel.pixel_outputs[2] == FloatBits(0.75F) &&
            alternate_pixel.pixel_outputs[3] == FloatBits(1.0F),
        "conditionals FS executes frag-coord math and the false ternary branch");

  fragment_context.sample_x = FloatBits(34.5F);
  fragment_context.sample_y = FloatBits(55.5F);
  const auto mediump_pixel = ExecuteFragment(
      fragment.summary, fragment.instructions, fragment_context);
  Check(mediump_pixel.pixel_outputs[0] == FloatBits(0.57470703125F) &&
            mediump_pixel.pixel_outputs[1] == FloatBits(0.57470703125F) &&
            mediump_pixel.pixel_outputs[2] == FloatBits(0.57470703125F) &&
            mediump_pixel.pixel_outputs[3] == FloatBits(1.0F),
        "conditionals FS rounds each mediump ALU result through binary16");

  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, vertex_binary); },
      "conditionals VS binary in the fragment stage");
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, fragment_binary); },
      "conditionals FS binary in the vertex stage");

  PcoVertexExecutionContext short_vertex_context = vertex_context;
  short_vertex_context.shared_count = 15;
  ExpectFailure(
      [&] {
        (void)ExecuteVertex(vertex.summary, vertex.instructions,
                            {FloatBits(1.25F), FloatBits(2.0F),
                             FloatBits(0.5F)},
                            short_vertex_context);
      },
      "conditionals VS truncated shared-register ABI");
  PcoFragmentExecutionContext short_fragment_context = fragment_context;
  short_fragment_context.shared_count = 3;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              short_fragment_context);
      },
      "conditionals FS truncated shared-register ABI");

  auto tampered_vertex = vertex;
  tampered_vertex.instructions[11].immediate ^= UINT32_C(1);
  ExpectFailure(
      [&] {
        (void)ExecuteVertex(tampered_vertex.summary,
                            tampered_vertex.instructions,
                            {FloatBits(1.25F), FloatBits(2.0F),
                             FloatBits(0.5F)},
                            vertex_context);
      },
      "conditionals VS semantic immediate mutation");
  auto tampered_fragment = fragment;
  tampered_fragment.instructions[15].immediate ^= UINT32_C(1);
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(tampered_fragment.summary,
                              tampered_fragment.instructions,
                              fragment_context);
      },
      "conditionals FS semantic immediate mutation");
  tampered_fragment = fragment;
  tampered_fragment.instructions[20].opcode = PcoOpcode::kPackHalf2x16;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(tampered_fragment.summary,
                              tampered_fragment.instructions,
                              fragment_context);
      },
      "conditionals FS scalar mediump-pack mutation");
}

void TestDecodeAndExecuteIdeasLightingSelect() {
  /* Reduced from the real 1128-byte Ideas lighting vertex program
   * (SHA-256 b07e4c4498a2a1f5c8b40f6ae39884d66acd2ec75c07f8f5fa84b98878bb4e2f).
   * The four public groups below are copied byte-for-byte from offsets
   * 0x38e..0x3cf and 0x42c..0x435: BCMP.F32.E, three BCSELs and MBYP.neg.
   * MOVI/UVSW scaffolding makes both select paths independently executable. */
  const auto vertex_binary = BytesFromHex(R"hex(
86 92 40 13 00 00 80 40 00 00 44 ff
86 92 40 13 00 00 a0 40 00 00 45 ff
86 92 40 13 00 00 c0 40 00 00 46 ff
86 92 40 13 00 00 00 00 00 00 47 ff
86 92 40 13 00 00 40 41 00 00 4c ff
86 92 40 13 00 00 50 41 00 00 4d ff
86 92 40 13 00 00 60 41 00 00 4e ff
89 d2 00 d3 3c e8 9c 1e 87 87 c7 cf 80 11 00 20 4f ff
78 d2 00 d1 3c f0 b0 87 87 c4 ef 80 10 4d 01 4d
78 d2 00 d1 3c f0 b0 87 87 c5 ef 80 10 4e 01 4e
78 d2 00 d1 3c f0 b0 87 87 c6 ef 80 10 4c 01 4c
35 82 00 97 02 4d 00 00 4b ff
55 a0 06 08 00 cb 00 00 00 30
44 a0 80 05 00 00 00 ff
)hex");

  Check(vertex_binary.size() == 178,
        "Ideas select fixture preserves exact public group sizes");
  const auto decoded = Decode(ShaderStage::kVertex, vertex_binary);
  Check(decoded.summary.group_count == 14 &&
            decoded.summary.instruction_count == 14 &&
            decoded.summary.vertex_input_mask == 0 &&
            decoded.summary.vertex_output_mask == UINT64_C(0x0f) &&
            decoded.summary.ends_task == 1,
        "Ideas select fixture decodes the exact vertex ABI");
  Check(decoded.instructions[7].opcode == PcoOpcode::kFloatEqual &&
            decoded.instructions[7].source.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[7].source.index == 7 &&
            decoded.instructions[7].source1.bank == PcoRegisterBank::kSpecial &&
            decoded.instructions[7].source1.index == 0 &&
            decoded.instructions[7].output_index == 15,
        "Ideas BCMP.F32.E compares TEMP7 with positive zero");
  for (std::size_t index = 8; index <= 10; ++index) {
    Check(decoded.instructions[index].opcode ==
                  PcoOpcode::kConditionalSelect &&
              decoded.instructions[index].source.bank ==
                  PcoRegisterBank::kTemporary &&
              decoded.instructions[index].source.index == 15 &&
              decoded.instructions[index].source_count == 3,
          "Ideas BCSEL preserves its Boolean condition and three sources");
  }
  Check(decoded.instructions[8].source1.index == 4 &&
            decoded.instructions[8].source2.index == 13 &&
            decoded.instructions[8].output_index == 13 &&
            decoded.instructions[9].source1.index == 5 &&
            decoded.instructions[9].source2.index == 14 &&
            decoded.instructions[9].output_index == 14 &&
            decoded.instructions[10].source1.index == 6 &&
            decoded.instructions[10].source2.index == 12 &&
            decoded.instructions[10].output_index == 12,
        "Ideas BCSEL register routing is exact");
  Check(decoded.instructions[11].opcode == PcoOpcode::kFloatNegate &&
            decoded.instructions[11].source.index == 13 &&
            decoded.instructions[11].output_index == 11 &&
            decoded.instructions[11].source_count == 1,
        "Ideas MBYP.neg routes TEMP13 to TEMP11");

  const auto equal_zero =
      ExecuteVertex(decoded.summary, decoded.instructions, {});
  Check(equal_zero.written_mask == UINT64_C(0x0f) &&
            equal_zero.outputs[0] == FloatBits(-4.0F) &&
            equal_zero.outputs[1] == FloatBits(6.0F) &&
            equal_zero.outputs[2] == FloatBits(4.0F) &&
            equal_zero.outputs[3] == FloatBits(5.0F) &&
            equal_zero.emitted == 1 && equal_zero.ended_task == 1,
        "Ideas BCMP true path selects, negates and exports exact values");

  auto not_equal_binary = vertex_binary;
  not_equal_binary[40] = 0x00;
  not_equal_binary[41] = 0x00;
  not_equal_binary[42] = 0x80;
  not_equal_binary[43] = 0x3f;
  const auto not_equal = Decode(ShaderStage::kVertex, not_equal_binary);
  const auto unequal =
      ExecuteVertex(not_equal.summary, not_equal.instructions, {});
  Check(unequal.outputs[0] == FloatBits(-13.0F) &&
            unequal.outputs[1] == FloatBits(12.0F) &&
            unequal.outputs[2] == FloatBits(13.0F) &&
            unequal.outputs[3] == FloatBits(14.0F),
        "Ideas BCMP false path preserves the fallback values");

  auto mutation = vertex_binary;
  mutation[90] ^= 0x01;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, mutation); },
                "Ideas BCMP phase sequence mutation");
  mutation = vertex_binary;
  mutation[95] ^= 0x01;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, mutation); },
                "Ideas BCMP internal true source mutation");
  mutation = vertex_binary;
  mutation[114] = 0x00;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, mutation); },
                "Ideas BCSEL is0=s1 selector mutation");
  mutation = vertex_binary;
  mutation[154] = 0x00;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, mutation); },
                "Ideas MBYP negate modifier mutation");
}

void TestDecodeAndExecuteShadowFloatLess() {
  /* Exact 216-byte MASK fragment PCO from the public glmark2 Shadow capture
   * (FNV-1a64 1ac54b25af8de102).  Groups 7 and 10 are Mesa's five-phase
   * BCMP.F32.L forms: ordered 0 < W and sampled-red < z/W + 0.1505. */
  const auto fragment_binary = BytesFromHex(R"hex(
56 A0 00 B0 04 C4 40 10 C0 40 00 FF 02 80 6A FF
34 82 00 80 43 00 00 44 35 82 00 40 C0 A4 00 00
45 FF 35 82 00 40 C1 A4 00 00 46 FF 57 A0 00 F4
4C 80 65 80 08 88 80 A5 00 FF 02 80 6A FF 99 D2
00 D3 3C F0 01 9C 1E 87 87 80 CF 80 11 43 20 49
86 92 40 13 AC 1C 1A 3E 00 00 4A FF 36 82 00 C0
C2 64 00 4A 00 00 40 FF 99 D2 00 D3 3C F0 01 9C
1E 87 87 C5 CF 80 11 40 20 40 56 B2 40 41 02 80
40 00 49 40 40 FF 86 92 40 13 9A 99 99 3E 00 00
41 FF 86 92 40 13 9A 99 19 3E 00 00 42 FF 78 D2
00 D1 3C F0 B0 87 87 C2 E0 80 10 41 01 40 34 8A
00 87 40 00 00 20 34 8A 00 87 40 00 00 21 35 8A
00 87 40 00 00 22 F1 FF 38 8A 80 87 80 01 00 00
00 23 F3 FF FF FF FF FF
)hex");
  Check(fragment_binary.size() == 216 &&
            Fnv1a64(fragment_binary) == UINT64_C(0x1ac54b25af8de102),
        "Shadow MASK fixture preserves its exact binary and public hash");

  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.binary_size == 216 &&
            decoded.summary.group_count == 19 &&
            decoded.summary.instruction_count == 19 &&
            decoded.summary.pixel_output_mask == 0x0f,
        "Shadow MASK decodes its complete fragment ABI");
  Check(decoded.instructions[7].opcode == PcoOpcode::kFloatLess &&
            decoded.instructions[7].source.bank ==
                PcoRegisterBank::kSpecial &&
            decoded.instructions[7].source.index == 0 &&
            decoded.instructions[7].source1.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[7].source1.index == 3 &&
            decoded.instructions[7].output_index == 9 &&
            decoded.instructions[10].opcode == PcoOpcode::kFloatLess &&
            decoded.instructions[10].source.index == 5 &&
            decoded.instructions[10].source1.index == 0 &&
            decoded.instructions[10].output_index == 0,
        "Shadow BCMP.F32.L preserves ordered operand direction and targets");

  PcoFragmentExecutionContext context;
  context.coefficient_count = 20;
  context.shared_count = 20;
  context.sample_x = FloatBits(0.0F);
  context.sample_y = FloatBits(0.0F);
  context.coefficients[2] = FloatBits(1.0F);  // perspective W
  context.coefficients[6] = FloatBits(0.25F);
  context.coefficients[10] = FloatBits(0.75F);
  context.coefficients[14] = FloatBits(0.10F); // z
  context.coefficients[18] = FloatBits(1.0F);  // varying W

  const auto execute_with_red = [&](std::uint32_t red) {
    const auto suspended =
        ExecuteFragment(decoded.summary, decoded.instructions, context);
    Check(suspended.suspended == 1 &&
              suspended.texture_request_valid == 1 &&
              suspended.continuation.valid == 1 &&
              suspended.texture_request.descriptor_set == 0,
          "Shadow MASK reaches its real SMP before BCMP");
    return ResumeFragment(
        decoded.summary, decoded.instructions, suspended.continuation,
        {red, FloatBits(0.0F), FloatBits(0.0F), FloatBits(1.0F)});
  };

  const auto shadowed = execute_with_red(FloatBits(0.20F));
  Check(shadowed.written_mask == 0x0f && !shadowed.discarded &&
            shadowed.pixel_outputs[0] == UINT32_C(0x3e19999a) &&
            shadowed.pixel_outputs[1] == UINT32_C(0x3e19999a) &&
            shadowed.pixel_outputs[2] == UINT32_C(0x3e19999a) &&
            shadowed.pixel_outputs[3] == FloatBits(1.0F),
        "Shadow ordered less-than true path selects exact 0.15 mask");
  const auto lit = execute_with_red(FloatBits(0.50F));
  Check(lit.pixel_outputs[0] == UINT32_C(0x3e99999a) &&
            lit.pixel_outputs[1] == UINT32_C(0x3e99999a) &&
            lit.pixel_outputs[2] == UINT32_C(0x3e99999a),
        "Shadow ordered less-than false path selects exact 0.3 mask");
  const auto unordered = execute_with_red(UINT32_C(0x7fc00000));
  Check(unordered.pixel_outputs[0] == UINT32_C(0x3e99999a),
        "Shadow ordered less-than is false for NaN");

  auto mutation = fragment_binary;
  mutation[0x43] ^= 0x01U;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Shadow BCMP.F32.L TST phase mutation");
  mutation = fragment_binary;
  mutation[0x4a] ^= 0x01U;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Shadow BCMP.F32.L internal sc143 mutation");
}

void TestDecodeAndExecuteTerrainFloatGreaterEqual() {
  /* The 18-byte BCMP group is copied byte-for-byte from Terrain D1 FS offset
   * 0xdc.  MOVI and PIXOUT groups form only a closed executable envelope for
   * its ordered r8 >= r10 compare and canonical sc143/zero MOVC result. */
  const auto make_binary = [](std::uint32_t left) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 48 ff
86 92 40 13 00 00 80 3f 00 00 4a ff
89 d2 00 d3 3c ec 9c 1e 87 87 c8 cf 80 11 4a 20 4c ff
34 8a 00 87 4c 00 00 20
34 8a 00 87 4c 00 00 21
34 8a 00 87 4c 00 00 22
34 8a 80 87 4c 00 00 23
)hex");
    binary[4] = static_cast<std::uint8_t>(left);
    binary[5] = static_cast<std::uint8_t>(left >> 8U);
    binary[6] = static_cast<std::uint8_t>(left >> 16U);
    binary[7] = static_cast<std::uint8_t>(left >> 24U);
    return binary;
  };

  const auto fragment_binary = make_binary(FloatBits(2.0F));
  Check(fragment_binary.size() == 74,
        "Terrain BCMP.F32.GE fixture preserves exact group sizes");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 7 &&
            decoded.summary.instruction_count == 7 &&
            decoded.summary.pixel_output_mask == 0x0f &&
            decoded.instructions[2].opcode ==
                PcoOpcode::kFloatGreaterEqual &&
            decoded.instructions[2].source.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[2].source.index == 8 &&
            decoded.instructions[2].source1.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[2].source1.index == 10 &&
            decoded.instructions[2].output_index == 12 &&
            decoded.instructions[2].binary_offset == 27,
        "Terrain BCMP.F32.GE preserves ordered operands and destination");

  const auto greater =
      ExecuteFragment(decoded.summary, decoded.instructions);
  Check(greater.written_mask == 0x0f &&
            greater.pixel_outputs[0] == UINT32_C(0xffffffff) &&
            greater.pixel_outputs[1] == UINT32_C(0xffffffff) &&
            greater.pixel_outputs[2] == UINT32_C(0xffffffff) &&
            greater.pixel_outputs[3] == UINT32_C(0xffffffff),
        "Terrain BCMP.F32.GE materializes canonical true bits");

  const auto less_binary = make_binary(FloatBits(0.5F));
  const auto less = Decode(ShaderStage::kFragment, less_binary);
  const auto false_result =
      ExecuteFragment(less.summary, less.instructions);
  Check(false_result.pixel_outputs[0] == 0 &&
            false_result.pixel_outputs[1] == 0 &&
            false_result.pixel_outputs[2] == 0 &&
            false_result.pixel_outputs[3] == 0,
        "Terrain BCMP.F32.GE materializes canonical false bits");

  const auto nan_binary = make_binary(UINT32_C(0x7fc00000));
  const auto nan = Decode(ShaderStage::kFragment, nan_binary);
  Check(ExecuteFragment(nan.summary, nan.instructions).pixel_outputs[0] == 0,
        "Terrain BCMP.F32.GE is an ordered comparison for NaN");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{24, 0x99},
           {29, 0xed}, {30, 0x9d}, {35, 0xce}, {39, 0x21}, {40, 0x20}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Terrain BCMP.F32.GE near-neighbor encoding mutation");
  }

  auto malformed_instructions = decoded.instructions;
  malformed_instructions[2].source0_floor = 1;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, malformed_instructions);
      },
      "Terrain BCMP.F32.GE rejects a noncanonical decoded modifier");
}

void TestDecodeAndExecuteTerrainFloatGreaterEqualOneZero() {
  /* Exact Terrain D1 FS group at offset 0xee.  Unlike the sc143 Boolean form
   * above, PCK.ONE plus MOVC materializes binary32 1.0 or 0.0. */
  const auto make_binary = [](std::uint32_t left) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 48 ff
86 92 40 13 00 00 80 3f 00 00 4a ff
89 d2 00 d2 3c ec 9c 1f 87 87 c8 c0 80 10 4a 30 4d ff
34 8a 00 87 4d 00 00 20
34 8a 00 87 4d 00 00 21
34 8a 00 87 4d 00 00 22
34 8a 80 87 4d 00 00 23
)hex");
    binary[4] = static_cast<std::uint8_t>(left);
    binary[5] = static_cast<std::uint8_t>(left >> 8U);
    binary[6] = static_cast<std::uint8_t>(left >> 16U);
    binary[7] = static_cast<std::uint8_t>(left >> 24U);
    return binary;
  };

  const auto fragment_binary = make_binary(FloatBits(2.0F));
  Check(fragment_binary.size() == 74,
        "Terrain BCMP.F32.GE.ONE fixture preserves exact group sizes");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 7 &&
            decoded.summary.instruction_count == 7 &&
            decoded.instructions[2].opcode ==
                PcoOpcode::kFloatGreaterEqual &&
            decoded.instructions[2].source.index == 8 &&
            decoded.instructions[2].source1.index == 10 &&
            decoded.instructions[2].output_index == 13 &&
            decoded.instructions[2].comparison_result_float_one == 1 &&
            decoded.instructions[2].binary_offset == 27,
        "Terrain BCMP.F32.GE.ONE preserves operands, result form and target");

  const auto greater =
      ExecuteFragment(decoded.summary, decoded.instructions);
  Check(greater.written_mask == 0x0f &&
            greater.pixel_outputs[0] == FloatBits(1.0F) &&
            greater.pixel_outputs[1] == FloatBits(1.0F) &&
            greater.pixel_outputs[2] == FloatBits(1.0F) &&
            greater.pixel_outputs[3] == FloatBits(1.0F),
        "Terrain BCMP.F32.GE.ONE materializes binary32 one on true");

  const auto less_binary = make_binary(FloatBits(0.5F));
  const auto less = Decode(ShaderStage::kFragment, less_binary);
  const auto false_result =
      ExecuteFragment(less.summary, less.instructions);
  Check(false_result.pixel_outputs[0] == 0 &&
            false_result.pixel_outputs[1] == 0 &&
            false_result.pixel_outputs[2] == 0 &&
            false_result.pixel_outputs[3] == 0,
        "Terrain BCMP.F32.GE.ONE materializes binary32 zero on false");

  const auto nan_binary = make_binary(UINT32_C(0x7fc00000));
  const auto nan = Decode(ShaderStage::kFragment, nan_binary);
  Check(ExecuteFragment(nan.summary, nan.instructions).pixel_outputs[0] == 0,
        "Terrain BCMP.F32.GE.ONE is an ordered comparison for NaN");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{24, 0x99},
           {27, 0xd3}, {29, 0xed}, {31, 0x1e}, {35, 0xc1},
           {39, 0x31}, {40, 0x20}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Terrain BCMP.F32.GE.ONE near-neighbor mutation");
  }

  auto malformed_instructions = decoded.instructions;
  malformed_instructions[2].comparison_result_float_one = 2;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, malformed_instructions);
      },
      "Terrain BCMP.F32.GE.ONE rejects a reserved result encoding");
}

void TestDecodeAndExecuteTerrainLogicalXnor() {
  /* Exact Terrain D1 FS LOGICAL.XNOR group at offset 0x112.  The MOVI and
   * PIXOUT groups expose its in-place BBYP0S1 r14 / XNOR sc0 result. */
  const auto fragment_binary = BytesFromHex(R"hex(
86 92 40 13 5a a5 f0 f0 00 00 4e ff
56 b2 40 46 02 80 40 00 4e 00 4e ff
34 8a 00 87 4e 00 00 20
34 8a 00 87 4e 00 00 21
34 8a 00 87 4e 00 00 22
34 8a 80 87 4e 00 00 23
)hex");
  Check(fragment_binary.size() == 56,
        "Terrain LOGICAL.XNOR fixture preserves exact group sizes");

  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 6 &&
            decoded.summary.instruction_count == 6 &&
            decoded.summary.pixel_output_mask == 0x0f &&
            decoded.instructions[1].opcode == PcoOpcode::kBitwiseXnor &&
            decoded.instructions[1].source.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[1].source.index == 14 &&
            decoded.instructions[1].source1.bank ==
                PcoRegisterBank::kSpecial &&
            decoded.instructions[1].source1.index == 0 &&
            decoded.instructions[1].output_index == 14 &&
            decoded.instructions[1].binary_offset == 15,
        "Terrain LOGICAL.XNOR preserves its exact source and in-place target");
  const auto counts = CountPcoInstructions(decoded.instructions, false);
  Check(counts.alu == 6 && counts.texture == 0 && counts.memory == 0,
        "Terrain LOGICAL.XNOR remains an ALU instruction-counter operation");

  const auto pixel =
      ExecuteFragment(decoded.summary, decoded.instructions);
  Check(pixel.written_mask == 0x0f &&
            pixel.pixel_outputs[0] == UINT32_C(0x0f0f5aa5) &&
            pixel.pixel_outputs[1] == UINT32_C(0x0f0f5aa5) &&
            pixel.pixel_outputs[2] == UINT32_C(0x0f0f5aa5) &&
            pixel.pixel_outputs[3] == UINT32_C(0x0f0f5aa5),
        "Terrain LOGICAL.XNOR executes exact 32-bit ~(r14 xor sc0)");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{14, 0x00},
           {15, 0x47}, {16, 0x03}, {17, 0x81}, {18, 0x41},
           {19, 0x01}, {20, 0x00}, {21, 0x01}, {22, 0x4f}, {23, 0xfe}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Terrain LOGICAL.XNOR near-neighbor mutation");
  }
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, fragment_binary); },
      "Terrain LOGICAL.XNOR is rejected from the vertex stage");

  auto malformed_instructions = decoded.instructions;
  malformed_instructions[1].source1.index = 1;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, malformed_instructions);
      },
      "Terrain LOGICAL.XNOR rejects a noncanonical decoded source");
}

void TestDecodeAndExecuteTerrainFloatMin() {
  /* Exact Terrain D1 FS FMIN group at offset 0x12e.  TST.F32.L is ordered;
   * MOVC selects P0/r13 only on true and P1/r14 on false. */
  const auto make_binary = [](std::uint32_t left, std::uint32_t right) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 4d ff
86 92 40 13 00 00 00 00 00 00 4e ff
77 d2 00 d0 3c f0 11 87 87 4d 4e 10 4f ff
34 8a 00 87 4f 00 00 20
34 8a 00 87 4f 00 00 21
34 8a 00 87 4f 00 00 22
34 8a 80 87 4f 00 00 23
)hex");
    for (unsigned byte = 0; byte < 4; ++byte) {
      binary[4 + byte] =
          static_cast<std::uint8_t>(left >> (byte * 8U));
      binary[16 + byte] =
          static_cast<std::uint8_t>(right >> (byte * 8U));
    }
    return binary;
  };
  const auto execute = [&](std::uint32_t left, std::uint32_t right) {
    const auto binary = make_binary(left, right);
    const auto decoded = Decode(ShaderStage::kFragment, binary);
    return ExecuteFragment(decoded.summary, decoded.instructions)
        .pixel_outputs[0];
  };

  const auto fragment_binary =
      make_binary(FloatBits(2.0F), FloatBits(1.0F));
  Check(fragment_binary.size() == 70,
        "Terrain FMIN fixture preserves exact group sizes");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 7 &&
            decoded.summary.instruction_count == 7 &&
            decoded.instructions[2].opcode == PcoOpcode::kFloatMin &&
            decoded.instructions[2].source.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[2].source.index == 13 &&
            decoded.instructions[2].source1.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[2].source1.index == 14 &&
            decoded.instructions[2].output_index == 15 &&
            decoded.instructions[2].binary_offset == 27,
        "Terrain FMIN preserves its P0/P1 source order and destination");
  Check(execute(FloatBits(2.0F), FloatBits(1.0F)) == FloatBits(1.0F) &&
            execute(FloatBits(0.5F), FloatBits(1.0F)) == FloatBits(0.5F),
        "Terrain ordered FMIN selects the smaller finite source");

  constexpr std::uint32_t kLeftNan = UINT32_C(0x7fc12345);
  constexpr std::uint32_t kRightNan = UINT32_C(0x7fc23456);
  Check(execute(kLeftNan, FloatBits(1.0F)) == FloatBits(1.0F) &&
            execute(FloatBits(1.0F), kRightNan) == kRightNan,
        "Terrain ordered FMIN false predicate selects the exact right NaN/source");
  Check(execute(UINT32_C(0x00000000), UINT32_C(0x80000000)) ==
                UINT32_C(0x80000000) &&
            execute(UINT32_C(0x80000000), UINT32_C(0x00000000)) ==
                UINT32_C(0x00000000),
        "Terrain ordered FMIN equal signed zeros select the right source");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{26, 0x80},
           {27, 0xd1}, {28, 0x3d}, {29, 0xf1}, {30, 0x10},
           {31, 0x86}, {32, 0x86}, {33, 0x80}, {34, 0x80},
           {35, 0x11}, {36, 0x20}, {37, 0xfe}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Terrain FMIN near-neighbor encoding mutation");
  }

  auto malformed_instructions = decoded.instructions;
  malformed_instructions[2].source0_floor = 1;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, malformed_instructions);
      },
      "Terrain FMIN rejects a noncanonical decoded source modifier");
}

void TestDecodeAndExecuteTerrainFloatAbs() {
  /* Exact Terrain D1 FS scalar FABS group at offset 0x7b8. */
  const auto make_fragment_binary = [](std::uint32_t input) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 52 ff
35 82 00 97 01 52 00 00 5d ff
34 8a 00 87 5d 00 00 20
34 8a 00 87 5d 00 00 21
34 8a 00 87 5d 00 00 22
34 8a 80 87 5d 00 00 23
)hex");
    for (unsigned byte = 0; byte < 4; ++byte) {
      binary[4 + byte] =
          static_cast<std::uint8_t>(input >> (byte * 8U));
    }
    return binary;
  };
  const auto execute_fragment = [&](std::uint32_t input) {
    const auto binary = make_fragment_binary(input);
    const auto decoded = Decode(ShaderStage::kFragment, binary);
    return ExecuteFragment(decoded.summary, decoded.instructions)
        .pixel_outputs[0];
  };

  const auto fragment_binary = make_fragment_binary(FloatBits(-2.0F));
  Check(fragment_binary.size() == 54,
        "Terrain FABS fixture preserves exact group sizes");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 6 &&
            decoded.instructions[1].opcode == PcoOpcode::kFloatAbs &&
            decoded.instructions[1].source.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[1].source.index == 18 &&
            decoded.instructions[1].output_index == 29 &&
            decoded.instructions[1].source_count == 1 &&
            decoded.instructions[1].binary_offset == 15,
        "Terrain FABS preserves its exact source, destination and modifier");
  Check(execute_fragment(FloatBits(-2.0F)) == FloatBits(2.0F) &&
            execute_fragment(UINT32_C(0x80000000)) == 0 &&
            execute_fragment(UINT32_C(0xff800000)) ==
                UINT32_C(0x7f800000) &&
            execute_fragment(UINT32_C(0xffc12345)) ==
                UINT32_C(0x7fc12345),
        "Terrain FABS clears only the sign bit for finite/zero/Inf/NaN");

  auto negate_binary = make_fragment_binary(FloatBits(2.0F));
  negate_binary[16] = 0x02;
  const auto negate = Decode(ShaderStage::kFragment, negate_binary);
  Check(negate.instructions[1].opcode == PcoOpcode::kFloatNegate &&
            ExecuteFragment(negate.summary, negate.instructions)
                    .pixel_outputs[0] == FloatBits(-2.0F),
        "scalar MBYP modifier 0x02 remains the distinct FNEG operation");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{14, 0x80},
           {15, 0x96}, {16, 0x00}, {17, 0x7f}, {18, 0x01},
           {19, 0x01}, {20, 0x80}, {21, 0xfe}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Terrain FABS near-neighbor/bounds mutation");
  }

  auto malformed_instructions = decoded.instructions;
  malformed_instructions[1].source_count = 2;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, malformed_instructions);
      },
      "Terrain FABS rejects malformed decoded unary metadata");

  auto vertex_binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 c0 00 00 52 ff
35 82 00 97 01 52 00 00 5d ff
86 92 40 13 00 00 00 00 00 00 5e ff
86 92 40 13 00 00 00 00 00 00 5f ff
86 92 40 13 00 00 80 3f 00 00 60 ff
55 a0 06 08 00 dd 00 00 00 30
44 a0 80 05 00 00 00 ff
)hex");
  const auto vertex = Decode(ShaderStage::kVertex, vertex_binary);
  Check(vertex.instructions[1].opcode == PcoOpcode::kFloatAbs &&
            vertex.summary.vertex_output_mask == UINT64_C(0x0f) &&
            vertex.summary.ends_task == 1,
        "public FABS decodes in the vertex stage with bounded TEMP export");
  const auto vertex_result =
      ExecuteVertex(vertex.summary, vertex.instructions, {});
  Check(vertex_result.outputs[0] == FloatBits(2.0F) &&
            vertex_result.outputs[1] == 0 &&
            vertex_result.outputs[2] == 0 &&
            vertex_result.outputs[3] == FloatBits(1.0F) &&
            vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "public vertex FABS executes and exports its exact sign-cleared value");
}

void TestFloatMaxOrderedSourceRouting() {
  const auto make_binary = [](std::uint32_t left, std::uint32_t right) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 40 ff
86 92 40 13 00 00 00 00 00 00 41 ff
77 d2 00 d0 3c fa 10 87 87 40 41 10 42 ff
34 8a 00 87 42 00 00 20
34 8a 00 87 42 00 00 21
34 8a 00 87 42 00 00 22
34 8a 80 87 42 00 00 23
)hex");
    for (unsigned byte = 0; byte < 4; ++byte) {
      binary[4 + byte] =
          static_cast<std::uint8_t>(left >> (byte * 8U));
      binary[16 + byte] =
          static_cast<std::uint8_t>(right >> (byte * 8U));
    }
    return binary;
  };
  const auto execute = [&](std::uint32_t left, std::uint32_t right) {
    const auto binary = make_binary(left, right);
    const auto decoded = Decode(ShaderStage::kFragment, binary);
    return ExecuteFragment(decoded.summary, decoded.instructions)
        .pixel_outputs[0];
  };
  constexpr std::uint32_t kLeftNan = UINT32_C(0x7fc34567);
  constexpr std::uint32_t kRightNan = UINT32_C(0x7fc45678);
  Check(execute(FloatBits(2.0F), FloatBits(1.0F)) == FloatBits(2.0F) &&
            execute(kLeftNan, FloatBits(1.0F)) == FloatBits(1.0F) &&
            execute(FloatBits(1.0F), kRightNan) == kRightNan &&
            execute(UINT32_C(0x00000000), UINT32_C(0x80000000)) ==
                UINT32_C(0x80000000) &&
            execute(UINT32_C(0x80000000), UINT32_C(0x00000000)) ==
                UINT32_C(0x00000000),
        "ordered FMAX true selects left and false/NaN/equal-zero selects right");
}

void TestDecodeAndExecuteTerrainFloatAddSaturate() {
  /* Exact Terrain D1 FS FADD.SAT group at offset 0xde4. */
  const auto make_binary = [](std::uint32_t input) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 48 ff
35 82 00 10 c8 80 00 00 48 ff
34 8a 00 87 48 00 00 20
34 8a 00 87 48 00 00 21
34 8a 00 87 48 00 00 22
34 8a 80 87 48 00 00 23
)hex");
    for (unsigned byte = 0; byte < 4; ++byte) {
      binary[4 + byte] =
          static_cast<std::uint8_t>(input >> (byte * 8U));
    }
    return binary;
  };
  const auto execute = [&](std::uint32_t input) {
    const auto binary = make_binary(input);
    const auto decoded = Decode(ShaderStage::kFragment, binary);
    return ExecuteFragment(decoded.summary, decoded.instructions)
        .pixel_outputs[0];
  };

  const auto fragment_binary = make_binary(FloatBits(2.0F));
  Check(fragment_binary.size() == 54,
        "Terrain FADD.SAT fixture preserves exact group sizes");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 6 &&
            decoded.instructions[1].opcode == PcoOpcode::kFloatAdd &&
            decoded.instructions[1].saturate == 1 &&
            decoded.instructions[1].source.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[1].source.index == 8 &&
            decoded.instructions[1].source1.bank ==
                PcoRegisterBank::kSpecial &&
            decoded.instructions[1].source1.index == 0 &&
            decoded.instructions[1].output_index == 8 &&
            decoded.instructions[1].binary_offset == 15,
        "Terrain FADD.SAT preserves exact operands, target and modifier");

  Check(execute(FloatBits(-0.25F)) == UINT32_C(0x00000000) &&
            execute(UINT32_C(0x80000000)) == UINT32_C(0x00000000) &&
            execute(UINT32_C(0x00000000)) == UINT32_C(0x00000000) &&
            execute(FloatBits(0.25F)) == FloatBits(0.25F) &&
            execute(FloatBits(1.0F)) == FloatBits(1.0F) &&
            execute(FloatBits(2.0F)) == FloatBits(1.0F) &&
            execute(UINT32_C(0x7fc12345)) == UINT32_C(0x00000000) &&
            execute(UINT32_C(0xff800000)) == UINT32_C(0x00000000) &&
            execute(UINT32_C(0x7f800000)) == FloatBits(1.0F),
        "Terrain FADD.SAT follows NIR minimumNumber(maximumNumber(x,+0),1)");

  auto unsaturated_binary = fragment_binary;
  unsaturated_binary[15] = 0x00;
  const auto unsaturated =
      Decode(ShaderStage::kFragment, unsaturated_binary);
  Check(unsaturated.instructions[1].opcode == PcoOpcode::kFloatAdd &&
            unsaturated.instructions[1].saturate == 0 &&
            ExecuteFragment(unsaturated.summary, unsaturated.instructions)
                    .pixel_outputs[0] == FloatBits(2.0F),
        "scalar main 0x00 remains the distinct unsaturated FADD encoding");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{14, 0x80},
           {15, 0x11}, {16, 0xff}, {18, 0x01}, {19, 0x01},
           {20, 0x20}, {21, 0xfe}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Terrain FADD.SAT near-neighbor encoding mutation");
  }

  auto malformed_instructions = decoded.instructions;
  malformed_instructions[1].saturate = 2;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, malformed_instructions);
      },
      "Terrain FADD.SAT rejects a reserved saturate flag");
  malformed_instructions = decoded.instructions;
  malformed_instructions[1].opcode = PcoOpcode::kFloatMultiply;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(decoded.summary, malformed_instructions);
      },
      "Terrain saturate flag is rejected on non-FADD operations");

  auto post_add = decoded.instructions;
  post_add[0].immediate = FloatBits(-2.0F);
  post_add[1].source1 = {PcoRegisterBank::kSpecial, 64}; // +1.0
  Check(ExecuteFragment(decoded.summary, post_add).pixel_outputs[0] == 0,
        "Terrain saturation occurs after binary32 FADD, not on its operands");
}

void TestDecodeAndExecuteTerrainAbsoluteBinarySources() {
  /* Exact adjacent Terrain D1 FS groups at offsets 0x1d1e/0x1d28.  The
   * surrounding MOVI/PIXOUT groups expose both modifiers without changing
   * either compiler-emitted scalar group. */
  const auto make_binary = [](std::uint32_t right_multiply_source,
                              std::uint32_t left_add_source) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 44 ff
35 82 00 42 8b e4 04 00 00 44
86 92 40 13 00 00 00 00 00 00 43 ff
35 82 00 04 c3 a4 00 00 43 ff
34 8a 00 87 43 00 00 20
34 8a 00 87 43 00 00 21
34 8a 00 87 43 00 00 22
34 8a 80 87 43 00 00 23
)hex");
    for (unsigned byte = 0; byte < 4; ++byte) {
      binary[4 + byte] = static_cast<std::uint8_t>(
          right_multiply_source >> (byte * 8U));
      binary[26 + byte] =
          static_cast<std::uint8_t>(left_add_source >> (byte * 8U));
    }
    return binary;
  };
  const auto execute = [&](std::uint32_t right_multiply_source,
                           std::uint32_t left_add_source) {
    const auto binary =
        make_binary(right_multiply_source, left_add_source);
    const auto decoded = Decode(ShaderStage::kFragment, binary);
    return ExecuteFragment(decoded.summary, decoded.instructions)
        .pixel_outputs[0];
  };

  const auto fragment_binary =
      make_binary(FloatBits(-2.0F), FloatBits(-3.0F));
  Check(fragment_binary.size() == 76,
        "Terrain binary ABS fixture preserves exact group sizes");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 8 &&
            decoded.summary.instruction_count == 8 &&
            decoded.summary.pixel_output_mask == 0x0f,
        "Terrain binary ABS fixture has a closed fragment envelope");
  const auto &multiply = decoded.instructions[1];
  Check(multiply.binary_offset == 15 &&
            multiply.opcode == PcoOpcode::kFloatMultiply &&
            multiply.source.bank == PcoRegisterBank::kSpecial &&
            multiply.source.index == 75 &&
            multiply.source1.bank == PcoRegisterBank::kTemporary &&
            multiply.source1.index == 4 && multiply.output_index == 4 &&
            multiply.source0_absolute == 0 &&
            multiply.source1_absolute == 1,
        "Terrain exact 35 82 00 42 8b e4 04 00 00 44 is FMUL s1.abs");
  const auto &add = decoded.instructions[3];
  Check(add.binary_offset == 37 && add.opcode == PcoOpcode::kFloatAdd &&
            add.source.bank == PcoRegisterBank::kTemporary &&
            add.source.index == 3 &&
            add.source1.bank == PcoRegisterBank::kTemporary &&
            add.source1.index == 4 && add.output_index == 3 &&
            add.source0_absolute == 1 && add.source1_absolute == 0,
        "Terrain exact 35 82 00 04 c3 a4 00 00 43 is FADD s0.abs");

  Check(execute(FloatBits(-2.0F), FloatBits(-3.0F)) == FloatBits(4.0F),
        "Terrain FMUL/FADD clear the selected source sign before arithmetic");
  Check(execute(UINT32_C(0x80000000), UINT32_C(0x80000000)) ==
            UINT32_C(0x00000000),
        "Terrain binary ABS maps both selected negative-zero operands to +0");
  Check(execute(UINT32_C(0xff800000), UINT32_C(0x80000000)) ==
            UINT32_C(0x7f800000),
        "Terrain FMUL source1 ABS maps -Inf to +Inf before multiplying");
  Check(execute(FloatBits(-2.0F), UINT32_C(0xffc12345)) ==
            UINT32_C(0x7fc00000),
        "Terrain FADD source0 ABS accepts a signed NaN payload before IEEE add");

  auto plain_multiply = fragment_binary;
  plain_multiply[15] = 0x40;
  const auto decoded_plain_multiply =
      Decode(ShaderStage::kFragment, plain_multiply);
  Check(decoded_plain_multiply.instructions[1].source1_absolute == 0 &&
            ExecuteFragment(decoded_plain_multiply.summary,
                            decoded_plain_multiply.instructions)
                    .pixel_outputs[0] == FloatBits(2.0F),
        "main 0x40 remains the distinct unmodified FMUL encoding");
  auto plain_add = fragment_binary;
  plain_add[37] = 0x00;
  const auto decoded_plain_add = Decode(ShaderStage::kFragment, plain_add);
  Check(decoded_plain_add.instructions[3].source0_absolute == 0 &&
            ExecuteFragment(decoded_plain_add.summary,
                            decoded_plain_add.instructions)
                    .pixel_outputs[0] == FloatBits(-2.0F),
        "main 0x00 remains the distinct unmodified FADD encoding");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{14, 0x80},
           {15, 0x43}, {16, 0x8a}, {17, 0x64}, {18, 0x24},
           {19, 0x01}, {20, 0x01}, {21, 0x84},
           {36, 0x80}, {37, 0x05}, {38, 0x03}, {39, 0x24},
           {40, 0x01}, {41, 0x01}, {42, 0x83}, {43, 0xfe}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Terrain binary ABS near-neighbor/reserved mutation");
  }
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, fragment_binary); },
      "Terrain fragment ABS envelope is rejected by the vertex decoder");

  const auto expect_decoded_failure = [&](std::size_t instruction_index,
                                          auto mutate,
                                          const std::string &description) {
    auto malformed = decoded.instructions;
    mutate(malformed[instruction_index]);
    ExpectFailure(
        [&] { (void)ExecuteFragment(decoded.summary, malformed); },
        description);
  };
  expect_decoded_failure(
      1, [](PcoInstruction &instruction) { instruction.source1_absolute = 2; },
      "Terrain FMUL rejects a reserved source1-absolute flag");
  expect_decoded_failure(
      1, [](PcoInstruction &instruction) { instruction.source0_absolute = 1; },
      "Terrain FMUL rejects simultaneous source0/source1 ABS");
  expect_decoded_failure(
      1, [](PcoInstruction &instruction) { instruction.source0_floor = 1; },
      "Terrain FMUL source1 ABS rejects a combined floor modifier");
  expect_decoded_failure(
      1, [](PcoInstruction &instruction) { instruction.saturate = 1; },
      "Terrain FMUL source1 ABS rejects a combined saturate modifier");
  expect_decoded_failure(
      1, [](PcoInstruction &instruction) { instruction.opcode = PcoOpcode::kFloatAdd; },
      "Terrain source1 ABS is rejected on FADD");
  expect_decoded_failure(
      3, [](PcoInstruction &instruction) { instruction.source0_absolute = 2; },
      "Terrain FADD rejects a reserved source0-absolute flag");
  expect_decoded_failure(
      3, [](PcoInstruction &instruction) { instruction.source1_absolute = 1; },
      "Terrain FADD rejects simultaneous source0/source1 ABS");
  expect_decoded_failure(
      3, [](PcoInstruction &instruction) { instruction.source0_floor = 1; },
      "Terrain FADD source0 ABS rejects a combined floor modifier");
  expect_decoded_failure(
      3, [](PcoInstruction &instruction) { instruction.saturate = 1; },
      "Terrain FADD source0 ABS rejects a combined saturate modifier");
  expect_decoded_failure(
      3,
      [](PcoInstruction &instruction) {
        instruction.opcode = PcoOpcode::kFloatMultiply;
      },
      "Terrain source0 ABS is rejected on FMUL");
}

void TestDecodeAndExecuteTerrainOneOver256SpecialConstant() {
  /* Exact D5 fragment group at byte 0x28 of the 1880-byte Terrain shader
   * (FNV-1a64 76fac56a9fbc5918).  MOVI/PIXOUT only provide a closed
   * executable envelope around the byte-identical compiler-emitted FMUL. */
  const auto fragment_binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 c0 00 00 50 ff
35 82 00 42 92 f0 04 00 00 51
34 8a 00 87 51 00 00 20
34 8a 00 87 51 00 00 21
34 8a 00 87 51 00 00 22
34 8a 80 87 51 00 00 23
)hex");
  Check(fragment_binary.size() == 54,
        "Terrain D5 SC82 fixture preserves exact group sizes");

  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 6 &&
            decoded.summary.instruction_count == 6 &&
            decoded.summary.pixel_output_mask == 0x0f,
        "Terrain D5 SC82 fixture has a closed fragment envelope");
  const auto &multiply = decoded.instructions[1];
  Check(multiply.binary_offset == 15 &&
            multiply.opcode == PcoOpcode::kFloatMultiply &&
            multiply.source.bank == PcoRegisterBank::kSpecial &&
            multiply.source.index == 82 &&
            multiply.source1.bank == PcoRegisterBank::kTemporary &&
            multiply.source1.index == 16 && multiply.output_index == 17 &&
            multiply.source0_absolute == 0 &&
            multiply.source1_absolute == 1,
        "Terrain exact 35 82 00 42 92 f0 04 00 00 51 is FMUL SC82, "
        "abs(TEMP16)");
  const auto executed = ExecuteFragment(decoded.summary, decoded.instructions);
  Check(executed.pixel_outputs[0] == UINT32_C(0x3c000000),
        "Terrain SC82 is exact binary32 1/256 before FMUL source1 ABS");

  auto unmodeled_neighbor = fragment_binary;
  unmodeled_neighbor[16] = 0x93;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, unmodeled_neighbor); },
      "Terrain D5 adjacent but unmodeled SC83 remains fail-closed");
}

void TestDecodeAndExecuteIdeasLogicalAnd() {
  /* Actual Ideas fragment group at byte 0x114.  The MOVI/PIXOUT groups only
   * provide a closed executable envelope around the byte-identical Boolean
   * phase pair emitted three times by the 2144-byte lighting FS. */
  const auto fragment_binary = BytesFromHex(R"hex(
86 92 40 13 5a a5 f0 f0 00 00 59 ff
86 92 40 13 f0 0f f0 0f 00 00 5a ff
56 b2 40 41 02 80 40 00 59 5a 5b ff
34 8a 00 87 5b 00 00 20
34 8a 00 87 00 00 00 21
34 8a 00 87 00 00 00 22
34 8a 80 87 00 00 00 23
)hex");
  Check(fragment_binary.size() == 68,
        "Ideas LOGICAL.AND fixture preserves exact group sizes");

  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 7 &&
            decoded.summary.instruction_count == 7 &&
            decoded.summary.pixel_output_mask == 0x0f &&
            decoded.summary.early_hsr_safe == 1,
        "Ideas LOGICAL.AND fixture decodes a complete fragment ABI");
  const auto &logical_and = decoded.instructions[2];
  Check(logical_and.opcode == PcoOpcode::kBitwiseAnd &&
            logical_and.target == PcoWriteTarget::kTemporary &&
            logical_and.source.bank == PcoRegisterBank::kTemporary &&
            logical_and.source.index == 25 &&
            logical_and.source1.bank == PcoRegisterBank::kTemporary &&
            logical_and.source1.index == 26 &&
            logical_and.output_index == 27 &&
            logical_and.source_count == 2,
        "Ideas BBYP0S1/LOGICAL.AND routes TEMP25,TEMP26 to TEMP27");
  const auto pixel =
      ExecuteFragment(decoded.summary, decoded.instructions);
  Check(pixel.written_mask == 0x0f &&
            pixel.pixel_outputs[0] == UINT32_C(0x00f00550) &&
            pixel.pixel_outputs[1] == 0 && pixel.pixel_outputs[2] == 0 &&
            pixel.pixel_outputs[3] == 0,
        "Ideas LOGICAL.AND executes an exact 32-bit Boolean conjunction");

  auto mutation = fragment_binary;
  mutation[27] ^= 0x01;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Ideas LOGICAL.AND phase-1 opcode mutation");
  mutation = fragment_binary;
  mutation[28] ^= 0x01;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Ideas LOGICAL.AND BBYP0S1 phase mutation");
  mutation = fragment_binary;
  mutation[30] ^= 0x01;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Ideas LOGICAL.AND lower selector mutation");
  mutation = fragment_binary;
  mutation[26] |= 0x80;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Ideas LOGICAL.AND unexpected end-group mutation");
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, fragment_binary); },
      "Ideas fragment LOGICAL.AND is rejected from the vertex stage");
}

void TestDecodeAndExecuteIdeasReciprocalZero() {
  /* The FRCP group is copied byte-for-byte from the real 2144-byte Ideas
   * lighting FS (SHA-256
   * 85ab87db7272598af7a50358475d219c1e4ba6af0c41de7103711eb6578250aa),
   * group 31 at source offset 0x13e: `34 82 00 80 50 00 00 50`.
   * MOVI/PIXOUT groups close the reduced executable envelope and make signed
   * zero plus the retained exceptional-class gates directly observable. */
  const auto make_binary = [](std::uint32_t input) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 00 00 00 50 ff
34 82 00 80 50 00 00 50
34 8a 00 87 50 00 00 20
34 8a 00 87 00 00 00 21
34 8a 00 87 00 00 00 22
34 8a 80 87 00 00 00 23
)hex");
    binary[4] = static_cast<std::uint8_t>(input);
    binary[5] = static_cast<std::uint8_t>(input >> 8U);
    binary[6] = static_cast<std::uint8_t>(input >> 16U);
    binary[7] = static_cast<std::uint8_t>(input >> 24U);
    return binary;
  };

  const auto positive = Decode(ShaderStage::kFragment, make_binary(0));
  Check(positive.summary.group_count == 6 &&
            positive.instructions[1].opcode == PcoOpcode::kReciprocal &&
            positive.instructions[1].source.bank ==
                PcoRegisterBank::kTemporary &&
            positive.instructions[1].source.index == 16 &&
            positive.instructions[1].output_index == 16 &&
            positive.instructions[1].source_count == 1,
        "Ideas exact FRCP group decodes TEMP16 in place");
  const auto positive_pixel =
      ExecuteFragment(positive.summary, positive.instructions);
  Check(positive_pixel.pixel_outputs[0] == UINT32_C(0x7f800000),
        "Ideas FRCP maps positive zero to positive infinity");

  const auto negative =
      Decode(ShaderStage::kFragment, make_binary(UINT32_C(0x80000000)));
  const auto negative_pixel =
      ExecuteFragment(negative.summary, negative.instructions);
  Check(negative_pixel.pixel_outputs[0] == UINT32_C(0xff800000),
        "Ideas FRCP preserves signed-zero in the infinity sign");

  const auto negative_two =
      Decode(ShaderStage::kFragment, make_binary(FloatBits(-2.0F)));
  Check(ExecuteFragment(negative_two.summary, negative_two.instructions)
                .pixel_outputs[0] == FloatBits(-0.5F),
        "Ideas FRCP preserves a finite negative sign");

  const auto positive_infinity =
      Decode(ShaderStage::kFragment, make_binary(UINT32_C(0x7f800000)));
  Check(ExecuteFragment(positive_infinity.summary,
                        positive_infinity.instructions)
                .pixel_outputs[0] == UINT32_C(0x00000000),
        "PCO FRCP maps positive infinity to positive zero");
  const auto negative_infinity =
      Decode(ShaderStage::kFragment, make_binary(UINT32_C(0xff800000)));
  Check(ExecuteFragment(negative_infinity.summary,
                        negative_infinity.instructions)
                .pixel_outputs[0] == UINT32_C(0x80000000),
        "PCO FRCP maps negative infinity to negative zero");
  const auto quiet_nan =
      Decode(ShaderStage::kFragment, make_binary(UINT32_C(0xffc12345)));
  Check(ExecuteFragment(quiet_nan.summary, quiet_nan.instructions)
                .pixel_outputs[0] == UINT32_C(0x7fc00000),
        "PCO FRCP canonicalizes and propagates a quiet NaN");

  const std::pair<std::uint32_t, const char *> malformed_cases[] = {
      {UINT32_C(0x00000001), "subnormal input"},
      {UINT32_C(0x7f7fffff), "subnormal result"},
  };
  for (const auto &malformed : malformed_cases) {
    const auto decoded = Decode(ShaderStage::kFragment,
                                make_binary(malformed.first));
    ExpectFailure(
        [&] { (void)ExecuteFragment(decoded.summary, decoded.instructions); },
        "Ideas FRCP retains fail-closed " + std::string(malformed.second) +
            " policy");
  }

  const auto make_frsq_binary = [&](std::uint32_t input) {
    auto binary = make_binary(input);
    binary[15] = 0x81;
    return binary;
  };
  const auto frsq_positive_zero =
      Decode(ShaderStage::kFragment, make_frsq_binary(0));
  Check(frsq_positive_zero.instructions[1].opcode ==
                PcoOpcode::kReciprocalSquareRoot &&
            ExecuteFragment(frsq_positive_zero.summary,
                            frsq_positive_zero.instructions)
                    .pixel_outputs[0] == UINT32_C(0x7f800000),
        "PCO FRSQ maps positive zero to positive infinity");
  const auto frsq_negative_zero = Decode(
      ShaderStage::kFragment, make_frsq_binary(UINT32_C(0x80000000)));
  Check(ExecuteFragment(frsq_negative_zero.summary,
                        frsq_negative_zero.instructions)
                .pixel_outputs[0] == UINT32_C(0xff800000),
        "PCO FRSQ preserves signed zero in the infinity sign");
  const auto frsq_negative = Decode(
      ShaderStage::kFragment, make_frsq_binary(FloatBits(-4.0F)));
  Check(ExecuteFragment(frsq_negative.summary, frsq_negative.instructions)
                .pixel_outputs[0] == UINT32_C(0x7fc00000),
        "PCO FRSQ maps a finite negative radicand to canonical quiet NaN");
  const auto frsq_positive = Decode(
      ShaderStage::kFragment, make_frsq_binary(FloatBits(4.0F)));
  Check(ExecuteFragment(frsq_positive.summary, frsq_positive.instructions)
                .pixel_outputs[0] == FloatBits(0.5F),
        "PCO FRSQ retains the finite positive reciprocal-square-root path");

  const auto frsq_positive_infinity = Decode(
      ShaderStage::kFragment, make_frsq_binary(UINT32_C(0x7f800000)));
  Check(ExecuteFragment(frsq_positive_infinity.summary,
                        frsq_positive_infinity.instructions)
                .pixel_outputs[0] == UINT32_C(0x00000000),
        "PCO FRSQ maps positive infinity to positive zero");
  for (const auto non_numeric : {
           UINT32_C(0xff800000), UINT32_C(0x7fc12345)}) {
    const auto decoded =
        Decode(ShaderStage::kFragment, make_frsq_binary(non_numeric));
    Check(ExecuteFragment(decoded.summary, decoded.instructions)
                  .pixel_outputs[0] == UINT32_C(0x7fc00000),
          "PCO FRSQ canonicalizes negative infinity and NaN");
  }

  for (const auto malformed : {UINT32_C(0x00000001)}) {
    const auto decoded =
        Decode(ShaderStage::kFragment, make_frsq_binary(malformed));
    ExpectFailure(
        [&] { (void)ExecuteFragment(decoded.summary, decoded.instructions); },
        "PCO FRSQ keeps subnormal inputs outside the public gate");
  }
}

void TestDecodeAndExecuteIdeasNegatedFloatSources() {
  /* Byte-identical modifier groups from the Ideas lighting FS: FADD -s0+s1
   * at 0x120 and all three FMAD source-negation combinations at 0x16c,
   * 0x278 and 0x284. */
  const auto fragment_binary = BytesFromHex(R"hex(
86 92 40 13 00 00 c0 3f 00 00 40 ff
86 92 40 13 00 00 80 40 00 00 4d ff
86 92 40 13 00 00 40 40 00 00 50 ff
86 92 40 13 00 00 80 40 00 00 60 ff
86 92 40 13 00 00 00 40 00 00 4e ff
86 92 40 13 00 00 40 40 00 00 4c ff
86 92 40 13 00 00 40 40 00 00 4b ff
35 82 00 08 c0 ad 00 00 5c ff
36 82 00 c2 cd 70 00 60 00 00 62 ff
36 82 00 ca ce 6c 00 50 00 00 5a ff
36 82 00 c8 cd 6b 00 5a 00 00 5a ff
34 8a 00 87 5c 00 00 20
34 8a 00 87 62 00 00 21
34 8a 00 87 5a 00 00 22
34 8a 80 87 00 00 00 23
)hex");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 15 &&
            decoded.instructions[7].opcode ==
                PcoOpcode::kFloatAddNegateSource0 &&
            decoded.instructions[8].opcode ==
                PcoOpcode::kFloatMadNegateSource2 &&
            decoded.instructions[9].opcode ==
                PcoOpcode::kFloatMadNegateSource0Source2 &&
            decoded.instructions[10].opcode ==
                PcoOpcode::kFloatMadNegateSource0,
        "Ideas FADD/FMAD source-negation modifiers decode distinctly");
  const auto pixel =
      ExecuteFragment(decoded.summary, decoded.instructions);
  Check(pixel.pixel_outputs[0] == FloatBits(2.5F) &&
            pixel.pixel_outputs[1] == FloatBits(8.0F) &&
            pixel.pixel_outputs[2] == FloatBits(-21.0F) &&
            pixel.pixel_outputs[3] == 0,
        "Ideas source-negated FADD/FMAD operations preserve exact operand order");

  auto normal_fadd = fragment_binary;
  normal_fadd[87] = 0x00;
  const auto normal_decoded = Decode(ShaderStage::kFragment, normal_fadd);
  const auto normal_pixel = ExecuteFragment(
      normal_decoded.summary, normal_decoded.instructions);
  Check(normal_pixel.pixel_outputs[0] == FloatBits(5.5F),
        "clearing the valid FADD negate modifier changes real arithmetic");

  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{87, 0x09},
           {97, 0xc3}, {109, 0xcb}, {121, 0xc9}}) {
    auto malformed = fragment_binary;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Ideas float source-negation modifier reserved bit");
  }
}

void TestDecodeAndExecuteIdeasNegatedBcsel() {
  /* Actual 18-byte BCSEL group from Ideas FS offset 0x1c0.  Its true value is
   * produced by the two-byte MBYP.neg P0 phase; the false value is unchanged. */
  const auto fragment_binary = BytesFromHex(R"hex(
86 92 40 13 00 00 00 40 00 00 40 ff
86 92 40 13 ff ff ff ff 00 00 59 ff
86 92 40 13 00 00 e0 40 00 00 4d ff
79 d2 00 d1 3c f0 b0 87 97 02 c0 f9 80 10 4d 01 4d ff
34 8a 00 87 4d 00 00 20
34 8a 00 87 00 00 00 21
34 8a 00 87 00 00 00 22
34 8a 80 87 00 00 00 23
)hex");
  const auto decoded = Decode(ShaderStage::kFragment, fragment_binary);
  Check(decoded.summary.group_count == 8 &&
            decoded.instructions[3].opcode ==
                PcoOpcode::kConditionalSelectNegateTrue &&
            decoded.instructions[3].source.index == 25 &&
            decoded.instructions[3].source1.index == 0 &&
            decoded.instructions[3].source2.index == 13 &&
            decoded.instructions[3].output_index == 13,
        "Ideas negated BCSEL preserves condition/true/false/destination ABI");
  const auto true_pixel =
      ExecuteFragment(decoded.summary, decoded.instructions);
  Check(true_pixel.pixel_outputs[0] == FloatBits(-2.0F),
        "Ideas negated BCSEL toggles the true source sign bit");

  auto false_binary = fragment_binary;
  false_binary[16] = 0;
  false_binary[17] = 0;
  false_binary[18] = 0;
  false_binary[19] = 0;
  const auto false_decoded = Decode(ShaderStage::kFragment, false_binary);
  const auto false_pixel = ExecuteFragment(
      false_decoded.summary, false_decoded.instructions);
  Check(false_pixel.pixel_outputs[0] == FloatBits(7.0F),
        "Ideas negated BCSEL leaves the false source unchanged");

  auto mutation = fragment_binary;
  mutation[45] = 0x03;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Ideas BCSEL true-source negate modifier mutation");
  mutation = fragment_binary;
  mutation[44] = 0x96;
  ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutation); },
                "Ideas BCSEL P0 MBYP operation mutation");
}

void TestDecodeAndExecuteConditionalSelectGreaterZero() {
  /* Exact group 53 at fragment byte 0x202 from the Refract composite PCO.
   * Mesa lowers fcsel_gt(condition, true, false) as TST.F32.GZ + MOVC; it
   * shares D0 with FMAX but has the distinct F2 TST phase byte.  The two
   * leading MOV_IMM groups make r14/r17 self-contained for execution. */
  const auto make_binary = [](std::uint32_t condition_bits) {
    auto binary = BytesFromHex(R"hex(
86 92 40 13 00 00 80 3f 00 00 4e ff
86 92 40 13 00 00 e0 40 00 00 51 ff
78 d2 00 d0 3c f2 10 87 87 80 ee 80 10 51 11 50
34 8a 80 87 50 00 00 20
)hex");
    std::memcpy(binary.data() + 4, &condition_bits, sizeof(condition_bits));
    return binary;
  };

  const auto positive = make_binary(FloatBits(1.0F));
  const auto decoded = Decode(ShaderStage::kFragment, positive);
  Check(decoded.summary.group_count == 4 &&
            decoded.instructions[2].opcode ==
                PcoOpcode::kConditionalSelectGreaterZero &&
            decoded.instructions[2].source.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[2].source.index == 14 &&
            decoded.instructions[2].source1.bank ==
                PcoRegisterBank::kSpecial &&
            decoded.instructions[2].source1.index == 0 &&
            decoded.instructions[2].source2.bank ==
                PcoRegisterBank::kTemporary &&
            decoded.instructions[2].source2.index == 17 &&
            decoded.instructions[2].output_index == 16,
        "Refract CSEL.GZ preserves condition/true/false/destination ABI");
  Check(ExecuteFragment(decoded.summary, decoded.instructions)
                .pixel_outputs[0] == FloatBits(0.0F),
        "Refract CSEL.GZ positive condition selects its true source");

  for (const std::uint32_t false_condition : {
           FloatBits(-1.0F), FloatBits(0.0F), UINT32_C(0x80000000)}) {
    const auto false_binary = make_binary(false_condition);
    const auto false_decoded = Decode(ShaderStage::kFragment, false_binary);
    Check(ExecuteFragment(false_decoded.summary, false_decoded.instructions)
                  .pixel_outputs[0] == FloatBits(7.0F),
          "Refract CSEL.GZ negative and signed-zero conditions select false");
  }

  const auto nan_binary = make_binary(UINT32_C(0x7fc00000));
  const auto nan_decoded = Decode(ShaderStage::kFragment, nan_binary);
  ExpectFailure(
      [&] { (void)ExecuteFragment(nan_decoded.summary,
                                  nan_decoded.instructions); },
      "Refract CSEL.GZ non-finite comparison policy remains fail closed");

  constexpr std::size_t group = 24;
  for (const std::pair<std::size_t, std::uint8_t> mutation : {
           std::pair<std::size_t, std::uint8_t>{group + 5, 0xf3},
           {group + 12, 0x00},
           {group + 14, 0x10},
           {group + 9, 0xbf},
           {group + 12, 0x11},
           {group + 13, 0x91},
           {group + 15, 0x90}}) {
    auto malformed = positive;
    malformed[mutation.first] = mutation.second;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, malformed); },
                  "Refract CSEL.GZ phase/selector/ISS/register mutation");
  }
}

void TestDecodeAndExecuteLitShading() {
  /* Actual PCO from the standalone glmark2 shading compiler probe.  This pair
   * deliberately exercises the generic path rather than an embedded profile.
   * The capture uses another 752-byte VS allocation with a different FNV, so
   * this fixture intentionally carries its own verified probe identity. */
  const auto vertex_binary = BytesFromHex(R"hex(
35 82 00 87 90 08 00 00 00 40 35 82 00 87 91 08
00 00 00 41 35 82 00 87 92 08 00 00 00 42 35 82
00 87 94 08 00 00 00 43 35 82 00 87 95 08 00 00
00 44 35 82 00 87 96 08 00 00 00 45 35 82 00 40
c3 c5 08 00 00 43 35 82 00 40 c4 c5 08 00 00 44
35 82 00 40 c5 c5 08 00 00 45 36 82 00 c0 c0 44
08 43 00 00 40 ff 36 82 00 c0 c1 44 08 44 00 00
41 ff 36 82 00 c0 c2 44 08 45 00 00 42 ff 35 82
00 87 98 08 00 00 00 43 35 82 00 87 99 08 00 00
00 44 35 82 00 87 9a 08 00 00 00 45 36 82 00 c0
c3 46 08 40 00 00 40 ff 36 82 00 c0 c4 46 08 41
00 00 41 ff 36 82 00 c0 c5 46 08 42 00 00 42 ff
35 82 00 87 9c 08 00 00 00 43 35 82 00 87 9d 08
00 00 00 44 35 82 00 87 9e 08 00 00 00 45 35 82
00 00 c0 a3 00 00 40 ff 35 82 00 00 c1 a4 00 00
41 ff 35 82 00 00 c2 a5 00 00 42 ff 35 82 00 40
c2 a2 00 00 43 ff 36 82 00 c0 c1 61 00 43 00 00
43 ff 36 82 00 c0 c0 60 00 43 00 00 43 ff 86 92
40 13 ab aa 2a 3f 00 00 44 ff 36 82 00 40 c2 d8
80 01 00 00 42 ff 36 82 00 c0 c1 64 00 42 00 00
41 ff 36 82 00 c0 c0 64 00 41 00 00 40 ff 34 82
00 81 43 00 00 41 35 82 00 40 c1 a0 00 00 40 ff
77 d2 00 d0 3c fa 10 87 87 40 00 10 40 ff 57 92
00 9c 0e 80 40 a0 40 10 00 2c 44 ff 35 82 00 87
80 08 00 00 00 40 35 82 00 87 81 08 00 00 00 41
35 82 00 87 82 08 00 00 00 42 35 82 00 87 83 08
00 00 00 43 35 82 00 87 84 08 00 00 00 45 35 82
00 87 85 08 00 00 00 46 35 82 00 87 86 08 00 00
00 47 35 82 00 87 87 08 00 00 00 48 35 82 00 40
c5 c1 08 00 00 45 35 82 00 40 c6 c1 08 00 00 46
35 82 00 40 c7 c1 08 00 00 47 35 82 00 40 c8 c1
08 00 00 48 36 82 00 c0 c0 40 08 45 00 00 40 ff
36 82 00 c0 c1 40 08 46 00 00 41 ff 36 82 00 c0
c2 40 08 47 00 00 42 ff 36 82 00 c0 c3 40 08 48
00 00 43 ff 35 82 00 87 88 08 00 00 00 45 35 82
00 87 89 08 00 00 00 46 35 82 00 87 8a 08 00 00
00 47 35 82 00 87 8b 08 00 00 00 48 36 82 00 c0
c5 42 08 40 00 00 40 ff 36 82 00 c0 c6 42 08 41
00 00 45 ff 36 82 00 c0 c7 42 08 42 00 00 46 ff
36 82 00 c0 c8 42 08 43 00 00 47 ff 35 82 00 87
8c 08 00 00 00 41 35 82 00 87 8d 08 00 00 00 48
35 82 00 87 8e 08 00 00 00 49 35 82 00 87 8f 08
00 00 00 4a 35 82 00 00 c0 a1 00 00 40 ff 35 82
00 00 c5 a8 00 00 41 ff 35 82 00 00 c6 a9 00 00
42 ff 35 82 00 00 c7 aa 00 00 43 ff 55 a0 06 08
00 c0 00 00 00 30 35 82 00 9c 0e 44 00 00 40 ff
58 a0 80 0e 04 c0 00 00 00 30 f3 ff ff ff ff ff
)hex");
  const auto fragment_binary = BytesFromHex(R"hex(
56 a0 00 b0 01 c4 40 10 c0 40 00 ff 02 80 6a ff
57 92 00 9c 0e 80 40 a0 40 10 00 2c 40 ff 34 8a
00 87 00 00 00 20 34 8a 00 87 00 00 00 21 35 82
00 9c 0e 40 00 00 40 ff 34 8a 00 87 40 00 00 22
38 8a 80 87 80 01 00 00 00 23 f3 ff ff ff ff ff
)hex");

  Check(vertex_binary.size() == 752 &&
            Fnv1a64(vertex_binary) == UINT64_C(0x7731281411489b7f),
        "lit shading probe VS exact bytes and FNV");
  Check(fragment_binary.size() == 80 &&
            Fnv1a64(fragment_binary) == UINT64_C(0xff81c50fbad62cdd),
        "lit shading FS exact bytes and FNV");

  const auto vertex = Decode(ShaderStage::kVertex, vertex_binary);
  Check(vertex.summary.group_count == 70 &&
            vertex.summary.vertex_input_mask == UINT32_C(0x77) &&
            vertex.summary.vertex_output_mask == UINT64_C(0x1f) &&
            vertex.summary.ends_task == 1,
        "lit shading VS generic decode summary");
  PcoVertexExecutionContext vertex_context;
  vertex_context.shared_count = 32;
  for (std::size_t base : {std::size_t{0}, std::size_t{16}}) {
    vertex_context.shared_registers[base + 0] = FloatBits(1.0F);
    vertex_context.shared_registers[base + 5] = FloatBits(1.0F);
    vertex_context.shared_registers[base + 10] = FloatBits(1.0F);
    vertex_context.shared_registers[base + 15] = FloatBits(1.0F);
  }
  const auto vertex_result = ExecuteVertex(
      vertex.summary, vertex.instructions,
      {FloatBits(1.0F), FloatBits(2.0F), FloatBits(3.0F), FloatBits(1.0F),
       FloatBits(0.0F), FloatBits(0.0F), FloatBits(1.0F), FloatBits(0.0F)},
      vertex_context);
  Check(vertex_result.written_mask == UINT64_C(0x1f) &&
            vertex_result.emitted == 1 && vertex_result.ended_task == 1,
        "lit shading VS executes FMUL/FMAD/FRSQ/FMAX and scalar UVSW");
  auto tampered_vertex = vertex;
  tampered_vertex.instructions[0].component_count = 4;
  ExpectFailure(
      [&] {
        (void)ExecuteVertex(tampered_vertex.summary,
                            tampered_vertex.instructions,
                            {FloatBits(1.0F), FloatBits(2.0F), FloatBits(3.0F),
                             FloatBits(1.0F), FloatBits(0.0F), FloatBits(0.0F),
                             FloatBits(1.0F), FloatBits(0.0F)},
                            vertex_context);
      },
      "generic VS rejects noncanonical serialized ALU metadata");
  const auto zero_normal_result = ExecuteVertex(
      vertex.summary, vertex.instructions,
      {FloatBits(1.0F), FloatBits(2.0F), FloatBits(3.0F), FloatBits(1.0F),
       FloatBits(0.0F), FloatBits(0.0F), FloatBits(0.0F), FloatBits(0.0F)},
      vertex_context);
  Check(zero_normal_result.written_mask == UINT64_C(0x1f) &&
            zero_normal_result.emitted == 1 &&
            zero_normal_result.ended_task == 1,
        "generic VS carries the public FRSQ signed-zero result");

  const auto fragment = Decode(ShaderStage::kFragment, fragment_binary);
  Check(fragment.summary.group_count == 8 &&
            fragment.summary.pixel_output_mask == 0x0f &&
            fragment.summary.early_hsr_safe == 1,
        "lit shading FS generic decode summary");
  PcoFragmentExecutionContext fragment_context;
  fragment_context.coefficient_count = 16;
  fragment_context.coefficients[2] = FloatBits(1.0F);
  fragment_context.coefficients[6] = FloatBits(0.5F);
  fragment_context.sample_x = FloatBits(0.5F);
  fragment_context.sample_y = FloatBits(0.5F);
  const auto pixel = ExecuteFragment(fragment.summary, fragment.instructions,
                                     fragment_context);
  Check(pixel.written_mask == 0x0f &&
            pixel.pixel_outputs[0] == FloatBits(0.0F) &&
            pixel.pixel_outputs[1] == FloatBits(0.0F) &&
            pixel.pixel_outputs[2] == FloatBits(0.5F) &&
            pixel.pixel_outputs[3] == FloatBits(1.0F),
        "lit shading FS executes scalar FITRP and binary16 round trip");
  auto tampered_fragment = fragment;
  tampered_fragment.instructions[2].data_request = 1;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(tampered_fragment.summary,
                              tampered_fragment.instructions,
                              fragment_context);
      },
      "generic FS rejects noncanonical serialized PCK metadata");

  auto shared_last = vertex_binary;
  shared_last[4] = 0xab;
  const auto shared_last_decoded = Decode(ShaderStage::kVertex, shared_last);
  Check(shared_last_decoded.instructions[0].source.bank ==
                PcoRegisterBank::kShared &&
            shared_last_decoded.instructions[0].source.index == 43,
        "lit shading VS decoder accepts the last modeled SH43");
  auto shared_extended = vertex_binary;
  shared_extended[4] = 0xac;
  const auto shared_extended_decoded =
      Decode(ShaderStage::kVertex, shared_extended);
  Check(shared_extended_decoded.instructions[0].source.bank ==
                PcoRegisterBank::kShared &&
            shared_extended_decoded.instructions[0].source.index == 44,
        "lit shading VS decoder accepts SH44 for multi-descriptor ABI");
  auto shared_file_last = vertex_binary;
  shared_file_last[4] = 0x9f;
  shared_file_last[5] = 0x09;
  const auto shared_file_last_decoded =
      Decode(ShaderStage::kVertex, shared_file_last);
  Check(shared_file_last_decoded.instructions[0].source.index == 95,
        "generic VS decoder accepts the last modeled terrain SH95");
  auto shared_overflow = vertex_binary;
  shared_overflow[4] = 0xa0;
  shared_overflow[5] = 0x09;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, shared_overflow); },
                "generic VS shared source exceeds terrain transport SH95");
  auto long_fmul_mux = vertex_binary;
  long_fmul_mux[0x130] = 0xa0;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, long_fmul_mux); },
                "lit shading VS long FMUL has noncanonical embedded is0");
  auto fmad_mux = vertex_binary;
  fmad_mux[0x60] = 0x28;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, fmad_mux); },
                "lit shading VS FMAD has noncanonical embedded is0");
  auto pack_mux = vertex_binary;
  pack_mux[0x175] = 0x80;
  ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, pack_mux); },
                "lit shading VS PCK no longer selects s2");
  auto zero_component_fitrp = fragment_binary;
  zero_component_fitrp[4] = 0x00;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, zero_component_fitrp); },
      "lit shading FS FITRP has zero components");
  fragment_context.coefficient_count = 7;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment.summary, fragment.instructions,
                              fragment_context);
      },
      "lit shading FS coefficient span is truncated");
}

void TestSharedRegisterFileBoundary() {
  // These are workload transport gates rather than hardware limits, so the
  // property worth pinning is that the file holds what the bounds promise:
  // every descriptor set plus room for push constants, and execution storage
  // covering the larger of the two stages.
  static_assert(pvrgpu::stub::kPcoMaximumVertexSharedCount >= 96,
                "terrain VS transport requires SH0..95");
  static_assert(pvrgpu::stub::kPcoMaximumFragmentSharedCount >= 164,
                "terrain FS transport requires SH0..163");
  static_assert(pvrgpu::stub::kPcoMaximumSharedCount ==
                    pvrgpu::stub::kPcoMaximumFragmentSharedCount,
                "execution storage covers the larger stage transport");
  static_assert(pvrgpu::stub::kPcoTextureDescriptorDwordCount *
                        pvrgpu::stub::kPcoMaximumTextureDescriptorSets <=
                    pvrgpu::stub::kPcoMaximumFragmentSharedCount,
                "combined descriptors must fit the fragment transport");

  std::vector<PcoInstruction> instructions(2);
  instructions[0].opcode = PcoOpcode::kMoveBypass;
  instructions[0].target = PcoWriteTarget::kTemporary;
  instructions[0].source = {PcoRegisterBank::kShared, 63};
  instructions[0].output_index = 0;
  instructions[0].binary_offset = 0;
  instructions[0].group_index = 0;

  instructions[1].opcode = PcoOpcode::kUvsWriteEmitEndTask;
  instructions[1].target = PcoWriteTarget::kVertexOutput;
  instructions[1].source = {PcoRegisterBank::kTemporary, 0};
  instructions[1].output_index = 0;
  instructions[1].binary_offset = 8;
  instructions[1].group_index = 1;
  instructions[1].end_group = 1;

  PcoProgramSummary summary;
  summary.stage = ShaderStage::kVertex;
  summary.binary_size = 16;
  summary.group_count = 2;
  summary.instruction_count = 2;
  summary.vertex_output_mask = 1;
  summary.ends_task = 1;

  PcoVertexExecutionContext context;
  context.shared_count = 64;
  context.shared_registers[63] = UINT32_C(0x3f123456);
  const auto result = ExecuteVertex(summary, instructions, {}, context);
  Check(result.outputs[0] == context.shared_registers[63] &&
            result.written_mask == 1 && result.emitted == 1 &&
            result.ended_task == 1,
        "64-DWORD refract context accepts its last transported SH63");

  PcoVertexExecutionContext truncated = context;
  truncated.shared_count = 63;
  ExpectFailure(
      [&] { (void)ExecuteVertex(summary, instructions, {}, truncated); },
      "generic vertex execution rejects an absent SH63");

  instructions[0].source.index = 64;
  ExpectFailure(
      [&] { (void)ExecuteVertex(summary, instructions, {}, context); },
      "64-DWORD refract context rejects an absent SH64");

  PcoVertexExecutionContext terrain_vertex_context;
  terrain_vertex_context.shared_count = 96;
  terrain_vertex_context.shared_registers[95] = UINT32_C(0x3f654321);
  instructions[0].source.index = 95;
  const auto terrain_vertex =
      ExecuteVertex(summary, instructions, {}, terrain_vertex_context);
  Check(terrain_vertex.outputs[0] ==
            terrain_vertex_context.shared_registers[95],
        "terrain vertex context accepts its last transported SH95");
  instructions[0].source.index = 96;
  ExpectFailure(
      [&] {
        (void)ExecuteVertex(summary, instructions, {}, terrain_vertex_context);
      },
      "terrain vertex validation rejects SH96 beyond its transport gate");

  std::vector<PcoInstruction> fragment_instructions(5);
  fragment_instructions[0].opcode = PcoOpcode::kMoveBypass;
  fragment_instructions[0].target = PcoWriteTarget::kTemporary;
  fragment_instructions[0].source = {PcoRegisterBank::kShared, 163};
  fragment_instructions[0].output_index = 0;
  for (std::size_t component = 0; component < 4; ++component) {
    PcoInstruction &move = fragment_instructions[1U + component];
    move.opcode = PcoOpcode::kMoveBypass;
    move.target = PcoWriteTarget::kPixelOutput;
    move.source = {PcoRegisterBank::kTemporary, 0};
    move.output_index = static_cast<std::uint16_t>(component);
    move.binary_offset = static_cast<std::uint32_t>((component + 1U) * 8U);
    move.group_index = static_cast<std::uint16_t>(component + 1U);
  }
  fragment_instructions.back().end_group = 1;
  PcoProgramSummary fragment_summary;
  fragment_summary.stage = ShaderStage::kFragment;
  fragment_summary.binary_size = 40;
  fragment_summary.group_count = 5;
  fragment_summary.instruction_count = 5;
  fragment_summary.pixel_output_mask = 0x0f;
  fragment_summary.early_hsr_safe = 1;
  PcoFragmentExecutionContext fragment_context;
  fragment_context.shared_count = 164;
  fragment_context.shared_registers[163] = UINT32_C(0x3f2468ac);
  const auto terrain_fragment = ExecuteFragment(
      fragment_summary, fragment_instructions, fragment_context);
  Check(terrain_fragment.written_mask == 0x0f &&
            terrain_fragment.pixel_outputs[0] ==
                fragment_context.shared_registers[163] &&
            terrain_fragment.pixel_outputs[3] ==
                fragment_context.shared_registers[163],
        "terrain fragment context accepts its last transported SH163");
  PcoFragmentExecutionContext short_fragment_context = fragment_context;
  short_fragment_context.shared_count = 163;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment_summary, fragment_instructions,
                              short_fragment_context);
      },
      "terrain fragment context rejects an absent SH163");
  fragment_instructions[0].source.index = 164;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(fragment_summary, fragment_instructions,
                              fragment_context);
      },
      "terrain fragment validation rejects SH164 beyond its transport gate");
}

void TestExecuteVertexTextureContinuations() {
  /* These are the two exact SMP.2D.FCNORM+drc0/WDF group pairs from the
   * compiler-produced Terrain D3 VS (2440 bytes, FNV de47363e398a2bcc).
   * The shader samples descriptor set 1 first, then set 0.  Splicing the
   * canonical groups after the first two TEMP coordinate moves of the public
   * attribute-fetch VS gives a small independently decodable vertex program
   * while preserving the real descriptor and response encodings. */
  const auto terrain_groups = BytesFromHex(R"hex(
57 a0 00 f4 4c 94 60 80 1c 88 80 a0 00 ff 02 80 6a ff
57 a0 00 f4 4c 80 60 80 08 88 80 a0 00 ff 02 80 6a ff
)hex");
  Check(terrain_groups.size() == 36,
        "Terrain D3 VS contributes two exact SMP/WDF group pairs");

  auto vertex_binary = AttributeFetchVertexPcoBinary();
  const auto base = Decode(ShaderStage::kVertex, vertex_binary);
  const std::size_t insertion_offset = base.instructions[2].binary_offset - 3U;
  vertex_binary.insert(vertex_binary.begin() + insertion_offset,
                       terrain_groups.begin(), terrain_groups.end());
  const auto vertex = Decode(ShaderStage::kVertex, vertex_binary);
  Check(vertex.instructions.size() == 10 &&
            vertex.instructions[2].opcode == PcoOpcode::kTextureSample &&
            vertex.instructions[3].opcode == PcoOpcode::kWaitDataFence &&
            vertex.instructions[4].opcode == PcoOpcode::kTextureSample &&
            vertex.instructions[5].opcode == PcoOpcode::kWaitDataFence &&
            vertex.instructions[2].source1.index == 20 &&
            vertex.instructions[4].source1.index == 0,
        "minimal real Terrain vertex PCO decodes set1 then set0 SMP/WDF");

  PcoVertexExecutionContext context;
  context.shared_count = 40;
  for (std::size_t index = 0; index < context.shared_count; ++index)
    context.shared_registers[index] =
        static_cast<std::uint32_t>(UINT32_C(0x3000) + index);
  const std::vector<std::uint32_t> inputs = {
      FloatBits(0.25F), FloatBits(0.75F)};
  const std::array<std::uint32_t, 4> first_response = {
      FloatBits(0.2F), FloatBits(0.4F), FloatBits(0.6F), FloatBits(0.8F)};
  const std::array<std::uint32_t, 4> second_response = {
      FloatBits(0.1F), FloatBits(0.3F), FloatBits(0.5F), FloatBits(0.7F)};

  auto execution =
      ExecuteVertex(vertex.summary, vertex.instructions, inputs, context);
  std::uint32_t executed_groups = execution.executed_instruction_count;
  Check(execution.suspended == 1 && execution.texture_request_valid == 1 &&
            execution.continuation.valid == 1 &&
            execution.executed_instruction_count == 3 &&
            execution.texture_request.descriptor_set == 1 &&
            execution.texture_request.binding == 0 &&
            execution.texture_request.coordinates[0] == inputs[0] &&
            execution.texture_request.coordinates[1] == inputs[1] &&
            execution.texture_request.texture_state[0] == UINT32_C(0x3014) &&
            execution.texture_request.sampler_state[0] == UINT32_C(0x301c) &&
            execution.continuation.vertex_input_count == inputs.size() &&
            execution.continuation.vertex_inputs[0] == inputs[0] &&
            execution.continuation.shared_count == context.shared_count &&
            execution.continuation.shared_registers[39] == UINT32_C(0x3027) &&
            execution.continuation.temporary_written_mask == UINT64_C(0x3) &&
            execution.continuation.output_written_mask == 0 &&
            execution.continuation.emitted == 0 &&
            execution.continuation.ended_task == 0,
        "vertex SMP suspends with complete lane-local input/shared/TEMP state");

  execution = ResumeVertex(vertex.summary, vertex.instructions,
                           execution.continuation, first_response);
  executed_groups += execution.executed_instruction_count;
  Check(execution.suspended == 1 && execution.texture_request_valid == 1 &&
            execution.executed_instruction_count == 2 &&
            execution.texture_request.descriptor_set == 0 &&
            execution.texture_request.coordinates[0] == first_response[0] &&
            execution.texture_request.coordinates[1] == first_response[1] &&
            execution.texture_request.texture_state[0] == UINT32_C(0x3000) &&
            execution.texture_request.sampler_state[0] == UINT32_C(0x3008) &&
            execution.continuation.temporary_written_mask == UINT64_C(0xf),
        "vertex continuation consumes WDF response then reaches the next SMP");

  const auto second_continuation = execution.continuation;
  execution = ResumeVertex(vertex.summary, vertex.instructions,
                           second_continuation, second_response);
  executed_groups += execution.executed_instruction_count;
  Check(execution.suspended == 0 && execution.texture_request_valid == 0 &&
            execution.written_mask == 0x0f && execution.emitted == 1 &&
            execution.ended_task == 1 &&
            execution.outputs[0] == second_response[0] &&
            execution.outputs[1] == second_response[1] &&
            execution.outputs[2] == 0 &&
            execution.outputs[3] == FloatBits(1.0F) &&
            executed_groups == vertex.instructions.size(),
        "two vertex SMP continuations finish without replay or state loss");

  PcoVertexExecutionContext truncated = context;
  truncated.shared_count = 20;
  ExpectFailure(
      [&] {
        (void)ExecuteVertex(vertex.summary, vertex.instructions, inputs,
                            truncated);
      },
      "Terrain set1 SMP rejects a truncated 40-DWORD descriptor span");

  PcoVertexExecutionContext mismatched_resume;
  mismatched_resume.continuation = second_continuation;
  mismatched_resume.texture_response = second_response;
  mismatched_resume.texture_response_valid = 1;
  const std::vector<std::uint32_t> wrong_inputs = {
      FloatBits(0.5F), FloatBits(0.75F)};
  ExpectFailure(
      [&] {
        (void)ExecuteVertex(vertex.summary, vertex.instructions, wrong_inputs,
                            mismatched_resume);
      },
      "vertex resume rejects inputs that differ from its saved lane");

  auto bad_continuation = second_continuation;
  bad_continuation.temporary_written_mask |= UINT64_C(1) << 12U;
  ExpectFailure(
      [&] {
        (void)ResumeVertex(vertex.summary, vertex.instructions,
                           bad_continuation, second_response);
      },
      "vertex resume rejects an invented pre-SMP TEMP write");

  auto ten_sample_binary = AttributeFetchVertexPcoBinary();
  const auto one_sample_pair = BytesFromHex(R"hex(
57 a0 00 f4 4c 80 60 80 08 88 80 a0 00 ff 02 80 6a ff
)hex");
  std::vector<std::uint8_t> ten_sample_pairs;
  for (std::size_t sample = 0;
       sample < pvrgpu::stub::kPcoMaximumTextureSampleInstructions + 1U;
       ++sample) {
    ten_sample_pairs.insert(ten_sample_pairs.end(), one_sample_pair.begin(),
                            one_sample_pair.end());
  }
  ten_sample_binary.insert(ten_sample_binary.begin() + insertion_offset,
                           ten_sample_pairs.begin(), ten_sample_pairs.end());
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, ten_sample_binary); },
      "the tenth sequential vertex SMP exceeds the bounded continuation gate");
}

void TestExecuteThreeTextureContinuations() {
  std::vector<PcoInstruction> instructions(12);
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    instructions[index].binary_offset =
        static_cast<std::uint32_t>(1U + index * 8U);
    instructions[index].group_index = static_cast<std::uint16_t>(index);
  }

  for (std::size_t coordinate = 0; coordinate < 2; ++coordinate) {
    PcoInstruction &move = instructions[coordinate];
    move.opcode = PcoOpcode::kMoveImmediate;
    move.target = PcoWriteTarget::kTemporary;
    move.source_count = 0;
    move.output_index = static_cast<std::uint16_t>(coordinate);
    move.immediate = FloatBits(coordinate == 0 ? 0.25F : 0.75F);
  }
  for (std::size_t set = 0; set < 3; ++set) {
    PcoInstruction &sample = instructions[2U + set * 2U];
    sample.opcode = PcoOpcode::kTextureSample;
    sample.target = PcoWriteTarget::kTemporary;
    sample.source = {PcoRegisterBank::kTemporary, 0};
    sample.source1 = {
        PcoRegisterBank::kShared,
        static_cast<std::uint16_t>(
            set * pvrgpu::stub::kPcoTextureDescriptorDwordCount)};
    sample.source2 = {
        PcoRegisterBank::kShared,
        static_cast<std::uint16_t>(
            set * pvrgpu::stub::kPcoTextureDescriptorDwordCount + 8U)};
    sample.source_count = 3;
    sample.component_count = 4;
    sample.output_index = static_cast<std::uint16_t>(4U + set * 4U);

    PcoInstruction &wait = instructions[3U + set * 2U];
    wait.opcode = PcoOpcode::kWaitDataFence;
    wait.target = PcoWriteTarget::kNone;
    wait.source_count = 0;
  }
  for (std::size_t component = 0; component < 4; ++component) {
    PcoInstruction &move = instructions[8U + component];
    move.opcode = PcoOpcode::kMoveBypass;
    move.target = PcoWriteTarget::kPixelOutput;
    move.source = {PcoRegisterBank::kTemporary,
                   static_cast<std::uint16_t>(12U + component)};
    move.output_index = static_cast<std::uint16_t>(component);
  }
  instructions.back().end_group = 1;

  PcoProgramSummary summary;
  summary.stage = ShaderStage::kFragment;
  summary.binary_size = 96;
  summary.group_count = static_cast<std::uint32_t>(instructions.size());
  summary.instruction_count = summary.group_count;
  summary.pixel_output_mask = 0x0f;
  summary.early_hsr_safe = 1;

  PcoFragmentExecutionContext base_context;
  base_context.shared_count = 60;
  for (std::size_t index = 0; index < base_context.shared_count; ++index)
    base_context.shared_registers[index] =
        static_cast<std::uint32_t>(UINT32_C(0x1000) + index);

  const std::array<std::array<std::uint32_t, 4>, 3> responses = {{
      {UINT32_C(0x3f000000), UINT32_C(0x3e800000), UINT32_C(0x3e000000),
       UINT32_C(0x3f800000)},
      {UINT32_C(0x3f19999a), UINT32_C(0x3f333333), UINT32_C(0x3f4ccccd),
       UINT32_C(0x3f800000)},
      {UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a),
       UINT32_C(0x3f800000)},
  }};

  auto execution =
      ExecuteFragment(summary, instructions, base_context);
  std::uint32_t executed_groups = execution.executed_instruction_count;
  for (std::size_t set = 0; set < 3; ++set) {
    Check(execution.suspended == 1 &&
              execution.texture_request_valid == 1 &&
              execution.continuation.valid == 1 &&
              execution.texture_request.descriptor_set == set &&
              execution.texture_request.binding == 0 &&
              execution.texture_request.coordinates[0] == FloatBits(0.25F) &&
              execution.texture_request.coordinates[1] == FloatBits(0.75F) &&
              execution.texture_request.texture_state[0] ==
                  UINT32_C(0x1000) + set * 20U &&
              execution.texture_request.sampler_state[0] ==
                  UINT32_C(0x1008) + set * 20U,
          "generic fragment continuation routes one exact descriptor set");

    PcoFragmentExecutionContext resume_context = base_context;
    resume_context.continuation = execution.continuation;
    resume_context.texture_response = responses[set];
    resume_context.texture_response_valid = 1;
    execution = ExecuteFragment(summary, instructions, resume_context);
    executed_groups += execution.executed_instruction_count;
  }
  Check(execution.suspended == 0 && execution.texture_request_valid == 0 &&
            execution.written_mask == 0x0f &&
            execution.pixel_outputs == responses[2] &&
            executed_groups == instructions.size(),
        "three SMP continuations resume without replay or state loss");

  auto misaligned = instructions;
  misaligned[4].source1.index = 21;
  misaligned[4].source2.index = 29;
  ExpectFailure(
      [&] { (void)ExecuteFragment(summary, misaligned, base_context); },
      "generic SMP rejects a non-20-dword descriptor base");

  auto wrong_sampler = instructions;
  wrong_sampler[4].source2.index = 27;
  ExpectFailure(
      [&] { (void)ExecuteFragment(summary, wrong_sampler, base_context); },
      "generic SMP rejects a sampler outside its combined descriptor");

  auto sixth_descriptor = instructions;
  sixth_descriptor[2].source1.index = 100;
  sixth_descriptor[2].source2.index = 108;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(summary, sixth_descriptor, base_context);
      },
      "generic SMP rejects a sixth descriptor beyond the five-set gate");

  PcoFragmentExecutionContext truncated = base_context;
  truncated.shared_count = 40;
  auto first = ExecuteFragment(summary, instructions, truncated);
  truncated.continuation = first.continuation;
  truncated.texture_response = responses[0];
  truncated.texture_response_valid = 1;
  auto second = ExecuteFragment(summary, instructions, truncated);
  truncated.continuation = second.continuation;
  truncated.texture_response = responses[1];
  ExpectFailure(
      [&] { (void)ExecuteFragment(summary, instructions, truncated); },
      "third SMP rejects a truncated descriptor-set shared span");
}

void TestLoweredPowSpecialValues() {
  /* Mesa lowers fpow(x, y) to exp2(log2(x) * y) for this PCO backend.
   * A clamped lighting term commonly makes x exactly zero, so the public
   * FLOG/FEXP chain must carry -infinity and produce zero instead of treating
   * the numeric value as malformed shader metadata. */
  std::vector<PcoInstruction> instructions(11);
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    instructions[index].binary_offset = static_cast<std::uint32_t>(index * 8);
    instructions[index].group_index = static_cast<std::uint16_t>(index);
  }

  instructions[0].opcode = PcoOpcode::kMoveImmediate;
  instructions[0].target = PcoWriteTarget::kTemporary;
  instructions[0].source_count = 0;
  instructions[0].output_index = 0;
  instructions[0].immediate = FloatBits(0.0F);

  instructions[1].opcode = PcoOpcode::kFloatLog2;
  instructions[1].target = PcoWriteTarget::kTemporary;
  instructions[1].source = {PcoRegisterBank::kTemporary, 0};
  instructions[1].output_index = 1;

  instructions[2].opcode = PcoOpcode::kMoveImmediate;
  instructions[2].target = PcoWriteTarget::kTemporary;
  instructions[2].source_count = 0;
  instructions[2].output_index = 2;
  instructions[2].immediate = FloatBits(100.0F);

  instructions[3].opcode = PcoOpcode::kFloatMultiply;
  instructions[3].target = PcoWriteTarget::kTemporary;
  instructions[3].source = {PcoRegisterBank::kTemporary, 1};
  instructions[3].source1 = {PcoRegisterBank::kTemporary, 2};
  instructions[3].source_count = 2;
  instructions[3].output_index = 3;

  instructions[4].opcode = PcoOpcode::kFloatExp2;
  instructions[4].target = PcoWriteTarget::kTemporary;
  instructions[4].source = {PcoRegisterBank::kTemporary, 3};
  instructions[4].output_index = 4;

  instructions[5].opcode = PcoOpcode::kMoveImmediate;
  instructions[5].target = PcoWriteTarget::kTemporary;
  instructions[5].source_count = 0;
  instructions[5].output_index = 5;
  instructions[5].immediate = FloatBits(0.1F);

  instructions[6].opcode = PcoOpcode::kFloatAdd;
  instructions[6].target = PcoWriteTarget::kTemporary;
  instructions[6].source = {PcoRegisterBank::kTemporary, 4};
  instructions[6].source1 = {PcoRegisterBank::kTemporary, 5};
  instructions[6].source_count = 2;
  instructions[6].output_index = 6;

  instructions[7].opcode = PcoOpcode::kMoveBypass;
  instructions[7].target = PcoWriteTarget::kPixelOutput;
  instructions[7].source = {PcoRegisterBank::kTemporary, 4};
  instructions[7].output_index = 0;
  instructions[8].opcode = PcoOpcode::kMoveBypass;
  instructions[8].target = PcoWriteTarget::kPixelOutput;
  instructions[8].source = {PcoRegisterBank::kTemporary, 6};
  instructions[8].output_index = 1;
  for (std::size_t output = 2; output < 4; ++output) {
    PcoInstruction &move = instructions[7 + output];
    move.opcode = PcoOpcode::kMoveBypass;
    move.target = PcoWriteTarget::kPixelOutput;
    move.source = {PcoRegisterBank::kSpecial,
                   static_cast<std::uint16_t>(output == 3 ? 64 : 0)};
    move.output_index = static_cast<std::uint16_t>(output);
  }
  instructions.back().end_group = 1;

  PcoProgramSummary summary;
  summary.stage = ShaderStage::kFragment;
  summary.binary_size = 88;
  summary.group_count = static_cast<std::uint32_t>(instructions.size());
  summary.instruction_count = summary.group_count;
  summary.pixel_output_mask = 0x0f;
  summary.early_hsr_safe = 1;

  const auto zero_pow = ExecuteFragment(summary, instructions);
  Check(zero_pow.pixel_outputs[0] == FloatBits(0.0F) &&
            zero_pow.pixel_outputs[1] == FloatBits(0.1F) &&
            zero_pow.pixel_outputs[3] == FloatBits(1.0F),
        "lowered fpow maps log2(0) through -infinity to zero");

  instructions[0].immediate = FloatBits(-1.0F);
  const auto negative_pow = ExecuteFragment(summary, instructions);
  Check(negative_pow.pixel_outputs[0] == UINT32_C(0x7fc00000) &&
            negative_pow.pixel_outputs[1] == UINT32_C(0x7fc00000),
        "lowered fpow preserves the log2 negative-domain quiet NaN");

  instructions[0].immediate = FloatBits(0.36F);
  const auto tiny_pow = ExecuteFragment(summary, instructions);
  Check((tiny_pow.pixel_outputs[0] & UINT32_C(0x7f800000)) == 0 &&
            (tiny_pow.pixel_outputs[0] & UINT32_C(0x007fffff)) != 0 &&
            tiny_pow.pixel_outputs[1] == FloatBits(0.1F),
        "lowered fpow carries a subnormal FEXP result through FMUL/FADD");

  instructions[1].source_count = 2;
  ExpectFailure([&] { (void)ExecuteFragment(summary, instructions); },
                "lowered fpow still rejects malformed LOG2 metadata");
}

} // namespace

int main() {
  try {
    TestEmbeddedBinaries();
    TestDecodeAndExecuteVertex();
    TestVertexOutput64BitBoundary();
    TestDecodeAndExecuteAttributeFetch();
    TestDecodeAndExecuteTwoAttributeFetch();
    TestDecodeAndExecuteScalarSource0Floor();
    TestDecodeAndExecuteSpecialConstant153();
    TestDecodeAndExecuteFourAttributeFetch();
    TestDecodeAndExecuteEightAttributeFetch();
    TestIntegerFloatAddRoundingAndClasses();
    TestDecodeAndExecuteFragment();
    TestDecodeAndExecuteHalfAlphaFragments();
    TestDecodeAndExecuteTriangleSetupOrange();
    TestDecodeAndExecuteTriangleSetupHalfCulledCyan();
    TestDecodeFailsClosed();
    TestAttributeFetchFailsClosed();
    TestTwoAttributeFetchFailsClosed();
    TestFourAttributeFetchFailsClosed();
    TestEightAttributeFetchFailsClosed();
    TestDecodeAndExecuteVaryingsOne();
    TestVaryingsOneFailsClosed();
    TestDecodeAndExecuteVaryingsTwo();
    TestVaryingsTwoFailsClosed();
    TestDecodeAndExecuteVaryingsFour();
    TestVaryingsFourFailsClosed();
    TestDecodeAndExecuteVaryingsEight();
    TestVaryingsEightFailsClosed();
    TestDecodeAndExecuteFillTexNearest();
    TestFillTexNearestFailsClosed();
    TestDecodeAndExecuteGlmarkTexture();
    TestDecodeAndExecuteGlmarkTextureMediump();
    TestDecodeAndExecuteConditionals();
    TestDecodeAndExecuteIdeasLightingSelect();
    TestDecodeAndExecuteShadowFloatLess();
    TestDecodeAndExecuteTerrainFloatGreaterEqual();
    TestDecodeAndExecuteTerrainFloatGreaterEqualOneZero();
    TestDecodeAndExecuteTerrainLogicalXnor();
    TestDecodeAndExecuteTerrainFloatMin();
    TestDecodeAndExecuteTerrainFloatAbs();
    TestFloatMaxOrderedSourceRouting();
    TestDecodeAndExecuteTerrainFloatAddSaturate();
    TestDecodeAndExecuteTerrainAbsoluteBinarySources();
    TestDecodeAndExecuteTerrainOneOver256SpecialConstant();
    TestDecodeAndExecuteIdeasLogicalAnd();
    TestDecodeAndExecuteIdeasReciprocalZero();
    TestDecodeAndExecuteIdeasNegatedFloatSources();
    TestDecodeAndExecuteIdeasNegatedBcsel();
    TestDecodeAndExecuteConditionalSelectGreaterZero();
    TestDecodeAndExecuteLitShading();
    TestSharedRegisterFileBoundary();
    TestExecuteVertexTextureContinuations();
    TestExecuteThreeTextureContinuations();
    TestLoweredPowSpecialValues();
    TestExecuteFailsClosed();
    std::cout << "pco_iss_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pco_iss_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
