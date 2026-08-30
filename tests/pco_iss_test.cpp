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
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pvrgpu::stub::CountPcoInstructions;
using pvrgpu::stub::Decode;
using pvrgpu::stub::ExecuteFragment;
using pvrgpu::stub::ExecuteVertex;
using pvrgpu::stub::ResumeFragment;
using pvrgpu::stub::AttributeFetchEightAttributeVertexPcoBinary;
using pvrgpu::stub::AttributeFetchFourAttributeVertexPcoBinary;
using pvrgpu::stub::AttributeFetchGrayFragmentPcoBinary;
using pvrgpu::stub::AttributeFetchTwoAttributeVertexPcoBinary;
using pvrgpu::stub::AttributeFetchVertexPcoBinary;
using pvrgpu::stub::FillSolidFragmentPcoBinary;
using pvrgpu::stub::FillSolidGreenHalfAlphaFragmentPcoBinary;
using pvrgpu::stub::FillSolidRedHalfAlphaFragmentPcoBinary;
using pvrgpu::stub::FillSolidVertexPcoBinary;
using pvrgpu::stub::FillTexNearestFragmentPcoBinary;
using pvrgpu::stub::FillTexNearestVertexPcoBinary;
using pvrgpu::stub::PcoOpcode;
using pvrgpu::stub::PcoFragmentExecutionContext;
using pvrgpu::stub::PcoVertexExecutionContext;
using pvrgpu::stub::PcoIterationMode;
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

  for (std::uint32_t invalid_input : {
           UINT32_C(0x7fc00000), // quiet NaN
           UINT32_C(0x7f800000), // positive infinity
           UINT32_C(0xff800000), // negative infinity
           UINT32_C(0x00000001), // positive subnormal
           UINT32_C(0x807fffff), // negative subnormal
       }) {
    ExpectFailure(
        [&] {
          (void)execute(invalid_input, UINT32_C(0x00000000),
                        UINT32_C(0x3f800000), UINT32_C(0x00000000));
        },
        "FADD unsupported input class fails closed");
  }

  ExpectFailure(
      [&] {
        (void)execute(UINT32_C(0x7f7fffff), UINT32_C(0x00000000),
                      UINT32_C(0x7f7fffff), UINT32_C(0x00000000));
      },
      "FADD finite overflow fails closed");
  ExpectFailure(
      [&] {
        (void)execute(UINT32_C(0x00800001), UINT32_C(0x00000000),
                      UINT32_C(0x80800000), UINT32_C(0x00000000));
      },
      "FADD finite operands with a subnormal result fail closed");
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
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_uvsw_repeat); },
      "attribute-fetch UVSW temporary export is not rpt4");

  auto bad_temp_source = AttributeFetchVertexPcoBinary();
  bad_temp_source[43] = 0xc1;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_temp_source); },
      "attribute-fetch non-canonical temp1 UVSW source");

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

void TestTwoAttributeFetchFailsClosed() {
  ExpectFailure(
      [] {
        (void)Decode(ShaderStage::kFragment,
                     AttributeFetchTwoAttributeVertexPcoBinary());
      },
      "two-attribute VS decoded as fragment stage");

  for (std::uint8_t main_mutation : {std::uint8_t{0x01},
                                     std::uint8_t{0x02},
                                     std::uint8_t{0x04},
                                     std::uint8_t{0x08},
                                     std::uint8_t{0x10},
                                     std::uint8_t{0x20},
                                     std::uint8_t{0x40},
                                     std::uint8_t{0x80}}) {
    auto binary = AttributeFetchTwoAttributeVertexPcoBinary();
    binary[3] = main_mutation;
    ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, binary); },
                  "modified/later FADD main encoding");
  }

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
  expect_byte_failure(6, 0x08, "FADD source0 bank changed");
  expect_byte_failure(6, 0x10, "FADD source1 bank changed");
  expect_byte_failure(4, 0xbf, "FADD source0 exceeds vi31");
  expect_byte_failure(7, 0x01, "FADD upper-source encoding changed");
  expect_byte_failure(8, 0x01, "FADD ISS selection changed");
  expect_byte_failure(9, 0x44, "FADD destination exceeds temp3");
  expect_byte_failure(32, 0x82, "sc65 changed to unsupported sc66");

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
  expect_byte_failure(23, 0x01, "TEMP+VI FADD modifier changed");
  expect_byte_failure(24, 0xc2,
                      "TEMP+VI FADD reads temp2 before it is written");
  expect_byte_failure(25, 0xc5, "TEMP+VI FADD vertex source changed");
  expect_byte_failure(26, 0x18, "TEMP+VI FADD source0 bank changed");
  expect_byte_failure(29, 0x42, "TEMP RMW destination dependency changed");
  expect_byte_failure(72, 0x81, "four-attribute sc66 changed to sc65");
  expect_byte_failure(72, 0x83, "four-attribute sc66 changed to unknown sc67");
  expect_byte_failure(77, 0x44, "sc66 MBYP destination changed from temp5");
  expect_byte_failure(80, 0x04, "four-attribute UVSW repeat is not four");
  expect_byte_failure(83, 0xc0,
                      "four-attribute UVSW export starts at temp0");
  expect_byte_failure(83, 0xc3,
                      "non-canonical overlong temp3 UVSW source");

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
  decoded.instructions[2].source.bank = PcoRegisterBank::kVertexInput;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "serialized four-attribute RMW bank changed");

  decoded = Decode(ShaderStage::kVertex,
                   AttributeFetchFourAttributeVertexPcoBinary());
  decoded.instructions[8].source.index = 0;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "serialized four-attribute export range changed");

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
  expect_byte_failure(49, 0x42,
                      "eight-attribute intermediate WAW destination changed");
  expect_byte_failure(65, 0xc7,
                      "eight-attribute vi8 source changed to vi7");
  expect_byte_failure(103, 0x01,
                      "eight-attribute later FADD modifier changed");
  expect_byte_failure(135, 0xce,
                      "eight-attribute final vi15 source changed to vi14");
  expect_byte_failure(152, 0x82,
                      "eight-attribute sc67 changed to sc66");
  expect_byte_failure(152, 0x84,
                      "eight-attribute sc67 changed to unknown sc68");
  expect_byte_failure(160, 0x04,
                      "eight-attribute UVSW repeat is not four");
  expect_byte_failure(163, 0xc0,
                      "eight-attribute UVSW export starts at temp0");

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
  decoded.instructions[15].source.index = 66;
  ExpectFailure(
      [&] { (void)ExecuteVertex(decoded.summary, decoded.instructions, inputs); },
      "serialized eight-attribute sc67 changed to sc66");

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

  auto bad_fmul_opcode = VaryingsTwoVertexPcoBinary();
  bad_fmul_opcode[51] = 0x41;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_fmul_opcode); },
      "varyings_shader_2 FMUL opcode changed");

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
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_v2_address); },
      "varyings_shader_2 VTXOUT8 base changed");

  auto bad_first_coeff = VaryingsTwoFragmentPcoBinary();
  bad_first_coeff[5] = 0xc4;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_first_coeff); },
      "varyings_shader_2 first FITRP coefficient base changed");

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
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_export_source); },
      "varyings_shader_2 first PIXOUT source changed");

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

  auto semantic = fragment;
  semantic.instructions[4].source.index = 5;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsTwoContext());
      },
      "varyings_shader_2 serialized FADD source changed");

  const auto varying_one =
      Decode(ShaderStage::kFragment, VaryingsOneFragmentPcoBinary());
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(varying_one.summary, varying_one.instructions,
                              MakeVaryingsTwoContext());
      },
      "varyings_shader_1 cannot execute with case-2 coefficient count");
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
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_quarter_constant); },
      "varyings_shader_4 FMUL sc76 changed to sc75");

  auto bad_v4_address = VaryingsFourVertexPcoBinary();
  bad_v4_address[120] = 0x0c;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_v4_address); },
      "varyings_shader_4 fourth varying VTXOUT base changed");

  auto bad_uvsw_padding = VaryingsFourVertexPcoBinary();
  bad_uvsw_padding[126] = 0xf2;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_uvsw_padding); },
      "varyings_shader_4 final UVSW alignment padding changed");

  auto bad_third_coefficient = VaryingsFourFragmentPcoBinary();
  bad_third_coefficient[77] = 0xd4;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_third_coefficient); },
      "varyings_shader_4 third FITRP coefficient base changed");

  auto bad_third_destination = VaryingsFourFragmentPcoBinary();
  bad_third_destination[81] = 0x40;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_third_destination); },
      "varyings_shader_4 third FITRP destination changed");

  auto bad_fourth_coefficient = VaryingsFourFragmentPcoBinary();
  bad_fourth_coefficient[133] = 0xe4;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fourth_coefficient); },
      "varyings_shader_4 fourth FITRP coefficient base changed");

  auto bad_second_layer_fadd = VaryingsFourFragmentPcoBinary();
  bad_second_layer_fadd[92] = 0xc9;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_second_layer_fadd); },
      "varyings_shader_4 second FADD layer source changed");

  auto bad_export_source = VaryingsFourFragmentPcoBinary();
  bad_export_source[188] = 0x44;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_export_source); },
      "varyings_shader_4 first PIXOUT source changed");

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

  auto semantic = fragment;
  semantic.instructions[8].source.index = 20;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsFourContext());
      },
      "varyings_shader_4 serialized third FITRP source changed");

  semantic = fragment;
  semantic.instructions[16].source.index = 6;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsFourContext());
      },
      "varyings_shader_4 serialized final FADD layer changed");

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
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_eighth_constant); },
      "varyings_shader_8 FMUL sc77 changed to sc76");

  auto bad_eighth_address = VaryingsEightVertexPcoBinary();
  bad_eighth_address[160] = 0x1c;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_eighth_address); },
      "varyings_shader_8 eighth varying VTXOUT base changed");

  auto bad_uvsw_padding = VaryingsEightVertexPcoBinary();
  bad_uvsw_padding[166] = 0xf2;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_uvsw_padding); },
      "varyings_shader_8 final UVSW alignment padding changed");

  auto bad_fifth_coefficient_high = VaryingsEightFragmentPcoBinary();
  bad_fifth_coefficient_high[191] = 0x10;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fifth_coefficient_high); },
      "varyings_shader_8 cf68 high source bit changed");

  auto bad_fifth_coefficient_low = VaryingsEightFragmentPcoBinary();
  bad_fifth_coefficient_low[189] = 0xd4;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_fifth_coefficient_low); },
      "varyings_shader_8 cf68 low source bits changed");

  auto bad_eighth_coefficient = VaryingsEightFragmentPcoBinary();
  bad_eighth_coefficient[359] = 0x10;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_eighth_coefficient); },
      "varyings_shader_8 final FITRP coefficient base changed");

  auto bad_final_layer_fadd = VaryingsEightFragmentPcoBinary();
  bad_final_layer_fadd[372] = 0xc6;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_final_layer_fadd); },
      "varyings_shader_8 final FADD layer source changed");

  auto bad_export_source = VaryingsEightFragmentPcoBinary();
  bad_export_source[412] = 0x44;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kFragment, bad_export_source); },
      "varyings_shader_8 first PIXOUT source changed");

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

  auto semantic = fragment;
  semantic.instructions[38].source.index = 100;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsEightContext());
      },
      "varyings_shader_8 serialized final FITRP source changed");

  semantic = fragment;
  semantic.instructions[40].source.index = 6;
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeVaryingsEightContext());
      },
      "varyings_shader_8 serialized final FADD layer changed");

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
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_vertex_repeat); },
      "varyings_shader_1 second UVSW repeat is not four");

  auto bad_vertex_address = VaryingsOneVertexPcoBinary();
  bad_vertex_address[52] = 0x03;
  ExpectFailure(
      [&] { (void)Decode(ShaderStage::kVertex, bad_vertex_address); },
      "varyings_shader_1 VTXOUT4 base address changed");

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
                "FITRP component count changed");

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

  for (std::size_t offset = 0;
       offset < FillTexNearestVertexPcoBinary().size(); ++offset) {
    auto mutated = FillTexNearestVertexPcoBinary();
    mutated[offset] ^= 1U;
    ExpectFailure([&] { (void)Decode(ShaderStage::kVertex, mutated); },
                  "fill_tex_nearest VS byte mutation " +
                      std::to_string(offset));
  }
  for (std::size_t offset = 0;
       offset < FillTexNearestFragmentPcoBinary().size(); ++offset) {
    auto mutated = FillTexNearestFragmentPcoBinary();
    mutated[offset] ^= 1U;
    ExpectFailure([&] { (void)Decode(ShaderStage::kFragment, mutated); },
                  "fill_tex_nearest FS byte mutation " +
                      std::to_string(offset));
  }

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
  ExpectFailure(
      [&] {
        (void)ExecuteFragment(semantic.summary, semantic.instructions,
                              MakeFillTexNearestContext());
      },
      "serialized SMP coordinate source mutation");
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

} // namespace

int main() {
  try {
    TestEmbeddedBinaries();
    TestDecodeAndExecuteVertex();
    TestVertexOutput64BitBoundary();
    TestDecodeAndExecuteAttributeFetch();
    TestDecodeAndExecuteTwoAttributeFetch();
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
    TestExecuteFailsClosed();
    std::cout << "pco_iss_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pco_iss_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
