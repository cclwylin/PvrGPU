/*
 * SPDX-License-Identifier: MIT
 * Encoding layout derived from Mesa code copyrighted by
 * Imagination Technologies Ltd.
 *
 * PowerVR PCO instruction-set simulator (ISS) decoder and executor.
 *
 * PCO is Mesa's public PowerVR shader-backend name, ISS means Instruction Set
 * Simulator, and USC means Unified Shading Cluster.  This implementation reads
 * the exact public PCO instruction-group encoding used by Fill.Solid, all four
 * attribute-fetch cases, varyings_shader_1, and varyings_shader_2: group
 * headers, one/two-source lower fields, upper-source and internal-source-
 * selection (ISS) fields, destinations, FADD, FMUL, MBYP, TEMP, UVSW,
 * FITRP.PIXEL and WDF.
 * Unsupported or non-canonical bytes are errors; no project-local
 * "PowerVR-like" opcode format or shader-name shortcut exists.
 */
#include "pco_iss.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pvrgpu::stub {
namespace {

/*
 * Mesa 26.2.1 commit da14d65e4499e66468094be52bff9ea0915a695e.
 * VS bytes are the public uscgen VS_PASSTHROUGH_COMMON binary after its
 * eight-byte pco_precomp_data header.  FS bytes are emitted by
 * tools/pco-fixtures/generate_fill_solid_fs.c from the same public compiler.
 */
const std::vector<std::uint8_t> kFillSolidVertexBinary = {
    0x58, 0xa0, 0x04, 0x08, 0x00, 0x80, 0x04, 0x00, 0x00, 0x30, 0xf3,
    0xff, 0xff, 0xff, 0xff, 0xff, 0x58, 0xa0, 0x80, 0x0e, 0x03, 0x80,
    0x01, 0x00, 0x00, 0x30, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

const std::vector<std::uint8_t> kFillSolidFragmentBinary = {
    0x35, 0x8a, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x20, 0x34, 0x8a,
    0x00, 0x87, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x00, 0x00,
    0x00, 0x22, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
    0x80, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

/*
 * Opaque black fragment fixture.  This uses the same public MBYP-to-PIXOUT
 * profile as FillSolidFragmentPcoBinary but sources sc0 for RGB and sc64 for
 * alpha, matching the resolved color of the current RDC texture-filtering
 * indexed-quad captures while still exercising USC -> PBE -> DRAM.
 */
const std::vector<std::uint8_t> kFillSolidBlackFragmentBinary = {
    0x34, 0x8a, 0x00, 0x87, 0x00, 0x00, 0x00, 0x20,
    0x34, 0x8a, 0x00, 0x87, 0x00, 0x00, 0x00, 0x21,
    0x37, 0x8a, 0x00, 0x87, 0x00, 0x00, 0x00, 0x22, 0xf3, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x38, 0x8a, 0x80, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x23,
    0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

/*
 * Emitted by tools/pco-fixtures/generate_fill_solid_fs.c with the
 * red-half-alpha fixture.  SHA-256:
 * 1c7d3a1b653f8f05d6fe7f92450a0da6b46e5a9d4e075a8b9c6a143fab7dd26b.
 */
const std::vector<std::uint8_t> kFillSolidRedHalfAlphaFragmentBinary = {
    0x35, 0x8a, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x20, 0x34, 0x8a,
    0x00, 0x87, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x00, 0x00,
    0x00, 0x22, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
    0x8b, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

/*
 * Emitted by tools/pco-fixtures/generate_fill_solid_fs.c with the
 * green-half-alpha fixture.  SHA-256:
 * 37aa277bc79e6282f5003e0baca694de9ab31a89edfb264d55c5eb5186f34e8a.
 */
const std::vector<std::uint8_t> kFillSolidGreenHalfAlphaFragmentBinary = {
    0x34, 0x8a, 0x00, 0x87, 0x00, 0x00, 0x00, 0x20, 0x35, 0x8a, 0x00, 0x87,
    0x80, 0x01, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x00, 0x00,
    0x00, 0x22, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
    0x8b, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

/*
 * Emitted by tools/pco-fixtures/generate_fill_solid_fs.c with the
 * triangle-setup-orange fixture.  SHA-256:
 * e14182dc7a71bb669891fb58aaadae02f8eb457dd137735a5c60fc8b3b1e7727.
 */
const std::vector<std::uint8_t> kTriangleSetupOrangeFragmentBinary = {
    0x35, 0x8a, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x20, 0x35, 0x8a,
    0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x21, 0x36, 0x8a, 0x00, 0x87,
    0x00, 0x00, 0x00, 0x22, 0xf2, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
    0x80, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

/*
 * Emitted by tools/pco-fixtures/generate_fill_solid_fs.c with the
 * triangle-setup-half-culled-cyan fixture.  SHA-256:
 * bc2d2cd9ae2830653930b10ac2849c7e8a30f261cd45346bd482915dbc42ef0d.
 */
const std::vector<std::uint8_t> kTriangleSetupHalfCulledCyanFragmentBinary = {
    0x34, 0x8a, 0x00, 0x87, 0x00, 0x00, 0x00, 0x20, 0x35, 0x8a, 0x00, 0x87,
    0x8b, 0x01, 0x00, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x8b, 0x01,
    0x00, 0x00, 0x00, 0x22, 0xf2, 0xff, 0xff, 0xff, 0x38, 0x8a, 0x80, 0x87,
    0x80, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

/*
 * Emitted for GLBench attribute_fetch_shader case 1 by
 * tools/pco-fixtures/generate_attribute_fetch_shader.c.  The vertex shader
 * lowers a vec2 load_input to vi0/vi1, fills z/w from sc0/sc64, moves those
 * values through temp0..temp3, and exports VTXOUT0..3. SHA-256:
 * 01fb08add3c710fb9062ed0033fecc15e5cfbce56a38a49ed17db4e43f2bf026.
 */
const std::vector<std::uint8_t> kAttributeFetchVertexBinary = {
    0x35, 0x82, 0x00, 0x87, 0x80, 0x04, 0x00, 0x00, 0x00, 0x40,
    0x35, 0x82, 0x00, 0x87, 0x81, 0x04, 0x00, 0x00, 0x00, 0x41,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x42,
    0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x43,
    0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
    0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
};

/*
 * Emitted for GLBench attribute_fetch_shader case 2 by
 * tools/pco-fixtures/generate_attribute_fetch_shader.c.  Two vec2 attributes
 * occupy vi0/vi1 and vi2/vi3.  Public PCO FADD groups form x/y in temp0/1,
 * while sc0/sc65 form z/w in temp2/3. SHA-256:
 * a275bcd7b146f7243e995528c197a04ee24e17f11d313313c7a5bea78030b88f.
 */
const std::vector<std::uint8_t> kAttributeFetchTwoAttributeVertexBinary = {
    0x35, 0x82, 0x00, 0x00, 0x80, 0xc2, 0x18, 0x00, 0x00, 0x40,
    0x35, 0x82, 0x00, 0x00, 0x81, 0xc3, 0x18, 0x00, 0x00, 0x41,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x42,
    0x35, 0x82, 0x00, 0x87, 0x81, 0x01, 0x00, 0x00, 0x00, 0x43,
    0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
    0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
};

/*
 * Emitted for GLBench attribute_fetch_shader case 4 by
 * tools/pco-fixtures/generate_attribute_fetch_shader.c.  Six public PCO FADD
 * groups accumulate vi0..vi7 through temp0..temp3, and sc0/sc66 fill temp4/5
 * before temp2..temp5 are exported. SHA-256:
 * 81b4bf2b412eb2ba35adcd1076d965918336ffb0ffb860e66547695ef4a6ae28.
 */
const std::vector<std::uint8_t> kAttributeFetchFourAttributeVertexBinary = {
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

/*
 * Emitted for GLBench attribute_fetch_shader case 8 by
 * tools/pco-fixtures/generate_attribute_fetch_shader.c. Fourteen public PCO
 * FADD groups accumulate vi0..vi15 through temp0/temp1, write the final x/y
 * into temp2/temp3, and use sc0/sc67 for z=0,w=8 in temp4/temp5. SHA-256:
 * 877802fe53fd258bb114aa2cf5713c317405c986b3a43b4612a58b6db9f7eccb.
 */
const std::vector<std::uint8_t> kAttributeFetchEightAttributeVertexBinary = {
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

/*
 * Exact gray fragment shader paired with the case-1/case-2/case-4 vertex
 * shaders. Each MBYP writes Mesa's public sc75 (IEEE-754 0.5) to PIXOUT0..3.
 * SHA-256:
 * 45a123bc247c1b37570721ad7a18894be4d6802dfb459de191ebfc3a32ec5959.
 */
const std::vector<std::uint8_t> kAttributeFetchGrayFragmentBinary = {
    0x35, 0x8a, 0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x20,
    0x35, 0x8a, 0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x21,
    0x36, 0x8a, 0x00, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x22,
    0xf1, 0xff,
    0x38, 0x8a, 0x80, 0x87, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x23,
    0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
};

/*
 * Exact GLBench varyings_shader_1 vertex fixture emitted by
 * tools/pco-fixtures/generate_varying_shader.c. The same vec2-backed GLES
 * input is exported as clip position VTXOUT0..3 and smooth v1 VTXOUT4..7.
 * SHA-256:
 * 09636842506c3a05b4dfae96d232274bb2eeb59876591e9fe29fc27a2e0860df.
 */
const std::vector<std::uint8_t> kVaryingsOneVertexBinary = {
    0x35, 0x82, 0x00, 0x87, 0x80, 0x04, 0x00, 0x00, 0x00, 0x40,
    0x35, 0x82, 0x00, 0x87, 0x81, 0x04, 0x00, 0x00, 0x00, 0x41,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x42,
    0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x43,
    0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
    0x58, 0xa0, 0x06, 0x08, 0x04, 0xc0, 0x00, 0x00, 0x00, 0x30,
    0xf3, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
};

/*
 * Exact GLBench varyings_shader_1 fragment fixture. Public FITRP.PIXEL reads
 * v1 coefficient sets cf4..19 and position-W set cf0..3, completes through
 * drc0/WDF, then exports temp0..3 to PIXOUT0..3. SHA-256:
 * a9c070ea3feb5dc4f7666b1fc019aaa9e3c522f5a8a73605ea07481104efc71c.
 */
const std::vector<std::uint8_t> kVaryingsOneFragmentBinary = {
    0x56, 0xa0, 0x00, 0xb0, 0x04, 0xc4, 0x40, 0x10, 0xc0, 0x40, 0x00, 0xff,
    0x02, 0x80, 0x6a, 0xff,
    0x34, 0x8a, 0x00, 0x87, 0x40, 0x00, 0x00, 0x20,
    0x34, 0x8a, 0x00, 0x87, 0x41, 0x00, 0x00, 0x21,
    0x34, 0x8a, 0x00, 0x87, 0x42, 0x00, 0x00, 0x22,
    0x34, 0x8a, 0x80, 0x87, 0x43, 0x00, 0x00, 0x23,
};

/*
 * Exact GLBench varyings_shader_2 vertex fixture emitted by the pinned public
 * Mesa generator. It exports c to VTXOUT0..3 and the independently executed
 * public PCO c*sc75 (0.5) result to both VTXOUT4..7 and VTXOUT8..11.
 * SHA-256:
 * 11a9256581cec718761818f8907337c86e458d2e44884ffe89a8d20c44647535.
 */
const std::vector<std::uint8_t> kVaryingsTwoVertexBinary = {
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

/*
 * Exact GLBench varyings_shader_2 fragment fixture. It issues two independent
 * public FITRP.PIXEL vec4 requests (cf20..35 and cf4..19), waits for each drc0
 * request, executes four public FADD groups, then exports the sum.
 * SHA-256:
 * 8c3c5427a0064009d8799a120f3e34645031f8c73b15a30ca224f0b007e21e99.
 */
const std::vector<std::uint8_t> kVaryingsTwoFragmentBinary = {
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

/*
 * Exact GLBench varyings_shader_4 vertex fixture. It exports c to
 * VTXOUT0..3, executes the public vi0/vi1 * sc76 (0.25) FMUL groups and
 * exports that one vec4 result independently to VTXOUT4..19. SHA-256:
 * d98cefa0385a774d1a7b0ddb0149cc6b5aca3023cccd287e3eeea1ca410f6538.
 */
const std::vector<std::uint8_t> kVaryingsFourVertexBinary = {
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

/*
 * Exact GLBench varyings_shader_4 fragment fixture. Four ordered public
 * FITRP.PIXEL/WDF pairs interpolate v1..v4; twelve public FADD groups retain
 * the source program's left-associative sum before four PIXOUT exports.
 * SHA-256:
 * f5c1fbac1b9281ce5093ba9c629c90ff5cd81e1807351f3bee2f1f5700f1a08a.
 */
const std::vector<std::uint8_t> kVaryingsFourFragmentBinary = {
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

/*
 * Exact GLBench varyings_shader_8 vertex fixture. It exports c to
 * VTXOUT0..3, executes the public vi0/vi1 * sc77 (0.125) FMUL groups and
 * exports that vec4 result independently to VTXOUT4..35. SHA-256:
 * f5314dcc5a24dca2c7d716b9d0c3bd1696df0038e826b34ce7f7e208945bb45a.
 */
const std::vector<std::uint8_t> kVaryingsEightVertexBinary = {
    0x35, 0x82, 0x00, 0x87, 0x80, 0x04, 0x00, 0x00, 0x00, 0x40, 0x35, 0x82,
    0x00, 0x87, 0x81, 0x04, 0x00, 0x00, 0x00, 0x41, 0x34, 0x82, 0x00, 0x87,
    0x00, 0x00, 0x00, 0x42, 0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00,
    0x00, 0x43, 0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30,
    0x35, 0x82, 0x00, 0x40, 0x81, 0xcd, 0x12, 0x00, 0x00, 0x45, 0x35, 0x82,
    0x00, 0x40, 0x80, 0xcd, 0x12, 0x00, 0x00, 0x44, 0x34, 0x82, 0x00, 0x87,
    0x00, 0x00, 0x00, 0x46, 0x35, 0x82, 0x00, 0x87, 0x8d, 0x01, 0x00, 0x00,
    0x00, 0x47, 0x55, 0xa0, 0x06, 0x08, 0x04, 0xc4, 0x00, 0x00, 0x00, 0x30,
    0x55, 0xa0, 0x06, 0x08, 0x08, 0xc4, 0x00, 0x00, 0x00, 0x30, 0x55, 0xa0,
    0x06, 0x08, 0x0c, 0xc4, 0x00, 0x00, 0x00, 0x30, 0x55, 0xa0, 0x06, 0x08,
    0x10, 0xc4, 0x00, 0x00, 0x00, 0x30, 0x55, 0xa0, 0x06, 0x08, 0x14, 0xc4,
    0x00, 0x00, 0x00, 0x30, 0x55, 0xa0, 0x06, 0x08, 0x18, 0xc4, 0x00, 0x00,
    0x00, 0x30, 0x55, 0xa0, 0x06, 0x08, 0x1c, 0xc4, 0x00, 0x00, 0x00, 0x30,
    0x56, 0xa0, 0x06, 0x08, 0x20, 0xc4, 0x00, 0x00, 0x00, 0x30, 0xf1, 0xff,
    0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
};

/*
 * Exact GLBench varyings_shader_8 fragment fixture. Eight ordered public
 * FITRP.PIXEL/WDF pairs interpolate v1..v8; twenty-eight public FADD groups
 * retain the source program's left-associative sum before PIXOUT export.
 * SHA-256:
 * aaebb7b4e027f846eecda4687dbc14fb10dc8b9bb3881ef0134cd0255449c385.
 */
const std::vector<std::uint8_t> kVaryingsEightFragmentBinary = {
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
    0x00, 0x00, 0x40, 0xff, 0x56, 0xa0, 0x00, 0xb0, 0x04, 0xc4, 0x40, 0x14,
    0xc0, 0x41, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff, 0x35, 0x82, 0x00, 0x00,
    0xc5, 0xa1, 0x00, 0x00, 0x45, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc6, 0xa2,
    0x00, 0x00, 0x46, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc7, 0xa3, 0x00, 0x00,
    0x47, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc0, 0xa4, 0x00, 0x00, 0x40, 0xff,
    0x56, 0xa0, 0x00, 0xb0, 0x04, 0xd4, 0x40, 0x14, 0xc0, 0x41, 0x00, 0xff,
    0x02, 0x80, 0x6a, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc5, 0xa1, 0x00, 0x00,
    0x45, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc6, 0xa2, 0x00, 0x00, 0x46, 0xff,
    0x35, 0x82, 0x00, 0x00, 0xc7, 0xa3, 0x00, 0x00, 0x47, 0xff, 0x35, 0x82,
    0x00, 0x00, 0xc0, 0xa4, 0x00, 0x00, 0x40, 0xff, 0x56, 0xa0, 0x00, 0xb0,
    0x04, 0xe4, 0x40, 0x14, 0xc0, 0x41, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff,
    0x35, 0x82, 0x00, 0x00, 0xc5, 0xa1, 0x00, 0x00, 0x45, 0xff, 0x35, 0x82,
    0x00, 0x00, 0xc6, 0xa2, 0x00, 0x00, 0x46, 0xff, 0x35, 0x82, 0x00, 0x00,
    0xc7, 0xa3, 0x00, 0x00, 0x47, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc0, 0xa4,
    0x00, 0x00, 0x40, 0xff, 0x56, 0xa0, 0x00, 0xb0, 0x04, 0xf4, 0x40, 0x14,
    0xc0, 0x41, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff, 0x35, 0x82, 0x00, 0x00,
    0xc5, 0xa1, 0x00, 0x00, 0x45, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc6, 0xa2,
    0x00, 0x00, 0x46, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc7, 0xa3, 0x00, 0x00,
    0x47, 0xff, 0x35, 0x82, 0x00, 0x00, 0xc0, 0xa4, 0x00, 0x00, 0x40, 0xff,
    0x34, 0x8a, 0x00, 0x87, 0x45, 0x00, 0x00, 0x20, 0x34, 0x8a, 0x00, 0x87,
    0x46, 0x00, 0x00, 0x21, 0x34, 0x8a, 0x00, 0x87, 0x47, 0x00, 0x00, 0x22,
    0x34, 0x8a, 0x80, 0x87, 0x40, 0x00, 0x00, 0x23,
};

/* Exact GLBench fill_tex_nearest fixtures emitted by
 * tools/pco-fixtures/generate_fill_tex_nearest.c. The VS keeps scale in SH0
 * and exports two linked texture-coordinate components. SHA-256:
 * 744eb5091914cc3b1d7c98ed59477b48615526e642556db5f0b317ce47f3b92b.
 */
const std::vector<std::uint8_t> kFillTexNearestVertexBinary = {
    0x35, 0x82, 0x00, 0x87, 0x80, 0x08, 0x00, 0x00, 0x00, 0x44, 0x35, 0x82,
    0x00, 0x40, 0x80, 0xe4, 0x10, 0x00, 0x00, 0x40, 0x35, 0x82, 0x00, 0x40,
    0x81, 0xe4, 0x10, 0x00, 0x00, 0x41, 0x34, 0x82, 0x00, 0x87, 0x00, 0x00,
    0x00, 0x42, 0x35, 0x82, 0x00, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x43,
    0x55, 0xa0, 0x06, 0x08, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x30, 0x57, 0xa0,
    0x02, 0x08, 0x04, 0x82, 0x04, 0x00, 0x00, 0x30, 0xf2, 0xff, 0xff, 0xff,
    0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff,
};

/* The FS executes public FITRP.PIXEL count2, reads the combined image/sampler
 * descriptor through SH0..3/SH8..11, issues SMP.2D.FCNORM count4 on DRC0,
 * waits with WDF, then exports TEMP0..3. SHA-256:
 * b3ff0bffe4326d88081904139f48ce3be1978559c229d64120ffb1172e43f1b4.
 */
const std::vector<std::uint8_t> kFillTexNearestFragmentBinary = {
    0x56, 0xa0, 0x00, 0xb0, 0x02, 0xc4, 0x40, 0x10, 0xc0, 0x50, 0x00, 0xff,
    0x02, 0x80, 0x6a, 0xff, 0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x52,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x53, 0x34, 0x82, 0x00, 0x87,
    0x00, 0x00, 0x00, 0x54, 0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x55,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x56, 0x34, 0x82, 0x00, 0x87,
    0x00, 0x00, 0x00, 0x57, 0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x58,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x59, 0x34, 0x82, 0x00, 0x87,
    0x00, 0x00, 0x00, 0x5a, 0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x5b,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x5c, 0x34, 0x82, 0x00, 0x87,
    0x00, 0x00, 0x00, 0x5d, 0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x5e,
    0x34, 0x82, 0x00, 0x87, 0x00, 0x00, 0x00, 0x5f, 0x57, 0xa0, 0x00, 0xf4,
    0x4c, 0x80, 0x70, 0x80, 0x08, 0x88, 0x80, 0xa0, 0x00, 0xff, 0x02, 0x80,
    0x6a, 0xff, 0x34, 0x8a, 0x00, 0x87, 0x40, 0x00, 0x00, 0x20, 0x34, 0x8a,
    0x00, 0x87, 0x41, 0x00, 0x00, 0x21, 0x37, 0x8a, 0x00, 0x87, 0x42, 0x00,
    0x00, 0x22, 0xf3, 0xff, 0xff, 0xff, 0xff, 0xff, 0x34, 0x8a, 0x80, 0x87,
    0x43, 0x00, 0x00, 0x23,
};

inline constexpr std::uint8_t kMainOpSingle = 0b100;
inline constexpr std::uint8_t kSingleOpBypass = 0b0111;
inline constexpr std::uint8_t kBackendOpUvs = 0b000;
inline constexpr std::uint8_t kBackendOpFitr = 0b101;
inline constexpr std::uint8_t kBackendOpDma = 0b111;
inline constexpr std::uint8_t kUvsOpWrite = 0b000;
inline constexpr std::uint8_t kUvsOpEmitEndTask = 0b101;
inline constexpr std::uint8_t kUvsOpWriteEmitEndTask = 0b110;
inline constexpr std::uint16_t kPixelOutput0SpecialIndex = 32;
inline constexpr std::uint16_t kSpecialConstantZero = 0;
inline constexpr std::uint16_t kSpecialConstantOne = 64;
inline constexpr std::uint16_t kSpecialConstantTwo = 65;
inline constexpr std::uint16_t kSpecialConstantFour = 66;
inline constexpr std::uint16_t kSpecialConstantEight = 67;
inline constexpr std::uint16_t kSpecialConstantHalf = 75;
inline constexpr std::uint16_t kSpecialConstantQuarter = 76;
inline constexpr std::uint16_t kSpecialConstantEighth = 77;
inline constexpr std::size_t kAttributeFetchTemporaryCount = 6;
inline constexpr std::uint8_t kControlOpWdf = 0b0101;

struct GroupHeader {
  std::size_t offset = 0;
  std::size_t total_bytes = 0;
  /* Public Mesa names this field "da"; no unverified expansion is assumed. */
  std::uint8_t da = 0;
  std::uint8_t operation_origin = 0;
  std::uint8_t repeat_count = 1;
  std::uint8_t control_op = 0;
  std::uint8_t control_misc = 0;
  bool control = false;
  bool output_load_check = false;
  bool write0_present = false;
  bool write1_present = false;
  bool end = false;
};

[[noreturn]] void DecodeError(std::size_t offset, const std::string &reason) {
  std::ostringstream message;
  message << "PCO decode error at byte " << offset << ": " << reason;
  throw std::runtime_error(message.str());
}

[[noreturn]] void ExecuteError(const std::string &reason) {
  throw std::runtime_error("PCO ISS execution error: " + reason);
}

std::uint32_t CheckedU32(std::size_t value, const char *description) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error(std::string(description) + " exceeds uint32_t");
  return static_cast<std::uint32_t>(value);
}

GroupHeader DecodeHeader(const std::vector<std::uint8_t> &binary,
                         std::size_t offset) {
  if (binary.size() - offset < 3)
    DecodeError(offset, "truncated extended instruction-group header");

  const std::uint8_t byte0 = binary[offset];
  const std::uint8_t byte1 = binary[offset + 1];
  const std::uint8_t byte2 = binary[offset + 2];

  if ((byte1 & 0x80U) == 0)
    DecodeError(offset + 1, "brief group headers are outside this subset");
  const std::uint8_t alu_type = (byte2 >> 5U) & 0x03U;
  const bool control = alu_type == 0x03U;
  if (!control && alu_type != 0)
    DecodeError(offset + 2, "non-main ALU group is outside this subset");
  if (!control && (byte2 & 0x18U) != 0)
    DecodeError(offset + 2, "reserved/atomic group bits are not supported");
  if ((byte1 & 0x01U) != 0 || (byte2 & 0x01U) != 0)
    DecodeError(offset + 1, "conditional execution is outside this subset");
  if (control && ((byte1 & 0x7eU) != 0 || (byte2 & 0x60U) != 0x60U))
    DecodeError(offset, "unsupported control instruction-group header");

  std::size_t encoded_words = byte0 & 0x0fU;
  if (encoded_words == 0)
    encoded_words = 16;
  const std::size_t total_bytes = encoded_words * 2;
  if (total_bytes < 3 || total_bytes > binary.size() - offset)
    DecodeError(offset, "instruction-group length exceeds the binary");

  GroupHeader header;
  header.offset = offset;
  header.total_bytes = total_bytes;
  header.da = byte0 >> 4U;
  header.operation_origin = (byte1 >> 4U) & 0x07U;
  header.output_load_check = (byte1 & 0x08U) != 0;
  header.write1_present = (byte1 & 0x04U) != 0;
  header.write0_present = (byte1 & 0x02U) != 0;
  header.control = control;
  if (control) {
    header.control_misc = (byte2 >> 7U) & 1U;
    header.control_op = (byte2 >> 1U) & 0x0fU;
    header.end = false;
    header.repeat_count = 1;
  } else {
    header.end = (byte2 & 0x80U) != 0;
    header.repeat_count =
        static_cast<std::uint8_t>(((byte2 >> 1U) & 3U) + 1U);
  }
  return header;
}

PcoRegisterRef DecodeOneLowerSource(const std::vector<std::uint8_t> &binary,
                                    std::size_t group_end,
                                    std::size_t &cursor) {
  if (cursor >= group_end)
    DecodeError(cursor, "missing lower source encoding");

  const std::size_t source_offset = cursor;
  const std::uint8_t byte0 = binary[cursor];
  PcoRegisterRef source;
  if ((byte0 & 0x80U) == 0) {
    source.bank = static_cast<PcoRegisterBank>((byte0 >> 6U) & 1U);
    source.index = byte0 & 0x3fU;
    ++cursor;
  } else {
    if (group_end - cursor < 3)
      DecodeError(cursor, "truncated extended lower source encoding");
    const std::uint8_t byte1 = binary[cursor + 1];
    const std::uint8_t byte2 = binary[cursor + 2];

    /* ext0=1, sel=0, ext1=0, is0=s0, and all reserved bits zero. */
    if ((byte1 & 0xf0U) != 0)
      DecodeError(cursor + 1, "unsupported lower-source selector/extension");
    if ((byte2 & 0xf8U) != 0)
      DecodeError(cursor + 2, "non-zero reserved lower-source bits");

    const std::uint8_t bank = static_cast<std::uint8_t>(
        ((byte0 >> 6U) & 1U) | (((byte1 >> 2U) & 3U) << 1U));
    const std::uint16_t index = static_cast<std::uint16_t>(
        (byte0 & 0x3fU) | ((byte1 & 3U) << 6U) | ((byte2 & 7U) << 8U));
    /* The public encoder uses the extended form for the observed UVSW temp0
     * and temp2 sources even though their bank/index fit in a short field. */
    const bool canonical_uvsw_temporary =
        bank == static_cast<std::uint8_t>(PcoRegisterBank::kTemporary) &&
        (index == 0 || index == 2 || index == 4) &&
        byte0 == static_cast<std::uint8_t>(0xc0U + index) &&
        byte1 == 0x00U && byte2 == 0x00U;
    if (bank <= 1 && index <= 63 && !canonical_uvsw_temporary)
      DecodeError(source_offset,
                  "non-canonical overlong lower-source encoding");

    source.bank = static_cast<PcoRegisterBank>(bank);
    source.index = index;
    cursor += 3;
  }

  if (source.bank == PcoRegisterBank::kSpecial) {
    if (source.index != kSpecialConstantZero &&
        source.index != kSpecialConstantOne &&
        source.index != kSpecialConstantTwo &&
        source.index != kSpecialConstantFour &&
        source.index != kSpecialConstantEight &&
        source.index != kSpecialConstantHalf &&
        source.index != kSpecialConstantQuarter &&
        source.index != kSpecialConstantEighth) {
      DecodeError(source_offset,
                  "unsupported public special-constant register");
    }
  } else if (source.bank == PcoRegisterBank::kVertexInput) {
    if (source.index >= kPcoVertexInputCount)
      DecodeError(source_offset,
                  "vertex-input register exceeds the modeled USC file");
  } else if (source.bank == PcoRegisterBank::kTemporary) {
    if (source.index >= kPcoTemporaryCount)
      DecodeError(source_offset,
                  "temporary register exceeds the modeled USC file");
  } else if (source.bank == PcoRegisterBank::kShared) {
    if (source.index != 0)
      DecodeError(source_offset,
                  "shared-register source is outside the exact SH0 profile");
  } else {
    DecodeError(source_offset,
                "source register bank is outside this PCO subset");
  }

  return source;
}

struct TwoLowerSources {
  PcoRegisterRef source0{};
  PcoRegisterRef source1{};
};

TwoLowerSources DecodeTwoLowerSources(
    const std::vector<std::uint8_t> &binary, std::size_t group_end,
    std::size_t &cursor) {
  const std::size_t source_offset = cursor;
  if (group_end - cursor < 2)
    DecodeError(cursor, "truncated two-source lower encoding");

  const std::uint8_t byte0 = binary[cursor];
  const std::uint8_t byte1 = binary[cursor + 1];
  const bool ext1 = (byte1 & 0x40U) != 0;
  if (ext1 && group_end - cursor < 3)
    DecodeError(cursor, "truncated extended two-source lower encoding");
  const std::uint8_t byte2 = ext1 ? binary[cursor + 2] : 0;

  /*
   * Exact public 2lo_2b7i_2b7i_2m packing selected by Mesa for FADD:
   * ext0=1, sel=1, ext2=0 and mux/is0=0. Public vertex-input/special forms
   * use ext1=1; the compiler's fragment TEMP+TEMP form canonically uses
   * ext1=0. Remaining bits carry the two independent bank/index pairs.
   */
  if ((byte0 & 0x80U) == 0)
    DecodeError(source_offset, "FADD lower source has ext0=0");
  if ((byte1 & 0x80U) == 0)
    DecodeError(source_offset + 1, "FADD lower source has sel=0");
  if (ext1 && (byte2 & 0x80U) != 0)
    DecodeError(source_offset + 2, "FADD lower source has ext2=1");
  if (ext1 && (byte2 & 0x60U) != 0)
    DecodeError(source_offset + 2, "FADD lower-source mux is unsupported");

  const std::uint8_t bank0 = static_cast<std::uint8_t>(
      ((byte0 >> 6U) & 1U) | (((byte2 >> 4U) & 1U) << 1U));
  const std::uint16_t index0 = static_cast<std::uint16_t>(
      (byte0 & 0x3fU) | (((byte2 >> 2U) & 1U) << 6U));
  const std::uint8_t bank1 = static_cast<std::uint8_t>(
      ((byte1 >> 5U) & 1U) | (((byte2 >> 3U) & 1U) << 1U));
  const std::uint16_t index1 = static_cast<std::uint16_t>(
      (byte1 & 0x1fU) | ((byte2 & 0x03U) << 5U));

  const PcoRegisterBank source0_bank = static_cast<PcoRegisterBank>(bank0);
  const PcoRegisterBank source1_bank = static_cast<PcoRegisterBank>(bank1);
  const bool canonical_scale_multiply =
      ext1 && source0_bank == PcoRegisterBank::kVertexInput &&
      (index0 == 0 || index0 == 1) &&
      source1_bank == PcoRegisterBank::kTemporary && index1 == 4 &&
      (byte0 == 0x80U || byte0 == 0x81U) && byte1 == 0xe4U &&
      byte2 == 0x10U;
  if ((source1_bank == PcoRegisterBank::kTemporary) != !ext1 &&
      !canonical_scale_multiply) {
    DecodeError(source_offset + 1,
                "two-source ext1 form is non-canonical for source1 bank");
  }
  const auto validate_source = [&](PcoRegisterBank bank, std::uint16_t index,
                                   std::size_t offset) {
    if (bank == PcoRegisterBank::kVertexInput) {
      if (index >= kPcoVertexInputCount)
        DecodeError(offset, "two-source vertex input exceeds the USC file");
      return;
    }
    if (bank == PcoRegisterBank::kTemporary) {
      if (index >= kPcoTemporaryCount)
        DecodeError(offset, "two-source temporary exceeds the USC file");
      return;
    }
    if (bank == PcoRegisterBank::kSpecial) {
      if (index != kSpecialConstantZero &&
          index != kSpecialConstantOne &&
          index != kSpecialConstantTwo &&
          index != kSpecialConstantFour &&
          index != kSpecialConstantEight &&
          index != kSpecialConstantHalf &&
          index != kSpecialConstantQuarter &&
          index != kSpecialConstantEighth) {
        DecodeError(offset, "unsupported two-source special constant");
      }
      return;
    }
    DecodeError(offset, "two-source register bank is outside this PCO subset");
  };
  validate_source(source0_bank, index0, source_offset);
  validate_source(source1_bank, index1, source_offset + 1);

  cursor += ext1 ? 3 : 2;
  return {
      {source0_bank, index0},
      {source1_bank, index1},
  };
}

std::uint16_t DecodePixelOutput(const std::vector<std::uint8_t> &binary,
                                std::size_t group_end, std::size_t &cursor) {
  if (cursor >= group_end)
    DecodeError(cursor, "missing destination encoding");
  const std::uint8_t byte = binary[cursor++];
  if ((byte & 0x80U) != 0)
    DecodeError(cursor - 1, "extended destinations are outside this subset");
  const std::uint8_t bank = (byte >> 6U) & 1U;
  const std::uint8_t special_index = byte & 0x3fU;
  if (bank != 0 || special_index < kPixelOutput0SpecialIndex ||
      special_index >= kPixelOutput0SpecialIndex + kPcoPixelOutputCount) {
    DecodeError(cursor - 1, "destination is not pixout0..pixout3");
  }
  return static_cast<std::uint16_t>(special_index - kPixelOutput0SpecialIndex);
}

void ValidateAlignmentPadding(const std::vector<std::uint8_t> &binary,
                              std::size_t group_start,
                              std::size_t semantic_end,
                              std::size_t group_end) {
  if (semantic_end > group_end)
    DecodeError(semantic_end,
                "semantic fields exceed the encoded group length");

  /* PCO groups are word encoded. An odd semantic byte count has a raw 0xff
   * word-padding byte before any length-coded alignment padding. */
  if (((semantic_end - group_start) & 1U) != 0) {
    if (semantic_end >= group_end || binary[semantic_end] != 0xffU)
      DecodeError(semantic_end, "invalid PCO word-padding byte");
    ++semantic_end;
  }

  const std::size_t padding = group_end - semantic_end;
  if (padding == 0)
    return;
  if ((padding & 1U) != 0 || padding / 2 > 15)
    DecodeError(semantic_end, "invalid PCO alignment-padding length");
  const std::uint8_t expected_first =
      static_cast<std::uint8_t>(0xf0U | (padding / 2U));
  if (binary[semantic_end] != expected_first)
    DecodeError(semantic_end, "invalid PCO alignment-padding marker");
  for (std::size_t index = semantic_end + 1; index < group_end; ++index) {
    if (binary[index] != 0xffU)
      DecodeError(index, "invalid PCO alignment-padding byte");
  }
}

std::uint16_t DecodeTemporaryDestination(
    const std::vector<std::uint8_t> &binary, std::size_t group_end,
    std::size_t &cursor);

PcoInstruction DecodeFragmentFitrpGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.control || header.da != 5 || header.operation_origin != 2 ||
      header.output_load_check || header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end) {
    DecodeError(header.offset, "unsupported FITRP instruction-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (group_end - cursor < 8)
    DecodeError(cursor, "truncated FITRP.PIXEL encoding");

  const std::uint8_t backend0 = binary[cursor++];
  const std::uint8_t backend1 = binary[cursor++];
  const std::uint8_t backend_op = backend0 >> 5U;
  const bool perspective = (backend0 & 0x10U) != 0;
  const std::uint8_t drc = (backend0 >> 3U) & 1U;
  const std::uint8_t reserved0 = (backend0 >> 2U) & 1U;
  const std::uint8_t iteration_mode = backend0 & 3U;
  const std::uint8_t reserved1 = backend1 >> 5U;
  const bool saturate = (backend1 & 0x10U) != 0;
  const std::uint8_t component_count = backend1 & 0x0fU;
  if (backend_op != kBackendOpFitr || !perspective || drc != 0 ||
      reserved0 != 0 || iteration_mode != 0 || reserved1 != 0 || saturate ||
      (component_count != 2 && component_count != 4)) {
    DecodeError(header.offset + 3,
                "FITRP must be perspective PIXEL/count2|4/drc0/no-saturate");
  }

  /* Public coefficient/temp source map used by the pinned varying programs:
   * s0 selects one vec4's four A/B/C/PAD coefficient sets, s2 selects the
   * position-W set cf0..3, and s3 selects a four-register TEMP destination.
   * Only compiler-emitted cf4/cf20/cf36/cf52/cf68/cf84/cf100/cf116 and
   * r0/r1/r4 encodings enter these gates; their exact pairings and ordering
   * are checked by profiles. */
  if (group_end - cursor < 6)
    DecodeError(cursor, "truncated FITRP source encoding");
  const std::uint8_t coefficient_byte = binary[cursor++];
  const std::uint8_t source1_byte = binary[cursor++];
  const std::uint8_t coefficient_high_byte = binary[cursor++];
  const std::uint8_t source3_byte = binary[cursor++];
  const std::uint16_t coefficient_index = static_cast<std::uint16_t>(
      (coefficient_byte & 0x3fU) |
      ((coefficient_high_byte & 0x04U) != 0 ? 0x40U : 0x00U));
  const bool supported_coefficient =
      coefficient_index == 4 || coefficient_index == 20 ||
      coefficient_index == 36 || coefficient_index == 52 ||
      coefficient_index == 68 || coefficient_index == 84 ||
      coefficient_index == 100 || coefficient_index == 116;
  if ((coefficient_byte & 0xc0U) != 0xc0U || !supported_coefficient ||
      source1_byte != 0x40U ||
      (coefficient_high_byte != 0x10U &&
       coefficient_high_byte != 0x14U) ||
      source3_byte != 0xc0U) {
    DecodeError(cursor - 1, "FITRP coefficient source encoding changed");
  }
  const std::uint8_t destination_byte = binary[cursor++];
  const std::uint16_t destination_index = destination_byte & 0x3fU;
  if ((destination_byte & 0xc0U) != 0x40U ||
      (destination_index != 0 && destination_index != 1 &&
       destination_index != 4 && destination_index != 16) ||
      binary[cursor++] != 0x00U) {
    DecodeError(cursor - 1, "FITRP temporary destination encoding changed");
  }
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatInterpolatePerspective;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = {PcoRegisterBank::kCoefficient, coefficient_index};
  instruction.source1 = {PcoRegisterBank::kCoefficient, 0};
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination_index;
  instruction.component_count = component_count;
  instruction.data_request = 0;
  instruction.iteration_mode = PcoIterationMode::kPixel;
  instruction.perspective = 1;
  instruction.saturate = 0;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeFragmentWdfGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (!header.control || header.da != 0 || header.total_bytes != 4 ||
      header.operation_origin != 0 || header.output_load_check ||
      header.write0_present || header.write1_present ||
      header.control_op != kControlOpWdf || header.control_misc != 0) {
    DecodeError(header.offset, "expected exact public WDF drc0 group");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  ValidateAlignmentPadding(binary, header.offset, header.offset + 3,
                           group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kWaitDataFence;
  instruction.target = PcoWriteTarget::kNone;
  instruction.binary_offset = CheckedU32(header.offset + 2, "PCO offset");
  instruction.group_index = group_index;
  instruction.component_count = 1;
  instruction.data_request = 0;
  instruction.source_count = 0;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeFragmentMoveGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.control)
    DecodeError(header.offset, "control group reached fragment MBYP decoder");
  if (header.da != 3 || header.operation_origin != 0 ||
      !header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1) {
    DecodeError(header.offset, "unsupported fragment instruction-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (cursor >= group_end)
    DecodeError(cursor, "missing MBYP instruction");
  const std::uint8_t main = binary[cursor++];
  if ((main >> 5U) != kMainOpSingle || (main & 0x10U) != 0 ||
      (main & 0x0fU) != kSingleOpBypass) {
    DecodeError(cursor - 1, "expected the public one-byte MBYP encoding");
  }

  const PcoRegisterRef source = DecodeOneLowerSource(binary, group_end, cursor);
  if (source.bank == PcoRegisterBank::kSpecial) {
    if (source.index != kSpecialConstantZero &&
        source.index != kSpecialConstantOne &&
        source.index != kSpecialConstantHalf) {
      DecodeError(cursor, "special constant is outside the fragment gate");
    }
  } else if (source.bank == PcoRegisterBank::kTemporary) {
    if (source.index >= kPcoTemporaryCount)
      DecodeError(cursor, "fragment MBYP source exceeds the TEMP file");
  } else {
    DecodeError(cursor,
                "fragment MBYP source must be a special constant or TEMP");
  }

  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported upper-source encoding");
  /* Public ISS field 0x00 selects ft0 for is4, the MBYP feed-through. */
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "expected ISS selection is4=ft0");

  const std::uint16_t output = DecodePixelOutput(binary, group_end, cursor);
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kMoveBypass;
  instruction.target = PcoWriteTarget::kPixelOutput;
  instruction.source = source;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = output;
  instruction.source_count = 1;
  instruction.repeat_count = 1;
  instruction.end_group = header.end ? 1U : 0U;
  return instruction;
}

PcoInstruction DecodeFragmentFloatAddGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.control || header.da != 3 || header.operation_origin != 0 ||
      header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end) {
    DecodeError(header.offset, "unsupported fragment FADD-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "expected unmodified public fragment FADD");
  const TwoLowerSources sources =
      DecodeTwoLowerSources(binary, group_end, cursor);
  if (sources.source0.bank != PcoRegisterBank::kTemporary ||
      sources.source1.bank != PcoRegisterBank::kTemporary) {
    DecodeError(header.offset,
                "varyings_shader_2 fragment FADD requires two TEMP sources");
  }
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported fragment FADD upper-source encoding");
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported fragment FADD ISS selection");
  const std::uint16_t temporary =
      DecodeTemporaryDestination(binary, group_end, cursor);
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatAdd;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = sources.source0;
  instruction.source1 = sources.source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = temporary;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeFragmentTemporaryMoveGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.control || header.da != 3 || header.operation_origin != 0 ||
      header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end ||
      header.total_bytes != 8) {
    DecodeError(header.offset,
                "unsupported fill_tex_nearest shared-to-TEMP group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (binary[cursor++] != 0x87U)
    DecodeError(cursor - 1, "expected exact public MBYP encoding");
  const PcoRegisterRef source =
      DecodeOneLowerSource(binary, group_end, cursor);
  /* These fourteen public groups initialize TEMP18..31 to the special zero
   * constant.  The SMP group itself reads the live descriptor from
   * SH0..3/SH8..11; the initialization groups must not be relabeled as
   * shared-register loads. */
  if (source.bank != PcoRegisterBank::kSpecial ||
      source.index != kSpecialConstantZero) {
    DecodeError(header.offset,
                "texture temporary initialization must read special zero");
  }
  if (cursor >= group_end || binary[cursor++] != 0x00U ||
      cursor >= group_end || binary[cursor++] != 0x00U) {
    DecodeError(cursor - 1,
                "shared-to-TEMP MBYP upper-source/ISS encoding changed");
  }
  const std::uint16_t destination =
      DecodeTemporaryDestination(binary, group_end, cursor);
  if (destination < 18 || destination > 31)
    DecodeError(header.offset,
                "texture state fill destination is outside TEMP18..31");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kMoveBypass;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = source;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination;
  instruction.source_count = 1;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeFragmentTextureSampleGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.control || header.da != 5 || header.operation_origin != 2 ||
      header.output_load_check || header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end ||
      header.total_bytes != 14) {
    DecodeError(header.offset, "unsupported SMP instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (group_end - cursor < 11)
    DecodeError(cursor, "truncated public SMP.2D.FCNORM group");

  /* Mesa pco_backend_smp_brief: backend_op=DMA, fcnorm=1, drc0,
   * dma_op=SMP; extb=0, dmn=2D, exta=0, chan=4, lodm=AUTO.
   */
  const std::uint8_t backend0 = binary[cursor++];
  const std::uint8_t backend1 = binary[cursor++];
  const std::uint8_t backend_op = backend0 >> 5U;
  const bool fcnorm = (backend0 & 0x10U) != 0;
  const std::uint8_t drc = (backend0 >> 3U) & 1U;
  const std::uint8_t dma_op = backend0 & 7U;
  const bool extb = (backend1 & 0x80U) != 0;
  const std::uint8_t dimension = (backend1 >> 5U) & 3U;
  const bool exta = (backend1 & 0x10U) != 0;
  const std::uint8_t channel_encoding = (backend1 >> 2U) & 3U;
  const std::uint8_t lod_mode = backend1 & 3U;
  if (backend_op != kBackendOpDma || !fcnorm || drc != 0 || dma_op != 4 ||
      extb || dimension != 2 || exta || channel_encoding != 3 ||
      lod_mode != 0) {
    DecodeError(header.offset + 3,
                "SMP must be exact 2D/FCNORM/count4/AUTO/drc0");
  }

  /* Exact public source/destination map printed by the pinned backend:
   * s0=SH0..3 texture state, s1=TEMP16..17 coordinates,
   * s2=SH8..11 sampler state, s4=TEMP0..3 result.
   */
  constexpr std::uint8_t kExactIo[] = {
      0x80, 0x70, 0x80, 0x08, 0x88, 0x80, 0xa0, 0x00,
  };
  for (std::uint8_t expected : kExactIo) {
    if (cursor >= group_end || binary[cursor] != expected)
      DecodeError(cursor, "SMP source/destination encoding changed");
    ++cursor;
  }
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kTextureSample;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = {PcoRegisterBank::kTemporary, 16};
  instruction.source1 = {PcoRegisterBank::kShared, 0};
  instruction.source2 = {PcoRegisterBank::kShared, 8};
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = 0;
  instruction.component_count = 4;
  instruction.data_request = 0;
  instruction.source_count = 3;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeFragmentGroup(const std::vector<std::uint8_t> &binary,
                                   const GroupHeader &header,
                                   std::uint16_t group_index) {
  if (header.control)
    return DecodeFragmentWdfGroup(binary, header, group_index);
  if (header.operation_origin == 2) {
    if (binary[header.offset + 3] >> 5U == kBackendOpDma)
      return DecodeFragmentTextureSampleGroup(binary, header, group_index);
    return DecodeFragmentFitrpGroup(binary, header, group_index);
  }
  if (!header.output_load_check &&
      binary[header.offset + 3] == 0x87U)
    return DecodeFragmentTemporaryMoveGroup(binary, header, group_index);
  if (!header.output_load_check)
    return DecodeFragmentFloatAddGroup(binary, header, group_index);
  return DecodeFragmentMoveGroup(binary, header, group_index);
}

std::uint16_t DecodeTemporaryDestination(
    const std::vector<std::uint8_t> &binary, std::size_t group_end,
    std::size_t &cursor) {
  if (cursor >= group_end)
    DecodeError(cursor, "missing temporary destination encoding");
  const std::size_t destination_offset = cursor;
  const std::uint8_t byte = binary[cursor++];
  if ((byte & 0x80U) != 0)
    DecodeError(destination_offset,
                "extended temporary destinations are outside this subset");
  const std::uint8_t bank = (byte >> 6U) & 1U;
  const std::uint8_t index = byte & 0x3fU;
  if (bank != static_cast<std::uint8_t>(PcoRegisterBank::kTemporary) ||
      index >= kPcoTemporaryCount) {
    DecodeError(destination_offset,
                "ALU destination exceeds the modeled temporary file");
  }
  return index;
}

PcoInstruction DecodeVertexMoveGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.da != 3 || header.operation_origin != 0 ||
      header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end) {
    DecodeError(header.offset, "unsupported vertex MBYP-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (cursor >= group_end || binary[cursor++] != 0x87U)
    DecodeError(cursor - 1, "expected the public one-byte MBYP encoding");

  const PcoRegisterRef source = DecodeOneLowerSource(binary, group_end, cursor);
  if (source.bank == PcoRegisterBank::kVertexInput) {
    if (source.index > 1)
      DecodeError(header.offset, "attribute-fetch MBYP only reads vi0 or vi1");
  } else if (source.bank == PcoRegisterBank::kSpecial) {
    if (source.index != kSpecialConstantZero &&
        source.index != kSpecialConstantOne &&
        source.index != kSpecialConstantTwo &&
        source.index != kSpecialConstantFour &&
        source.index != kSpecialConstantEight &&
        source.index != kSpecialConstantHalf &&
        source.index != kSpecialConstantQuarter &&
        source.index != kSpecialConstantEighth) {
      DecodeError(header.offset,
                  "vertex MBYP special constant is outside the exact gates");
    }
  } else if (source.bank == PcoRegisterBank::kShared) {
    if (source.index != 0)
      DecodeError(header.offset, "fill_tex_nearest VS must read SH0");
  } else {
    DecodeError(header.offset,
                "vertex MBYP source must be vertex input, shared or special");
  }

  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported upper-source encoding");
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "expected ISS selection is4=ft0");

  const std::uint16_t temporary =
      DecodeTemporaryDestination(binary, group_end, cursor);
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kMoveBypass;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = source;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = temporary;
  instruction.source_count = 1;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeVertexFloatAddGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.da != 3 || header.operation_origin != 0 ||
      header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end) {
    DecodeError(header.offset, "unsupported vertex FADD-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  /* main=0 is FADD with sat/s0neg/s0abs/s1abs/s0flr all clear. */
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "expected unmodified public FADD encoding");

  const TwoLowerSources sources =
      DecodeTwoLowerSources(binary, group_end, cursor);
  if ((sources.source0.bank != PcoRegisterBank::kVertexInput &&
       sources.source0.bank != PcoRegisterBank::kTemporary) ||
      sources.source1.bank != PcoRegisterBank::kVertexInput) {
    DecodeError(header.offset,
                "vertex FADD requires (vertex-input|temporary) + vertex-input");
  }
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported FADD upper-source encoding");
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported FADD ISS selection");

  const std::uint16_t temporary =
      DecodeTemporaryDestination(binary, group_end, cursor);
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatAdd;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = sources.source0;
  instruction.source1 = sources.source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = temporary;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeVertexFloatMultiplyGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.da != 3 || header.operation_origin != 0 ||
      header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end) {
    DecodeError(header.offset, "unsupported vertex FMUL-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  /* main=0x40 is public PCO FMUL with every modifier clear. */
  if (cursor >= group_end || binary[cursor++] != 0x40U)
    DecodeError(cursor - 1, "expected unmodified public FMUL encoding");

  const TwoLowerSources sources =
      DecodeTwoLowerSources(binary, group_end, cursor);
  const bool varying_constant =
      sources.source1.bank == PcoRegisterBank::kSpecial &&
      (sources.source1.index == kSpecialConstantHalf ||
       sources.source1.index == kSpecialConstantQuarter ||
       sources.source1.index == kSpecialConstantEighth);
  const bool fill_tex_scale =
      sources.source1.bank == PcoRegisterBank::kTemporary &&
      sources.source1.index == 4;
  if (sources.source0.bank != PcoRegisterBank::kVertexInput ||
      sources.source0.index > 1 ||
      (!varying_constant && !fill_tex_scale)) {
    DecodeError(header.offset,
                "FMUL must be varyings constant or fill_tex SH0 scale");
  }
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported FMUL upper-source encoding");
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported FMUL ISS selection");

  const std::uint16_t temporary =
      DecodeTemporaryDestination(binary, group_end, cursor);
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatMultiply;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = sources.source0;
  instruction.source1 = sources.source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = temporary;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeVertexBackendGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (header.operation_origin != 2 || header.output_load_check ||
      header.write0_present || header.write1_present) {
    DecodeError(header.offset, "unsupported vertex backend-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (cursor >= group_end)
    DecodeError(cursor, "truncated UVSW instruction");
  const std::uint8_t backend0 = binary[cursor++];
  const std::uint8_t backend_op = backend0 >> 5U;
  const bool dsel_w1 = (backend0 & 0x10U) != 0;
  const bool immediate_address = (backend0 & 0x08U) != 0;
  const std::uint8_t uvs_op = backend0 & 7U;

  if (header.da == 4 && backend_op == kBackendOpUvs && !dsel_w1 &&
      !immediate_address && uvs_op == kUvsOpEmitEndTask) {
    if (header.repeat_count != 1 || !header.end)
      DecodeError(header.offset,
                  "standalone UVSW.emit.endtask header is not exact");
    /* This group has three present but unused source/ISS fields. */
    for (const char *field : {"lower source", "upper source", "ISS"}) {
      if (cursor >= group_end || binary[cursor++] != 0x00U)
        DecodeError(cursor - 1,
                    std::string("standalone UVSW has non-zero ") + field);
    }
    ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

    PcoInstruction instruction;
    instruction.opcode = PcoOpcode::kUvsEmitEndTask;
    instruction.target = PcoWriteTarget::kNone;
    instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
    instruction.group_index = group_index;
    instruction.source_count = 0;
    instruction.repeat_count = 1;
    instruction.end_group = 1;
    return instruction;
  }

  if (header.da != 5 || group_end - cursor < 1)
    DecodeError(header.offset, "unsupported vertex backend-group header");
  const std::uint8_t address = binary[cursor++];
  if (backend_op != kBackendOpUvs || dsel_w1 || !immediate_address ||
      (uvs_op != kUvsOpWrite && uvs_op != kUvsOpWriteEmitEndTask)) {
    DecodeError(header.offset + 3,
                "unsupported public UVSW instruction encoding");
  }

  const PcoRegisterRef source = DecodeOneLowerSource(binary, group_end, cursor);
  if (source.bank == PcoRegisterBank::kVertexInput) {
    if (source.index + header.repeat_count > kPcoVertexInputCount)
      DecodeError(header.offset, "repeated vertex-input source exceeds USC file");
  } else if (source.bank == PcoRegisterBank::kSpecial) {
    if (header.repeat_count != 1)
      DecodeError(header.offset,
                  "repeated special-constant source is unsupported");
  } else if (source.bank == PcoRegisterBank::kTemporary) {
    if (header.repeat_count != 4 ||
        source.index + header.repeat_count > kPcoTemporaryCount) {
      DecodeError(header.offset,
                  "UVSW temporary range exceeds the modeled USC file");
    }
  } else {
    DecodeError(header.offset, "UVSW source bank is outside this subset");
  }
  if (static_cast<std::size_t>(address) + header.repeat_count >
      kPcoVertexOutputCount) {
    DecodeError(header.offset, "repeated UVSW write exceeds vertex outputs");
  }

  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported upper-source encoding");
  if (cursor >= group_end || binary[cursor++] != 0x30U)
    DecodeError(cursor - 1, "expected ISS selection is0=s0,is4=fte");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  const bool emits_and_ends = uvs_op == kUvsOpWriteEmitEndTask;
  if (emits_and_ends != header.end)
    DecodeError(header.offset,
                "UVSW end-task and instruction-group end bits disagree");

  PcoInstruction instruction;
  instruction.opcode =
      emits_and_ends ? PcoOpcode::kUvsWriteEmitEndTask : PcoOpcode::kUvsWrite;
  instruction.target = PcoWriteTarget::kVertexOutput;
  instruction.source = source;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = address;
  instruction.source_count = 1;
  instruction.repeat_count = header.repeat_count;
  instruction.end_group = header.end ? 1U : 0U;
  return instruction;
}

PcoInstruction DecodeVertexGroup(const std::vector<std::uint8_t> &binary,
                                 const GroupHeader &header,
                                 std::uint16_t group_index) {
  if (header.operation_origin == 0) {
    if (header.total_bytes <= 3)
      DecodeError(header.offset + 3, "vertex ALU group has no main operation");
    const std::uint8_t main = binary[header.offset + 3];
    if (main == 0x00U)
      return DecodeVertexFloatAddGroup(binary, header, group_index);
    if (main == 0x40U)
      return DecodeVertexFloatMultiplyGroup(binary, header, group_index);
    return DecodeVertexMoveGroup(binary, header, group_index);
  }
  if (header.operation_origin == 2)
    return DecodeVertexBackendGroup(binary, header, group_index);
  DecodeError(header.offset,
              "instruction origin is outside the vertex PCO subset");
}

bool IsRegister(const PcoRegisterRef &reference, PcoRegisterBank bank,
                std::uint16_t index) {
  return reference.bank == bank && reference.index == index;
}

bool HasDefaultNonFitrpFields(const PcoInstruction &instruction) {
  return instruction.component_count == 1 && instruction.data_request == 0 &&
         instruction.iteration_mode == PcoIterationMode::kPixel &&
         instruction.perspective == 0 && instruction.saturate == 0;
}

bool IsDefaultUnusedRegister(const PcoRegisterRef &reference) {
  return IsRegister(reference, PcoRegisterBank::kSpecial, 0);
}

bool MatchesMove(const PcoInstruction &instruction, PcoRegisterBank source_bank,
                 std::uint16_t source_index,
                 std::uint16_t destination_index) {
  return instruction.opcode == PcoOpcode::kMoveBypass &&
         instruction.target == PcoWriteTarget::kTemporary &&
         HasDefaultNonFitrpFields(instruction) &&
         instruction.source_count == 1 &&
         IsDefaultUnusedRegister(instruction.source1) &&
         IsRegister(instruction.source, source_bank, source_index) &&
         instruction.output_index == destination_index &&
         instruction.repeat_count == 1 && instruction.end_group == 0;
}

bool MatchesFadd(const PcoInstruction &instruction,
                 PcoRegisterBank source0_bank, std::uint16_t source0_index,
                 PcoRegisterBank source1_bank, std::uint16_t source1_index,
                 std::uint16_t destination_index) {
  return instruction.opcode == PcoOpcode::kFloatAdd &&
         instruction.target == PcoWriteTarget::kTemporary &&
         HasDefaultNonFitrpFields(instruction) &&
         instruction.source_count == 2 &&
         IsRegister(instruction.source, source0_bank, source0_index) &&
         IsRegister(instruction.source1, source1_bank, source1_index) &&
         instruction.output_index == destination_index &&
         instruction.repeat_count == 1 && instruction.end_group == 0;
}

bool MatchesFmul(const PcoInstruction &instruction,
                 PcoRegisterBank source0_bank, std::uint16_t source0_index,
                 PcoRegisterBank source1_bank, std::uint16_t source1_index,
                 std::uint16_t destination_index) {
  return instruction.opcode == PcoOpcode::kFloatMultiply &&
         instruction.target == PcoWriteTarget::kTemporary &&
         HasDefaultNonFitrpFields(instruction) &&
         instruction.source_count == 2 &&
         IsRegister(instruction.source, source0_bank, source0_index) &&
         IsRegister(instruction.source1, source1_bank, source1_index) &&
         instruction.output_index == destination_index &&
         instruction.repeat_count == 1 && instruction.end_group == 0;
}

bool MatchesTemporaryExport(const PcoInstruction &instruction,
                            std::uint16_t first_temporary,
                            std::uint16_t first_output = 0) {
  return instruction.opcode == PcoOpcode::kUvsWrite &&
         instruction.target == PcoWriteTarget::kVertexOutput &&
         HasDefaultNonFitrpFields(instruction) &&
         instruction.source_count == 1 &&
         IsDefaultUnusedRegister(instruction.source1) &&
         IsRegister(instruction.source, PcoRegisterBank::kTemporary,
                    first_temporary) &&
         instruction.output_index == first_output &&
         instruction.repeat_count == 4 &&
         instruction.end_group == 0;
}

bool MatchesStandaloneEmit(const PcoInstruction &instruction) {
  return instruction.opcode == PcoOpcode::kUvsEmitEndTask &&
         instruction.target == PcoWriteTarget::kNone &&
         HasDefaultNonFitrpFields(instruction) &&
         instruction.source_count == 0 && instruction.output_index == 0 &&
         IsDefaultUnusedRegister(instruction.source) &&
         IsDefaultUnusedRegister(instruction.source1) &&
         instruction.repeat_count == 1 && instruction.end_group == 1;
}

bool MatchesOneAttributeProfile(
    const std::vector<PcoInstruction> &instructions) {
  return instructions.size() == 6 &&
         MatchesMove(instructions[0], PcoRegisterBank::kVertexInput, 0, 0) &&
         MatchesMove(instructions[1], PcoRegisterBank::kVertexInput, 1, 1) &&
         MatchesMove(instructions[2], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 2) &&
         MatchesMove(instructions[3], PcoRegisterBank::kSpecial,
                     kSpecialConstantOne, 3) &&
         MatchesTemporaryExport(instructions[4], 0) &&
         MatchesStandaloneEmit(instructions[5]);
}

bool MatchesVaryingsOneProfile(
    const std::vector<PcoInstruction> &instructions) {
  return instructions.size() == 7 &&
         MatchesMove(instructions[0], PcoRegisterBank::kVertexInput, 0, 0) &&
         MatchesMove(instructions[1], PcoRegisterBank::kVertexInput, 1, 1) &&
         MatchesMove(instructions[2], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 2) &&
         MatchesMove(instructions[3], PcoRegisterBank::kSpecial,
                     kSpecialConstantOne, 3) &&
         MatchesTemporaryExport(instructions[4], 0, 0) &&
         MatchesTemporaryExport(instructions[5], 0, 4) &&
         MatchesStandaloneEmit(instructions[6]);
}

bool MatchesVaryingsTwoProfile(
    const std::vector<PcoInstruction> &instructions) {
  return instructions.size() == 12 &&
         MatchesMove(instructions[0], PcoRegisterBank::kVertexInput, 0, 0) &&
         MatchesMove(instructions[1], PcoRegisterBank::kVertexInput, 1, 1) &&
         MatchesMove(instructions[2], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 2) &&
         MatchesMove(instructions[3], PcoRegisterBank::kSpecial,
                     kSpecialConstantOne, 3) &&
         MatchesTemporaryExport(instructions[4], 0, 0) &&
         MatchesFmul(instructions[5], PcoRegisterBank::kVertexInput, 1,
                     PcoRegisterBank::kSpecial, kSpecialConstantHalf, 5) &&
         MatchesFmul(instructions[6], PcoRegisterBank::kVertexInput, 0,
                     PcoRegisterBank::kSpecial, kSpecialConstantHalf, 4) &&
         MatchesMove(instructions[7], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 6) &&
         MatchesMove(instructions[8], PcoRegisterBank::kSpecial,
                     kSpecialConstantHalf, 7) &&
         MatchesTemporaryExport(instructions[9], 4, 4) &&
         MatchesTemporaryExport(instructions[10], 4, 8) &&
         MatchesStandaloneEmit(instructions[11]);
}

bool MatchesVaryingsFourProfile(
    const std::vector<PcoInstruction> &instructions) {
  return instructions.size() == 14 &&
         MatchesMove(instructions[0], PcoRegisterBank::kVertexInput, 0, 0) &&
         MatchesMove(instructions[1], PcoRegisterBank::kVertexInput, 1, 1) &&
         MatchesMove(instructions[2], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 2) &&
         MatchesMove(instructions[3], PcoRegisterBank::kSpecial,
                     kSpecialConstantOne, 3) &&
         MatchesTemporaryExport(instructions[4], 0, 0) &&
         MatchesFmul(instructions[5], PcoRegisterBank::kVertexInput, 1,
                     PcoRegisterBank::kSpecial, kSpecialConstantQuarter, 5) &&
         MatchesFmul(instructions[6], PcoRegisterBank::kVertexInput, 0,
                     PcoRegisterBank::kSpecial, kSpecialConstantQuarter, 4) &&
         MatchesMove(instructions[7], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 6) &&
         MatchesMove(instructions[8], PcoRegisterBank::kSpecial,
                     kSpecialConstantQuarter, 7) &&
         MatchesTemporaryExport(instructions[9], 4, 4) &&
         MatchesTemporaryExport(instructions[10], 4, 8) &&
         MatchesTemporaryExport(instructions[11], 4, 12) &&
         MatchesTemporaryExport(instructions[12], 4, 16) &&
         MatchesStandaloneEmit(instructions[13]);
}

bool MatchesVaryingsEightProfile(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.size() != 18 ||
      !MatchesMove(instructions[0], PcoRegisterBank::kVertexInput, 0, 0) ||
      !MatchesMove(instructions[1], PcoRegisterBank::kVertexInput, 1, 1) ||
      !MatchesMove(instructions[2], PcoRegisterBank::kSpecial,
                   kSpecialConstantZero, 2) ||
      !MatchesMove(instructions[3], PcoRegisterBank::kSpecial,
                   kSpecialConstantOne, 3) ||
      !MatchesTemporaryExport(instructions[4], 0, 0) ||
      !MatchesFmul(instructions[5], PcoRegisterBank::kVertexInput, 1,
                   PcoRegisterBank::kSpecial, kSpecialConstantEighth, 5) ||
      !MatchesFmul(instructions[6], PcoRegisterBank::kVertexInput, 0,
                   PcoRegisterBank::kSpecial, kSpecialConstantEighth, 4) ||
      !MatchesMove(instructions[7], PcoRegisterBank::kSpecial,
                   kSpecialConstantZero, 6) ||
      !MatchesMove(instructions[8], PcoRegisterBank::kSpecial,
                   kSpecialConstantEighth, 7)) {
    return false;
  }
  for (std::size_t varying = 0; varying < 8; ++varying) {
    if (!MatchesTemporaryExport(
            instructions[varying + 9], 4,
            static_cast<std::uint16_t>(4 + varying * 4))) {
      return false;
    }
  }
  return MatchesStandaloneEmit(instructions[17]);
}

bool MatchesFillTexNearestVertexProfile(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.size() != 8 ||
      !MatchesMove(instructions[0], PcoRegisterBank::kShared, 0, 4) ||
      !MatchesFmul(instructions[1], PcoRegisterBank::kVertexInput, 0,
                   PcoRegisterBank::kTemporary, 4, 0) ||
      !MatchesFmul(instructions[2], PcoRegisterBank::kVertexInput, 1,
                   PcoRegisterBank::kTemporary, 4, 1) ||
      !MatchesMove(instructions[3], PcoRegisterBank::kSpecial,
                   kSpecialConstantZero, 2) ||
      !MatchesMove(instructions[4], PcoRegisterBank::kSpecial,
                   kSpecialConstantOne, 3) ||
      !MatchesTemporaryExport(instructions[5], 0, 0)) {
    return false;
  }
  const PcoInstruction &texcoord_export = instructions[6];
  return texcoord_export.opcode == PcoOpcode::kUvsWrite &&
         texcoord_export.target == PcoWriteTarget::kVertexOutput &&
         HasDefaultNonFitrpFields(texcoord_export) &&
         texcoord_export.source_count == 1 &&
         IsRegister(texcoord_export.source, PcoRegisterBank::kVertexInput, 2) &&
         IsDefaultUnusedRegister(texcoord_export.source1) &&
         texcoord_export.output_index == 4 &&
         texcoord_export.repeat_count == 2 && texcoord_export.end_group == 0 &&
         MatchesStandaloneEmit(instructions[7]);
}

bool MatchesTwoAttributeProfile(
    const std::vector<PcoInstruction> &instructions) {
  return instructions.size() == 6 &&
         MatchesFadd(instructions[0], PcoRegisterBank::kVertexInput, 0,
                     PcoRegisterBank::kVertexInput, 2, 0) &&
         MatchesFadd(instructions[1], PcoRegisterBank::kVertexInput, 1,
                     PcoRegisterBank::kVertexInput, 3, 1) &&
         MatchesMove(instructions[2], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 2) &&
         MatchesMove(instructions[3], PcoRegisterBank::kSpecial,
                     kSpecialConstantTwo, 3) &&
         MatchesTemporaryExport(instructions[4], 0) &&
         MatchesStandaloneEmit(instructions[5]);
}

bool MatchesFourAttributeProfile(
    const std::vector<PcoInstruction> &instructions) {
  return instructions.size() == 10 &&
         MatchesFadd(instructions[0], PcoRegisterBank::kVertexInput, 0,
                     PcoRegisterBank::kVertexInput, 2, 0) &&
         MatchesFadd(instructions[1], PcoRegisterBank::kVertexInput, 1,
                     PcoRegisterBank::kVertexInput, 3, 1) &&
         MatchesFadd(instructions[2], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 4, 0) &&
         MatchesFadd(instructions[3], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 5, 1) &&
         MatchesFadd(instructions[4], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 6, 2) &&
         MatchesFadd(instructions[5], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 7, 3) &&
         MatchesMove(instructions[6], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 4) &&
         MatchesMove(instructions[7], PcoRegisterBank::kSpecial,
                     kSpecialConstantFour, 5) &&
         MatchesTemporaryExport(instructions[8], 2) &&
         MatchesStandaloneEmit(instructions[9]);
}

bool MatchesEightAttributeProfile(
    const std::vector<PcoInstruction> &instructions) {
  return instructions.size() == 18 &&
         MatchesFadd(instructions[0], PcoRegisterBank::kVertexInput, 0,
                     PcoRegisterBank::kVertexInput, 2, 0) &&
         MatchesFadd(instructions[1], PcoRegisterBank::kVertexInput, 1,
                     PcoRegisterBank::kVertexInput, 3, 1) &&
         MatchesFadd(instructions[2], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 4, 0) &&
         MatchesFadd(instructions[3], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 5, 1) &&
         MatchesFadd(instructions[4], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 6, 0) &&
         MatchesFadd(instructions[5], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 7, 1) &&
         MatchesFadd(instructions[6], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 8, 0) &&
         MatchesFadd(instructions[7], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 9, 1) &&
         MatchesFadd(instructions[8], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 10, 0) &&
         MatchesFadd(instructions[9], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 11, 1) &&
         MatchesFadd(instructions[10], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 12, 0) &&
         MatchesFadd(instructions[11], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 13, 1) &&
         MatchesFadd(instructions[12], PcoRegisterBank::kTemporary, 0,
                     PcoRegisterBank::kVertexInput, 14, 2) &&
         MatchesFadd(instructions[13], PcoRegisterBank::kTemporary, 1,
                     PcoRegisterBank::kVertexInput, 15, 3) &&
         MatchesMove(instructions[14], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero, 4) &&
         MatchesMove(instructions[15], PcoRegisterBank::kSpecial,
                     kSpecialConstantEight, 5) &&
         MatchesTemporaryExport(instructions[16], 2) &&
         MatchesStandaloneEmit(instructions[17]);
}

void ValidateVertexTemporaryProgram(
    const std::vector<PcoInstruction> &instructions) {
  bool uses_temporary_program = false;
  std::uint32_t written_mask = 0;
  for (const PcoInstruction &instruction : instructions) {
    const auto require_written = [&](const PcoRegisterRef &source,
                                     std::uint8_t repeat_count) {
      if (source.bank != PcoRegisterBank::kTemporary)
        return;
      uses_temporary_program = true;
      for (std::uint8_t repeat = 0; repeat < repeat_count; ++repeat) {
        const std::size_t index = source.index + repeat;
        if (index >= kPcoTemporaryCount ||
            (written_mask & (UINT32_C(1) << index)) == 0) {
          DecodeError(instruction.binary_offset,
                      "temporary register is read before it is written");
        }
      }
    };
    if (instruction.source_count >= 1)
      require_written(instruction.source, instruction.repeat_count);
    if (instruction.source_count == 2)
      require_written(instruction.source1, instruction.repeat_count);

    if (instruction.target == PcoWriteTarget::kTemporary) {
      uses_temporary_program = true;
      if (instruction.output_index >= kPcoTemporaryCount)
        DecodeError(instruction.binary_offset,
                    "temporary destination exceeds the USC file");
      written_mask |= UINT32_C(1) << instruction.output_index;
    }
    if (instruction.opcode == PcoOpcode::kUvsEmitEndTask)
      uses_temporary_program = true;
  }

  if (uses_temporary_program &&
      !MatchesOneAttributeProfile(instructions) &&
      !MatchesTwoAttributeProfile(instructions) &&
      !MatchesFourAttributeProfile(instructions) &&
      !MatchesEightAttributeProfile(instructions) &&
      !MatchesVaryingsOneProfile(instructions) &&
      !MatchesVaryingsTwoProfile(instructions) &&
      !MatchesVaryingsFourProfile(instructions) &&
      !MatchesVaryingsEightProfile(instructions) &&
      !MatchesFillTexNearestVertexProfile(instructions)) {
    DecodeError(instructions.empty() ? 0 : instructions.front().binary_offset,
                "vertex TEMP program is outside the exact public profiles");
  }
}

bool MatchesVaryingsOneFragmentProfile(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.size() != 6)
    return false;
  const PcoInstruction &fitrp = instructions[0];
  const PcoInstruction &wdf = instructions[1];
  if (fitrp.opcode != PcoOpcode::kFloatInterpolatePerspective ||
      fitrp.target != PcoWriteTarget::kTemporary ||
      !IsRegister(fitrp.source, PcoRegisterBank::kCoefficient, 4) ||
      !IsRegister(fitrp.source1, PcoRegisterBank::kCoefficient, 0) ||
      fitrp.output_index != 0 || fitrp.component_count != 4 ||
      fitrp.data_request != 0 ||
      fitrp.iteration_mode != PcoIterationMode::kPixel ||
      fitrp.perspective != 1 || fitrp.saturate != 0 ||
      fitrp.source_count != 2 || fitrp.repeat_count != 1 ||
      fitrp.end_group != 0 || wdf.opcode != PcoOpcode::kWaitDataFence ||
      wdf.target != PcoWriteTarget::kNone || wdf.source_count != 0 ||
      !IsDefaultUnusedRegister(wdf.source) ||
      !IsDefaultUnusedRegister(wdf.source1) ||
      wdf.output_index != 0 || wdf.component_count != 1 ||
      wdf.data_request != 0 || wdf.repeat_count != 1 ||
      wdf.iteration_mode != PcoIterationMode::kPixel ||
      wdf.perspective != 0 || wdf.saturate != 0 || wdf.end_group != 0) {
    return false;
  }
  for (std::size_t component = 0; component < kPcoPixelOutputCount;
       ++component) {
    const PcoInstruction &move = instructions[component + 2];
    if (move.opcode != PcoOpcode::kMoveBypass ||
        move.target != PcoWriteTarget::kPixelOutput ||
        !IsRegister(move.source, PcoRegisterBank::kTemporary,
                    static_cast<std::uint16_t>(component)) ||
        move.source_count != 1 ||
        !IsDefaultUnusedRegister(move.source1) ||
        move.output_index != component ||
        !HasDefaultNonFitrpFields(move) || move.repeat_count != 1 ||
        move.end_group != (component + 1 == kPcoPixelOutputCount ? 1U : 0U)) {
      return false;
    }
  }
  return true;
}

bool MatchesVaryingsTwoFragmentProfile(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.size() != 12)
    return false;
  const auto matches_fitrp = [](const PcoInstruction &instruction,
                                std::uint16_t coefficient,
                                std::uint16_t destination) {
    return instruction.opcode ==
               PcoOpcode::kFloatInterpolatePerspective &&
           instruction.target == PcoWriteTarget::kTemporary &&
           IsRegister(instruction.source, PcoRegisterBank::kCoefficient,
                      coefficient) &&
           IsRegister(instruction.source1, PcoRegisterBank::kCoefficient, 0) &&
           instruction.output_index == destination &&
           instruction.component_count == 4 &&
           instruction.data_request == 0 &&
           instruction.iteration_mode == PcoIterationMode::kPixel &&
           instruction.perspective == 1 && instruction.saturate == 0 &&
           instruction.source_count == 2 && instruction.repeat_count == 1 &&
           instruction.end_group == 0;
  };
  const auto matches_wdf = [](const PcoInstruction &instruction) {
    return instruction.opcode == PcoOpcode::kWaitDataFence &&
           instruction.target == PcoWriteTarget::kNone &&
           instruction.source_count == 0 &&
           IsDefaultUnusedRegister(instruction.source) &&
           IsDefaultUnusedRegister(instruction.source1) &&
           instruction.output_index == 0 &&
           instruction.component_count == 1 &&
           instruction.data_request == 0 && instruction.repeat_count == 1 &&
           instruction.iteration_mode == PcoIterationMode::kPixel &&
           instruction.perspective == 0 && instruction.saturate == 0 &&
           instruction.end_group == 0;
  };
  if (!matches_fitrp(instructions[0], 20, 0) ||
      !matches_wdf(instructions[1]) ||
      !matches_fitrp(instructions[2], 4, 4) ||
      !matches_wdf(instructions[3]) ||
      !MatchesFadd(instructions[4], PcoRegisterBank::kTemporary, 4,
                   PcoRegisterBank::kTemporary, 0, 8) ||
      !MatchesFadd(instructions[5], PcoRegisterBank::kTemporary, 5,
                   PcoRegisterBank::kTemporary, 1, 9) ||
      !MatchesFadd(instructions[6], PcoRegisterBank::kTemporary, 6,
                   PcoRegisterBank::kTemporary, 2, 10) ||
      !MatchesFadd(instructions[7], PcoRegisterBank::kTemporary, 7,
                   PcoRegisterBank::kTemporary, 3, 0)) {
    return false;
  }

  constexpr std::uint16_t kSources[] = {8, 9, 10, 0};
  for (std::size_t component = 0; component < kPcoPixelOutputCount;
       ++component) {
    const PcoInstruction &move = instructions[component + 8];
    if (move.opcode != PcoOpcode::kMoveBypass ||
        move.target != PcoWriteTarget::kPixelOutput ||
        !IsRegister(move.source, PcoRegisterBank::kTemporary,
                    kSources[component]) ||
        move.source_count != 1 ||
        !IsDefaultUnusedRegister(move.source1) ||
        move.output_index != component ||
        !HasDefaultNonFitrpFields(move) || move.repeat_count != 1 ||
        move.end_group != (component + 1 == kPcoPixelOutputCount ? 1U : 0U)) {
      return false;
    }
  }
  return true;
}

bool MatchesVaryingsFourFragmentProfile(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.size() != 24)
    return false;
  const auto matches_fitrp = [](const PcoInstruction &instruction,
                                std::uint16_t coefficient,
                                std::uint16_t destination) {
    return instruction.opcode ==
               PcoOpcode::kFloatInterpolatePerspective &&
           instruction.target == PcoWriteTarget::kTemporary &&
           IsRegister(instruction.source, PcoRegisterBank::kCoefficient,
                      coefficient) &&
           IsRegister(instruction.source1, PcoRegisterBank::kCoefficient, 0) &&
           instruction.output_index == destination &&
           instruction.component_count == 4 &&
           instruction.data_request == 0 &&
           instruction.iteration_mode == PcoIterationMode::kPixel &&
           instruction.perspective == 1 && instruction.saturate == 0 &&
           instruction.source_count == 2 && instruction.repeat_count == 1 &&
           instruction.end_group == 0;
  };
  const auto matches_wdf = [](const PcoInstruction &instruction) {
    return instruction.opcode == PcoOpcode::kWaitDataFence &&
           instruction.target == PcoWriteTarget::kNone &&
           instruction.source_count == 0 &&
           IsDefaultUnusedRegister(instruction.source) &&
           IsDefaultUnusedRegister(instruction.source1) &&
           instruction.output_index == 0 && instruction.component_count == 1 &&
           instruction.data_request == 0 && instruction.repeat_count == 1 &&
           instruction.iteration_mode == PcoIterationMode::kPixel &&
           instruction.perspective == 0 && instruction.saturate == 0 &&
           instruction.end_group == 0;
  };

  if (!matches_fitrp(instructions[0], 20, 0) ||
      !matches_wdf(instructions[1]) ||
      !matches_fitrp(instructions[2], 4, 4) ||
      !matches_wdf(instructions[3]) ||
      !MatchesFadd(instructions[4], PcoRegisterBank::kTemporary, 4,
                   PcoRegisterBank::kTemporary, 0, 8) ||
      !MatchesFadd(instructions[5], PcoRegisterBank::kTemporary, 5,
                   PcoRegisterBank::kTemporary, 1, 9) ||
      !MatchesFadd(instructions[6], PcoRegisterBank::kTemporary, 6,
                   PcoRegisterBank::kTemporary, 2, 10) ||
      !MatchesFadd(instructions[7], PcoRegisterBank::kTemporary, 7,
                   PcoRegisterBank::kTemporary, 3, 0) ||
      !matches_fitrp(instructions[8], 36, 1) ||
      !matches_wdf(instructions[9]) ||
      !MatchesFadd(instructions[10], PcoRegisterBank::kTemporary, 8,
                   PcoRegisterBank::kTemporary, 1, 5) ||
      !MatchesFadd(instructions[11], PcoRegisterBank::kTemporary, 9,
                   PcoRegisterBank::kTemporary, 2, 6) ||
      !MatchesFadd(instructions[12], PcoRegisterBank::kTemporary, 10,
                   PcoRegisterBank::kTemporary, 3, 7) ||
      !MatchesFadd(instructions[13], PcoRegisterBank::kTemporary, 0,
                   PcoRegisterBank::kTemporary, 4, 0) ||
      !matches_fitrp(instructions[14], 52, 1) ||
      !matches_wdf(instructions[15]) ||
      !MatchesFadd(instructions[16], PcoRegisterBank::kTemporary, 5,
                   PcoRegisterBank::kTemporary, 1, 5) ||
      !MatchesFadd(instructions[17], PcoRegisterBank::kTemporary, 6,
                   PcoRegisterBank::kTemporary, 2, 6) ||
      !MatchesFadd(instructions[18], PcoRegisterBank::kTemporary, 7,
                   PcoRegisterBank::kTemporary, 3, 7) ||
      !MatchesFadd(instructions[19], PcoRegisterBank::kTemporary, 0,
                   PcoRegisterBank::kTemporary, 4, 0)) {
    return false;
  }

  constexpr std::uint16_t kSources[] = {5, 6, 7, 0};
  for (std::size_t component = 0; component < kPcoPixelOutputCount;
       ++component) {
    const PcoInstruction &move = instructions[component + 20];
    if (move.opcode != PcoOpcode::kMoveBypass ||
        move.target != PcoWriteTarget::kPixelOutput ||
        !IsRegister(move.source, PcoRegisterBank::kTemporary,
                    kSources[component]) ||
        move.source_count != 1 ||
        !IsDefaultUnusedRegister(move.source1) ||
        move.output_index != component ||
        !HasDefaultNonFitrpFields(move) || move.repeat_count != 1 ||
        move.end_group != (component + 1 == kPcoPixelOutputCount ? 1U : 0U)) {
      return false;
    }
  }
  return true;
}

bool MatchesVaryingsEightFragmentProfile(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.size() != 48)
    return false;
  const auto matches_fitrp = [](const PcoInstruction &instruction,
                                std::uint16_t coefficient,
                                std::uint16_t destination) {
    return instruction.opcode ==
               PcoOpcode::kFloatInterpolatePerspective &&
           instruction.target == PcoWriteTarget::kTemporary &&
           IsRegister(instruction.source, PcoRegisterBank::kCoefficient,
                      coefficient) &&
           IsRegister(instruction.source1, PcoRegisterBank::kCoefficient, 0) &&
           instruction.output_index == destination &&
           instruction.component_count == 4 &&
           instruction.data_request == 0 &&
           instruction.iteration_mode == PcoIterationMode::kPixel &&
           instruction.perspective == 1 && instruction.saturate == 0 &&
           instruction.source_count == 2 && instruction.repeat_count == 1 &&
           instruction.end_group == 0;
  };
  const auto matches_wdf = [](const PcoInstruction &instruction) {
    return instruction.opcode == PcoOpcode::kWaitDataFence &&
           instruction.target == PcoWriteTarget::kNone &&
           instruction.source_count == 0 &&
           IsDefaultUnusedRegister(instruction.source) &&
           IsDefaultUnusedRegister(instruction.source1) &&
           instruction.output_index == 0 && instruction.component_count == 1 &&
           instruction.data_request == 0 && instruction.repeat_count == 1 &&
           instruction.iteration_mode == PcoIterationMode::kPixel &&
           instruction.perspective == 0 && instruction.saturate == 0 &&
           instruction.end_group == 0;
  };

  if (!matches_fitrp(instructions[0], 20, 0) ||
      !matches_wdf(instructions[1]) ||
      !matches_fitrp(instructions[2], 4, 4) ||
      !matches_wdf(instructions[3]) ||
      !MatchesFadd(instructions[4], PcoRegisterBank::kTemporary, 4,
                   PcoRegisterBank::kTemporary, 0, 8) ||
      !MatchesFadd(instructions[5], PcoRegisterBank::kTemporary, 5,
                   PcoRegisterBank::kTemporary, 1, 9) ||
      !MatchesFadd(instructions[6], PcoRegisterBank::kTemporary, 6,
                   PcoRegisterBank::kTemporary, 2, 10) ||
      !MatchesFadd(instructions[7], PcoRegisterBank::kTemporary, 7,
                   PcoRegisterBank::kTemporary, 3, 0)) {
    return false;
  }

  constexpr std::uint16_t kCoefficients[] = {36, 52, 68, 84, 100, 116};
  constexpr std::uint16_t kInitialAccumulators[] = {8, 9, 10, 0};
  constexpr std::uint16_t kSteadyAccumulators[] = {5, 6, 7, 0};
  constexpr std::uint16_t kNewValues[] = {1, 2, 3, 4};
  constexpr std::uint16_t kDestinations[] = {5, 6, 7, 0};
  for (std::size_t layer = 0; layer < 6; ++layer) {
    const std::size_t group = 8 + layer * 6;
    if (!matches_fitrp(instructions[group], kCoefficients[layer], 1) ||
        !matches_wdf(instructions[group + 1])) {
      return false;
    }
    for (std::size_t component = 0; component < kPcoPixelOutputCount;
         ++component) {
      const std::uint16_t source0 =
          layer == 0 ? kInitialAccumulators[component]
                     : kSteadyAccumulators[component];
      if (!MatchesFadd(instructions[group + 2 + component],
                       PcoRegisterBank::kTemporary, source0,
                       PcoRegisterBank::kTemporary, kNewValues[component],
                       kDestinations[component])) {
        return false;
      }
    }
  }

  constexpr std::uint16_t kExportSources[] = {5, 6, 7, 0};
  for (std::size_t component = 0; component < kPcoPixelOutputCount;
       ++component) {
    const PcoInstruction &move = instructions[component + 44];
    if (move.opcode != PcoOpcode::kMoveBypass ||
        move.target != PcoWriteTarget::kPixelOutput ||
        !IsRegister(move.source, PcoRegisterBank::kTemporary,
                    kExportSources[component]) ||
        move.source_count != 1 ||
        !IsDefaultUnusedRegister(move.source1) ||
        move.output_index != component ||
        !HasDefaultNonFitrpFields(move) || move.repeat_count != 1 ||
        move.end_group != (component + 1 == kPcoPixelOutputCount ? 1U : 0U)) {
      return false;
    }
  }
  return true;
}

bool MatchesFillTexNearestFragmentProfile(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.size() != 22)
    return false;
  const auto matches_wdf = [](const PcoInstruction &instruction) {
    return instruction.opcode == PcoOpcode::kWaitDataFence &&
           instruction.target == PcoWriteTarget::kNone &&
           instruction.source_count == 0 &&
           IsDefaultUnusedRegister(instruction.source) &&
           IsDefaultUnusedRegister(instruction.source1) &&
           instruction.output_index == 0 && instruction.component_count == 1 &&
           instruction.data_request == 0 && instruction.repeat_count == 1 &&
           instruction.iteration_mode == PcoIterationMode::kPixel &&
           instruction.perspective == 0 && instruction.saturate == 0 &&
           instruction.end_group == 0;
  };
  const PcoInstruction &fitrp = instructions[0];
  if (fitrp.opcode != PcoOpcode::kFloatInterpolatePerspective ||
      fitrp.target != PcoWriteTarget::kTemporary ||
      !IsRegister(fitrp.source, PcoRegisterBank::kCoefficient, 4) ||
      !IsRegister(fitrp.source1, PcoRegisterBank::kCoefficient, 0) ||
      fitrp.output_index != 16 || fitrp.component_count != 2 ||
      fitrp.data_request != 0 ||
      fitrp.iteration_mode != PcoIterationMode::kPixel ||
      fitrp.perspective != 1 || fitrp.saturate != 0 ||
      fitrp.source_count != 2 || fitrp.repeat_count != 1 ||
      fitrp.end_group != 0 || !matches_wdf(instructions[1])) {
    return false;
  }
  for (std::size_t index = 0; index < 14; ++index) {
    if (!MatchesMove(instructions[index + 2], PcoRegisterBank::kSpecial,
                     kSpecialConstantZero,
                     static_cast<std::uint16_t>(18 + index))) {
      return false;
    }
  }
  const PcoInstruction &sample = instructions[16];
  if (sample.opcode != PcoOpcode::kTextureSample ||
      sample.target != PcoWriteTarget::kTemporary ||
      !IsRegister(sample.source, PcoRegisterBank::kTemporary, 16) ||
      !IsRegister(sample.source1, PcoRegisterBank::kShared, 0) ||
      !IsRegister(sample.source2, PcoRegisterBank::kShared, 8) ||
      sample.output_index != 0 || sample.component_count != 4 ||
      sample.data_request != 0 || sample.source_count != 3 ||
      sample.repeat_count != 1 || sample.end_group != 0 ||
      !matches_wdf(instructions[17])) {
    return false;
  }
  for (std::size_t component = 0; component < kPcoPixelOutputCount;
       ++component) {
    const PcoInstruction &move = instructions[component + 18];
    if (move.opcode != PcoOpcode::kMoveBypass ||
        move.target != PcoWriteTarget::kPixelOutput ||
        !IsRegister(move.source, PcoRegisterBank::kTemporary,
                    static_cast<std::uint16_t>(component)) ||
        move.source_count != 1 || !IsDefaultUnusedRegister(move.source1) ||
        move.output_index != component || !HasDefaultNonFitrpFields(move) ||
        move.repeat_count != 1 ||
        move.end_group != (component + 1 == kPcoPixelOutputCount ? 1U : 0U)) {
      return false;
    }
  }
  return true;
}

void ValidateFragmentProgram(
    const std::vector<PcoInstruction> &instructions) {
  if (instructions.empty())
    return;
  if (MatchesVaryingsOneFragmentProfile(instructions) ||
      MatchesVaryingsTwoFragmentProfile(instructions) ||
      MatchesVaryingsFourFragmentProfile(instructions) ||
      MatchesVaryingsEightFragmentProfile(instructions) ||
      MatchesFillTexNearestFragmentProfile(instructions))
    return;
  for (const PcoInstruction &instruction : instructions) {
    if (instruction.opcode == PcoOpcode::kFloatInterpolatePerspective ||
        instruction.opcode == PcoOpcode::kWaitDataFence) {
      DecodeError(instruction.binary_offset,
                  "FITRP/WDF program is outside the exact public profiles");
    }
  }
  // If instructions matching solid color raster, validate; otherwise allow general ALU stream
  bool all_mbyp = true;
  for (const PcoInstruction &inst : instructions) {
    if (inst.opcode != PcoOpcode::kMoveBypass) {
      all_mbyp = false;
      break;
    }
  }
  if (all_mbyp && instructions.size() == kPcoPixelOutputCount) {
    for (std::size_t component = 0; component < instructions.size();
         ++component) {
      const PcoInstruction &instruction = instructions[component];
      if (instruction.target != PcoWriteTarget::kPixelOutput ||
          instruction.source_count != 1 ||
          instruction.output_index != component) {
        DecodeError(instruction.binary_offset,
                    "fragment program is outside the exact public profiles");
      }
    }
  }
}

std::uint32_t ReadSource(const PcoRegisterRef &source,
                         const std::vector<std::uint32_t> &vertex_inputs,
                         const std::array<std::uint32_t, kPcoTemporaryCount>
                             &temporaries,
                         std::uint32_t temporary_written_mask,
                         std::uint8_t repeat_index, ShaderStage stage) {
  const std::size_t index =
      static_cast<std::size_t>(source.index) + repeat_index;
  switch (source.bank) {
  case PcoRegisterBank::kSpecial:
    if (repeat_index != 0)
      ExecuteError("special constants cannot be register-range repeated");
    if (source.index == kSpecialConstantZero)
      return UINT32_C(0x00000000);
    if (source.index == kSpecialConstantOne)
      return UINT32_C(0x3f800000);
    if (source.index == kSpecialConstantTwo)
      return UINT32_C(0x40000000);
    if (source.index == kSpecialConstantFour)
      return UINT32_C(0x40800000);
    if (source.index == kSpecialConstantEight)
      return UINT32_C(0x41000000);
    if (source.index == kSpecialConstantHalf)
      return UINT32_C(0x3f000000);
    if (source.index == kSpecialConstantQuarter)
      return UINT32_C(0x3e800000);
    if (source.index == kSpecialConstantEighth)
      return UINT32_C(0x3e000000);
    ExecuteError("unsupported special-constant register");
  case PcoRegisterBank::kVertexInput:
    if (stage != ShaderStage::kVertex)
      ExecuteError("fragment instruction read a vertex-input register");
    if (index >= vertex_inputs.size())
      ExecuteError("vertex-input register is absent from the lane");
    return vertex_inputs[index];
  case PcoRegisterBank::kTemporary:
    if (index >= temporaries.size())
      ExecuteError("temporary register exceeds the modeled USC file");
    if ((temporary_written_mask & (UINT32_C(1) << index)) == 0)
      ExecuteError("temporary register was read before it was written");
    return temporaries[index];
  default:
    ExecuteError("source register bank is outside this PCO subset");
  }
}

std::uint64_t ShiftRightJam(std::uint64_t value, std::uint32_t distance) {
  if (distance == 0)
    return value;
  if (distance >= 64)
    return value != 0 ? UINT64_C(1) : UINT64_C(0);
  const std::uint64_t discarded_mask =
      (UINT64_C(1) << distance) - UINT64_C(1);
  return (value >> distance) |
         ((value & discarded_mask) != 0 ? UINT64_C(1) : UINT64_C(0));
}

struct Binary32Operand {
  bool sign = false;
  bool zero = false;
  std::uint32_t exponent = 0;
  std::uint32_t significand = 0;
};

Binary32Operand DecodeFaddOperand(std::uint32_t bits) {
  Binary32Operand operand;
  operand.sign = (bits & UINT32_C(0x80000000)) != 0;
  operand.exponent = (bits >> 23U) & UINT32_C(0xff);
  const std::uint32_t fraction = bits & UINT32_C(0x007fffff);

  if (operand.exponent == UINT32_C(0xff))
    ExecuteError("FADD NaN/Inf policy is not in the public ISA gate");
  if (operand.exponent == 0) {
    if (fraction != 0)
      ExecuteError("FADD subnormal-input policy is not in the public ISA gate");
    operand.zero = true;
    return operand;
  }

  operand.significand = UINT32_C(0x00800000) | fraction;
  return operand;
}

std::uint16_t FloatToHalf(std::uint32_t bits) {
  std::uint32_t sign = (bits >> 16) & 0x8000;
  std::int32_t exp = ((bits >> 23) & 0xff) - 127;
  std::uint32_t mant = bits & 0x007fffff;

  if (exp == 128) {
    return static_cast<std::uint16_t>(sign | 0x7c00 | (mant != 0 ? 0x200 : 0));
  }
  if (exp > 15) {
    return static_cast<std::uint16_t>(sign | 0x7c00);
  }
  if (exp < -24) {
    return static_cast<std::uint16_t>(sign);
  }
  if (exp < -14) {
    std::uint32_t m = mant | 0x00800000;
    std::int32_t shift = -14 - exp;
    m >>= shift;
    return static_cast<std::uint16_t>(sign | m);
  }
  std::uint32_t m = mant >> 13;
  if ((mant & 0x1fff) > 0x1000 || ((mant & 0x3fff) == 0x3000)) {
    m++;
    if (m & 0x0400) {
      m = 0;
      exp++;
    }
  }
  if (exp > 15) {
    return static_cast<std::uint16_t>(sign | 0x7c00);
  }
  return static_cast<std::uint16_t>(sign | ((exp + 15) << 10) | m);
}

std::uint32_t HalfToFloat(std::uint16_t half) {
  std::uint32_t sign = (static_cast<std::uint32_t>(half) & 0x8000) << 16;
  std::int32_t exp = (half >> 10) & 0x1f;
  std::uint32_t mant = half & 0x03ff;

  if (exp == 31) {
    return sign | 0x7f800000 | (mant != 0 ? (mant << 13) : 0);
  }
  if (exp == 0) {
    if (mant == 0) {
      return sign;
    }
    while ((mant & 0x0400) == 0) {
      mant <<= 1;
      exp--;
    }
    exp++;
    mant &= ~0x0400;
  }
  std::uint32_t float_exp = static_cast<std::uint32_t>(exp - 15 + 127);
  return sign | (float_exp << 23) | (mant << 13);
}

std::uint32_t FloatAddBits(std::uint32_t left_bits,
                           std::uint32_t right_bits) {
  Binary32Operand left = DecodeFaddOperand(left_bits);
  Binary32Operand right = DecodeFaddOperand(right_bits);

  /* IEEE-754 round-to-nearest defines -0 + -0 as -0; every other exact-zero
   * sum in this supported rounding mode is +0. */
  if (left.zero && right.zero) {
    return left.sign && right.sign ? UINT32_C(0x80000000)
                                   : UINT32_C(0x00000000);
  }
  if (left.zero)
    return right_bits;
  if (right.zero)
    return left_bits;

  const auto magnitude_less = [](const Binary32Operand &first,
                                 const Binary32Operand &second) {
    return first.exponent < second.exponent ||
           (first.exponent == second.exponent &&
            first.significand < second.significand);
  };
  if (magnitude_less(left, right)) {
    const Binary32Operand temporary = left;
    left = right;
    right = temporary;
  }

  std::uint32_t exponent = left.exponent;
  std::uint64_t left_significand =
      static_cast<std::uint64_t>(left.significand) << 3U;
  std::uint64_t right_significand =
      static_cast<std::uint64_t>(right.significand) << 3U;
  right_significand =
      ShiftRightJam(right_significand, left.exponent - right.exponent);

  std::uint64_t result_significand = 0;
  const bool result_sign = left.sign;
  if (left.sign == right.sign) {
    result_significand = left_significand + right_significand;
    if ((result_significand & (UINT64_C(1) << 27U)) != 0) {
      result_significand = ShiftRightJam(result_significand, 1);
      ++exponent;
    }
  } else {
    result_significand = left_significand - right_significand;
    if (result_significand == 0)
      return UINT32_C(0x00000000);
    while ((result_significand & (UINT64_C(1) << 26U)) == 0) {
      if (exponent == 1) {
        ExecuteError(
            "FADD subnormal-result policy is not in the public ISA gate");
      }
      result_significand <<= 1U;
      --exponent;
    }
  }

  if (exponent == 0) {
    ExecuteError("FADD subnormal-result policy is not in the public ISA gate");
  }
  if (exponent >= UINT32_C(0xff))
    ExecuteError("FADD overflow policy is not in the public ISA gate");

  const std::uint32_t guard_round_sticky =
      static_cast<std::uint32_t>(result_significand & UINT64_C(7));
  std::uint32_t rounded_significand =
      static_cast<std::uint32_t>(result_significand >> 3U);
  const bool round_up =
      guard_round_sticky > 4 ||
      (guard_round_sticky == 4 && (rounded_significand & 1U) != 0);
  if (round_up)
    ++rounded_significand;

  if (rounded_significand == UINT32_C(0x01000000)) {
    rounded_significand >>= 1U;
    ++exponent;
    if (exponent >= UINT32_C(0xff))
      ExecuteError("FADD overflow policy is not in the public ISA gate");
  }
  if (rounded_significand < UINT32_C(0x00800000)) {
    ExecuteError("FADD subnormal-result policy is not in the public ISA gate");
  }

  return (result_sign ? UINT32_C(0x80000000) : UINT32_C(0)) |
         (exponent << 23U) |
         (rounded_significand & UINT32_C(0x007fffff));
}


std::uint32_t FloatMadBits(std::uint32_t a_bits, std::uint32_t b_bits, std::uint32_t c_bits) {
  float a_val = 0.0f, b_val = 0.0f, c_val = 0.0f;
  std::memcpy(&a_val, &a_bits, sizeof(a_val));
  std::memcpy(&b_val, &b_bits, sizeof(b_val));
  std::memcpy(&c_val, &c_bits, sizeof(c_val));
  float result_val = a_val * b_val + c_val;
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t FloatMinBits(std::uint32_t left_bits, std::uint32_t right_bits) {
  float left_val = 0.0f, right_val = 0.0f;
  std::memcpy(&left_val, &left_bits, sizeof(left_val));
  std::memcpy(&right_val, &right_bits, sizeof(right_val));
  float result_val = std::min(left_val, right_val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t FloatMaxBits(std::uint32_t left_bits, std::uint32_t right_bits) {
  float left_val = 0.0f, right_val = 0.0f;
  std::memcpy(&left_val, &left_bits, sizeof(left_val));
  std::memcpy(&right_val, &right_bits, sizeof(right_val));
  float result_val = std::max(left_val, right_val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t ReciprocalBits(std::uint32_t val_bits) {
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  float result_val = (val == 0.0f) ? 0.0f : (1.0f / val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t ReciprocalSquareRootBits(std::uint32_t val_bits) {
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  float result_val = (val <= 0.0f) ? 0.0f : (1.0f / std::sqrt(val));
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t FloatLog2Bits(std::uint32_t val_bits) {
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  float result_val = (val <= 0.0f) ? 0.0f : std::log2(val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t FloatExp2Bits(std::uint32_t val_bits) {
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  float result_val = std::exp2(val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t FloatSineBits(std::uint32_t val_bits) {
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  float result_val = std::sin(val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t FloatCosineBits(std::uint32_t val_bits) {
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  float result_val = std::cos(val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint64_t RoundRightToNearestEven(std::uint64_t value,
                                     std::uint32_t distance) {
  if (distance == 0)
    return value;
  if (distance >= 64)
    return 0;
  const std::uint64_t quotient = value >> distance;
  const std::uint64_t remainder_mask =
      (UINT64_C(1) << distance) - UINT64_C(1);
  const std::uint64_t remainder = value & remainder_mask;
  const std::uint64_t halfway = UINT64_C(1) << (distance - 1U);
  return quotient +
         (remainder > halfway ||
                  (remainder == halfway && (quotient & UINT64_C(1)) != 0)
              ? UINT64_C(1)
              : UINT64_C(0));
}

std::uint32_t FloatMultiplyBits(std::uint32_t left_bits,
                                std::uint32_t right_bits) {
  const Binary32Operand left = DecodeFaddOperand(left_bits);
  const Binary32Operand right = DecodeFaddOperand(right_bits);
  const bool sign = left.sign != right.sign;
  if (left.zero || right.zero)
    return sign ? UINT32_C(0x80000000) : UINT32_C(0x00000000);

  const std::uint64_t product =
      static_cast<std::uint64_t>(left.significand) * right.significand;
  const bool high = (product & (UINT64_C(1) << 47U)) != 0;
  const std::uint32_t shift = high ? 24U : 23U;
  std::int32_t exponent = static_cast<std::int32_t>(left.exponent) +
                          static_cast<std::int32_t>(right.exponent) - 127 +
                          (high ? 1 : 0);
  std::uint64_t significand = RoundRightToNearestEven(product, shift);
  if (significand == UINT64_C(0x01000000)) {
    significand >>= 1U;
    ++exponent;
  }
  if (exponent <= 0)
    ExecuteError("FMUL subnormal-result policy is outside this ISA gate");
  if (exponent >= 255)
    ExecuteError("FMUL overflow policy is outside this ISA gate");
  if (significand < UINT64_C(0x00800000) ||
      significand >= UINT64_C(0x01000000)) {
    ExecuteError("FMUL normalization failed");
  }
  return (sign ? UINT32_C(0x80000000) : UINT32_C(0)) |
         (static_cast<std::uint32_t>(exponent) << 23U) |
         (static_cast<std::uint32_t>(significand) & UINT32_C(0x007fffff));
}

std::uint32_t FloatDivideBits(std::uint32_t numerator_bits,
                              std::uint32_t denominator_bits) {
  const Binary32Operand numerator = DecodeFaddOperand(numerator_bits);
  const Binary32Operand denominator = DecodeFaddOperand(denominator_bits);
  if (denominator.zero)
    ExecuteError("FITRP perspective divide by zero");
  const bool sign = numerator.sign != denominator.sign;
  if (numerator.zero)
    return sign ? UINT32_C(0x80000000) : UINT32_C(0x00000000);

  const bool less = numerator.significand < denominator.significand;
  const std::uint32_t shift = less ? 24U : 23U;
  const std::uint64_t dividend =
      static_cast<std::uint64_t>(numerator.significand) << shift;
  std::uint64_t significand = dividend / denominator.significand;
  const std::uint64_t remainder = dividend % denominator.significand;
  const std::uint64_t twice_remainder = remainder << 1U;
  if (twice_remainder > denominator.significand ||
      (twice_remainder == denominator.significand &&
       (significand & UINT64_C(1)) != 0)) {
    ++significand;
  }

  std::int32_t exponent =
      static_cast<std::int32_t>(numerator.exponent) -
      static_cast<std::int32_t>(denominator.exponent) + 127 -
      (less ? 1 : 0);
  if (significand == UINT64_C(0x01000000)) {
    significand >>= 1U;
    ++exponent;
  }
  if (exponent <= 0)
    ExecuteError("FDIV subnormal-result policy is outside this ISA gate");
  if (exponent >= 255)
    ExecuteError("FDIV overflow policy is outside this ISA gate");
  if (significand < UINT64_C(0x00800000) ||
      significand >= UINT64_C(0x01000000)) {
    ExecuteError("FDIV normalization failed");
  }
  return (sign ? UINT32_C(0x80000000) : UINT32_C(0)) |
         (static_cast<std::uint32_t>(exponent) << 23U) |
         (static_cast<std::uint32_t>(significand) & UINT32_C(0x007fffff));
}

std::uint32_t EvaluateCoefficientPlane(
    const PcoFragmentExecutionContext &context, std::size_t coefficient_base) {
  if ((context.coefficient_count != kPcoFillTexNearestCoefficientCount &&
       context.coefficient_count != kPcoVaryingOneCoefficientCount &&
       context.coefficient_count != kPcoVaryingTwoCoefficientCount &&
       context.coefficient_count != kPcoVaryingFourCoefficientCount &&
       context.coefficient_count != kPcoVaryingEightCoefficientCount) ||
      coefficient_base + 3 >= context.coefficient_count) {
    ExecuteError("FITRP coefficient range is absent or truncated");
  }
  const std::uint32_t ax = FloatMultiplyBits(
      context.coefficients[coefficient_base], context.sample_x);
  const std::uint32_t by = FloatMultiplyBits(
      context.coefficients[coefficient_base + 1], context.sample_y);
  return FloatAddBits(FloatAddBits(ax, by),
                      context.coefficients[coefficient_base + 2]);
}

void ValidateExecutionEnvelope(const PcoProgramSummary &summary,
                               const std::vector<PcoInstruction> &instructions,
                               ShaderStage stage) {
  if (summary.stage != stage)
    ExecuteError("shader stage does not match the executor");
  if (summary.binary_size == 0 || summary.group_count == 0 ||
      summary.instruction_count != instructions.size() ||
      summary.group_count != instructions.size()) {
    ExecuteError("decoded program summary/instruction count mismatch");
  }
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    const PcoInstruction &instruction = instructions[index];
    if (instruction.group_index != index)
      ExecuteError("decoded groups are not contiguous and ordered");
    if (instruction.binary_offset >= summary.binary_size ||
        (index != 0 &&
         instruction.binary_offset <= instructions[index - 1].binary_offset)) {
      ExecuteError("decoded binary offsets are invalid or unordered");
    }
    if (instruction.repeat_count == 0 || instruction.repeat_count > 4)
      ExecuteError("invalid decoded group repeat count");
    if (instruction.component_count == 0 ||
        instruction.component_count > kPcoPixelOutputCount)
      ExecuteError("invalid decoded component count");
    if (instruction.source_count > 3)
      ExecuteError("invalid decoded source count");
    if (instruction.data_request > 1 || instruction.perspective > 1 ||
        instruction.saturate > 1)
      ExecuteError("decoded FITRP/control flag is not canonical");
    if (instruction.end_group > 1)
      ExecuteError("decoded .end marker is not Boolean");
    const bool is_final = index + 1 == instructions.size();
    if ((instruction.end_group != 0) != is_final)
      ExecuteError("decoded .end marker is not on the final group only");
  }
}

} // namespace

const std::vector<std::uint8_t> &FillSolidVertexPcoBinary() {
  return kFillSolidVertexBinary;
}

const std::vector<std::uint8_t> &FillSolidFragmentPcoBinary() {
  return kFillSolidFragmentBinary;
}

const std::vector<std::uint8_t> &FillSolidBlackFragmentPcoBinary() {
  return kFillSolidBlackFragmentBinary;
}

const std::vector<std::uint8_t> &FillSolidRedHalfAlphaFragmentPcoBinary() {
  return kFillSolidRedHalfAlphaFragmentBinary;
}

const std::vector<std::uint8_t> &FillSolidGreenHalfAlphaFragmentPcoBinary() {
  return kFillSolidGreenHalfAlphaFragmentBinary;
}

const std::vector<std::uint8_t> &TriangleSetupOrangeFragmentPcoBinary() {
  return kTriangleSetupOrangeFragmentBinary;
}

const std::vector<std::uint8_t> &TriangleSetupCyanFragmentPcoBinary() {
  return kTriangleSetupHalfCulledCyanFragmentBinary;
}

const std::vector<std::uint8_t> &AttributeFetchVertexPcoBinary() {
  return kAttributeFetchVertexBinary;
}

const std::vector<std::uint8_t> &
AttributeFetchTwoAttributeVertexPcoBinary() {
  return kAttributeFetchTwoAttributeVertexBinary;
}

const std::vector<std::uint8_t> &
AttributeFetchFourAttributeVertexPcoBinary() {
  return kAttributeFetchFourAttributeVertexBinary;
}

const std::vector<std::uint8_t> &
AttributeFetchEightAttributeVertexPcoBinary() {
  return kAttributeFetchEightAttributeVertexBinary;
}

const std::vector<std::uint8_t> &AttributeFetchGrayFragmentPcoBinary() {
  return kAttributeFetchGrayFragmentBinary;
}

const std::vector<std::uint8_t> &VaryingsOneVertexPcoBinary() {
  return kVaryingsOneVertexBinary;
}

const std::vector<std::uint8_t> &VaryingsOneFragmentPcoBinary() {
  return kVaryingsOneFragmentBinary;
}

const std::vector<std::uint8_t> &VaryingsTwoVertexPcoBinary() {
  return kVaryingsTwoVertexBinary;
}

const std::vector<std::uint8_t> &VaryingsTwoFragmentPcoBinary() {
  return kVaryingsTwoFragmentBinary;
}

const std::vector<std::uint8_t> &VaryingsFourVertexPcoBinary() {
  return kVaryingsFourVertexBinary;
}

const std::vector<std::uint8_t> &VaryingsFourFragmentPcoBinary() {
  return kVaryingsFourFragmentBinary;
}

const std::vector<std::uint8_t> &VaryingsEightVertexPcoBinary() {
  return kVaryingsEightVertexBinary;
}

const std::vector<std::uint8_t> &VaryingsEightFragmentPcoBinary() {
  return kVaryingsEightFragmentBinary;
}

const std::vector<std::uint8_t> &FillTexNearestVertexPcoBinary() {
  return kFillTexNearestVertexBinary;
}

const std::vector<std::uint8_t> &FillTexNearestFragmentPcoBinary() {
  return kFillTexNearestFragmentBinary;
}

PcoInstructionCounts
CountPcoInstructions(const std::vector<PcoInstruction> &instructions,
                     bool expand_repeats) {
  PcoInstructionCounts counts;
  for (const PcoInstruction &instruction : instructions) {
    if (instruction.repeat_count == 0 || instruction.repeat_count > 4)
      ExecuteError("instruction counter received an invalid repeat count");
    if (instruction.component_count == 0 ||
        instruction.component_count > kPcoPixelOutputCount) {
      ExecuteError("instruction counter received an invalid component count");
    }
    /* FITRP vec4 is one issued PCO instruction. component_count describes
     * result data width; only the encoded group repeat expands instruction
     * execution counters. */
    const std::uint64_t amount =
        expand_repeats ? instruction.repeat_count : UINT64_C(1);
    switch (instruction.opcode) {
    case PcoOpcode::kMoveBypass:
    case PcoOpcode::kFloatAdd:
    case PcoOpcode::kFloatMultiply:
    case PcoOpcode::kFloatMad:
    case PcoOpcode::kFloatMin:
    case PcoOpcode::kFloatMax:
    case PcoOpcode::kReciprocal:
    case PcoOpcode::kReciprocalSquareRoot:
    case PcoOpcode::kFloatLog2:
    case PcoOpcode::kFloatExp2:
    case PcoOpcode::kBranch:
    case PcoOpcode::kBranchConditional:
    case PcoOpcode::kLoopBegin:
    case PcoOpcode::kLoopEnd:
    case PcoOpcode::kIntegerAdd:
    case PcoOpcode::kBitwiseAnd:
    case PcoOpcode::kBitwiseOr:
    case PcoOpcode::kBitwiseXor:
    case PcoOpcode::kFloatSine:
    case PcoOpcode::kFloatCosine:
    case PcoOpcode::kDerivativeX:
    case PcoOpcode::kDerivativeY:
    case PcoOpcode::kPackHalf2x16:
    case PcoOpcode::kUnpackHalf2x16:
    case PcoOpcode::kFloatInterpolatePerspective:
      counts.alu += amount;
      break;
    case PcoOpcode::kTextureSample:
    case PcoOpcode::kTextureSampleLod:
    case PcoOpcode::kTextureGather:
      counts.texture += amount;
      break;
    case PcoOpcode::kWaitDataFence:
      /* Control/synchronization is intentionally not mislabeled ALU/TEX/MEM. */
      break;
    case PcoOpcode::kUvsWrite:
    case PcoOpcode::kUvsWriteEmitEndTask:
    case PcoOpcode::kUvsEmitEndTask:
    case PcoOpcode::kBufferLoad:
    case PcoOpcode::kBufferStore:
    case PcoOpcode::kDiscard:
    case PcoOpcode::kAtomicAdd:
    case PcoOpcode::kAtomicCompSwap:
      counts.memory += amount;
      break;
    default:
      ExecuteError("instruction counter received an unknown opcode");
    }
  }
  return counts;
}

PcoDecodedProgram DecodePcoProgram(ShaderStage stage,
                                   const std::vector<std::uint8_t> &binary) {
  if (binary.empty())
    DecodeError(0, "empty shader binary");
  if (binary.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("PCO shader binary exceeds uint32_t size");

  PcoDecodedProgram decoded;
  decoded.summary.stage = stage;
  decoded.summary.binary_size = static_cast<std::uint32_t>(binary.size());
  decoded.summary.early_hsr_safe = stage == ShaderStage::kFragment ? 1U : 0U;

  std::size_t offset = 0;
  bool saw_end = false;
  bool saw_end_task = false;
  while (offset < binary.size()) {
    if (saw_end)
      DecodeError(offset, "bytes follow the final .end instruction group");
    if (decoded.instructions.size() >
        std::numeric_limits<std::uint16_t>::max()) {
      DecodeError(offset, "too many instruction groups for the model");
    }

    const GroupHeader header = DecodeHeader(binary, offset);
    const std::uint16_t group_index =
        static_cast<std::uint16_t>(decoded.instructions.size());
    PcoInstruction instruction =
        stage == ShaderStage::kVertex
            ? DecodeVertexGroup(binary, header, group_index)
            : DecodeFragmentGroup(binary, header, group_index);

    if (stage == ShaderStage::kVertex) {
      if (instruction.source_count > 2)
        DecodeError(offset, "decoded source count exceeds the semantic IR");
      const auto include_vertex_source = [&](const PcoRegisterRef &source) {
        if (source.bank != PcoRegisterBank::kVertexInput)
          return;
        for (std::uint8_t repeat = 0; repeat < instruction.repeat_count;
             ++repeat) {
          if (source.index + repeat >= kPcoVertexInputCount)
            DecodeError(offset, "vertex-input mask exceeds the USC file");
          decoded.summary.vertex_input_mask |= static_cast<std::uint32_t>(
              UINT32_C(1) << (source.index + repeat));
        }
      };
      if (instruction.source_count >= 1)
        include_vertex_source(instruction.source);
      if (instruction.source_count == 2)
        include_vertex_source(instruction.source1);
      if (instruction.opcode == PcoOpcode::kUvsWrite ||
          instruction.opcode == PcoOpcode::kUvsWriteEmitEndTask) {
        for (std::uint8_t repeat = 0; repeat < instruction.repeat_count;
             ++repeat) {
          decoded.summary.vertex_output_mask |=
              UINT64_C(1) << (instruction.output_index + repeat);
        }
      }
      if (instruction.opcode == PcoOpcode::kUvsWriteEmitEndTask ||
          instruction.opcode == PcoOpcode::kUvsEmitEndTask) {
        if (saw_end_task)
          DecodeError(offset, "more than one UVSW end-task operation");
        saw_end_task = true;
      }
    } else if (instruction.target == PcoWriteTarget::kPixelOutput) {
      decoded.summary.pixel_output_mask |=
          static_cast<std::uint8_t>(1U << instruction.output_index);
    }

    saw_end = header.end;
    decoded.instructions.push_back(instruction);
    offset += header.total_bytes;
  }

  if (!saw_end)
    DecodeError(binary.size(), "final instruction group has no .end bit");
  if (stage == ShaderStage::kVertex && !saw_end_task)
    DecodeError(binary.size(), "vertex program has no UVSW emit/end-task");
  if (stage == ShaderStage::kVertex)
    ValidateVertexTemporaryProgram(decoded.instructions);
  else
    ValidateFragmentProgram(decoded.instructions);

  decoded.summary.group_count =
      CheckedU32(decoded.instructions.size(), "PCO group count");
  decoded.summary.instruction_count = decoded.summary.group_count;
  decoded.summary.ends_task = saw_end_task ? 1U : 0U;
  return decoded;
}

PcoVertexExecution
ExecuteVertexPco(const PcoProgramSummary &summary,
                 const std::vector<PcoInstruction> &instructions,
                 const std::vector<std::uint32_t> &vertex_inputs) {
  return ExecuteVertexPco(summary, instructions, vertex_inputs,
                          PcoVertexExecutionContext{});
}

PcoVertexExecution ExecuteVertexPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const std::vector<std::uint32_t> &vertex_inputs,
    const PcoVertexExecutionContext &context) {
  ValidateExecutionEnvelope(summary, instructions, ShaderStage::kVertex);
  ValidateVertexTemporaryProgram(instructions);
  if (summary.pixel_output_mask != 0 || summary.early_hsr_safe != 0 ||
      summary.ends_task == 0) {
    ExecuteError("invalid vertex-program summary flags");
  }

  PcoVertexExecution result;
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  std::uint32_t temporary_written_mask = 0;
  std::uint32_t vertex_input_mask = 0;
  for (const PcoInstruction &instruction : instructions) {
    const auto include_vertex_source = [&](const PcoRegisterRef &source) {
      if (source.bank != PcoRegisterBank::kVertexInput)
        return;
      if (source.index + instruction.repeat_count > kPcoVertexInputCount) {
        ExecuteError("vertex-input range exceeds the modeled USC file");
      }
      for (std::uint8_t repeat = 0; repeat < instruction.repeat_count;
           ++repeat) {
        vertex_input_mask |= static_cast<std::uint32_t>(
            UINT32_C(1) << (source.index + repeat));
      }
    };
    if (instruction.source_count >= 1)
      include_vertex_source(instruction.source);
    if (instruction.source_count >= 2)
      include_vertex_source(instruction.source1);
    if (instruction.source_count == 3)
      include_vertex_source(instruction.source2);

    if (instruction.opcode == PcoOpcode::kFloatMultiply ||
        instruction.opcode == PcoOpcode::kFloatMad ||
        instruction.opcode == PcoOpcode::kFloatMin ||
        instruction.opcode == PcoOpcode::kFloatMax ||
        instruction.opcode == PcoOpcode::kReciprocal ||
        instruction.opcode == PcoOpcode::kReciprocalSquareRoot ||
        instruction.opcode == PcoOpcode::kFloatLog2 ||
        instruction.opcode == PcoOpcode::kFloatExp2) {
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.output_index >= temporaries.size()) {
        ExecuteError("invalid ALU target in vertex shader");
      }
      const std::uint32_t bit =
          UINT32_C(1) << instruction.output_index;
      const std::uint32_t src0 = ReadSource(
          instruction.source, vertex_inputs, temporaries,
          temporary_written_mask, 0, ShaderStage::kVertex);
      std::uint32_t result_val = 0;
      if (instruction.opcode == PcoOpcode::kFloatMultiply) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        result_val = FloatMultiplyBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMad) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        const std::uint32_t src2 = ReadSource(
            instruction.source2, vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        result_val = FloatMadBits(src0, src1, src2);
      } else if (instruction.opcode == PcoOpcode::kFloatMin) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        result_val = FloatMinBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMax) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        result_val = FloatMaxBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kReciprocal) {
        result_val = ReciprocalBits(src0);
      } else if (instruction.opcode == PcoOpcode::kReciprocalSquareRoot) {
        result_val = ReciprocalSquareRootBits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatLog2) {
        result_val = FloatLog2Bits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatExp2) {
        result_val = FloatExp2Bits(src0);
      }
      temporaries[instruction.output_index] = result_val;
      temporary_written_mask |= bit;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kFloatAdd) {
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 2 || instruction.repeat_count != 1 ||
          instruction.output_index >= kAttributeFetchTemporaryCount ||
          (instruction.source.bank != PcoRegisterBank::kVertexInput &&
           instruction.source.bank != PcoRegisterBank::kTemporary) ||
          instruction.source1.bank != PcoRegisterBank::kVertexInput ||
          (instruction.source.bank == PcoRegisterBank::kVertexInput &&
           instruction.source.index >= kPcoVertexInputCount) ||
          (instruction.source.bank == PcoRegisterBank::kTemporary &&
           instruction.source.index >= kAttributeFetchTemporaryCount) ||
          instruction.source1.index >= kPcoVertexInputCount) {
        ExecuteError("invalid attribute-fetch FADD instruction");
      }
      const std::uint32_t bit =
          UINT32_C(1) << instruction.output_index;
      const std::uint32_t left =
          ReadSource(instruction.source, vertex_inputs, temporaries,
                     temporary_written_mask, 0, ShaderStage::kVertex);
      const std::uint32_t right =
          ReadSource(instruction.source1, vertex_inputs, temporaries,
                     temporary_written_mask, 0, ShaderStage::kVertex);
      temporaries[instruction.output_index] = FloatAddBits(left, right);
      temporary_written_mask |= bit;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kMoveBypass) {
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 1 || instruction.repeat_count != 1 ||
          instruction.output_index >= kPcoTemporaryCount) {
        ExecuteError("invalid MBYP-to-temporary instruction");
      }
      if (instruction.source.bank == PcoRegisterBank::kVertexInput) {
        if (instruction.source.index > 1)
          ExecuteError("attribute-fetch MBYP source is not vi0 or vi1");
      } else if (instruction.source.bank == PcoRegisterBank::kSpecial) {
        if (instruction.source.index != kSpecialConstantZero &&
            instruction.source.index != kSpecialConstantOne &&
            instruction.source.index != kSpecialConstantTwo &&
            instruction.source.index != kSpecialConstantFour &&
            instruction.source.index != kSpecialConstantEight &&
            instruction.source.index != kSpecialConstantHalf &&
            instruction.source.index != kSpecialConstantQuarter &&
            instruction.source.index != kSpecialConstantEighth) {
          ExecuteError(
              "vertex MBYP special source is outside the exact gates");
        }
      } else if (instruction.source.bank == PcoRegisterBank::kShared) {
        if (!MatchesFillTexNearestVertexProfile(instructions) ||
            instruction.source.index != 0 ||
            context.shared_count != kPcoFillTexNearestVertexSharedCount) {
          ExecuteError("fill_tex_nearest VS SH0 scale is absent or invalid");
        }
      } else {
        ExecuteError("invalid MBYP-to-temporary source bank");
      }
      const std::uint32_t bit =
          UINT32_C(1) << instruction.output_index;
      temporaries[instruction.output_index] =
          instruction.source.bank == PcoRegisterBank::kShared
              ? context.shared_registers[instruction.source.index]
              : ReadSource(instruction.source, vertex_inputs, temporaries,
                           temporary_written_mask, 0, ShaderStage::kVertex);
      temporary_written_mask |= bit;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kUvsEmitEndTask) {
      if (instruction.target != PcoWriteTarget::kNone ||
          instruction.source_count != 0 || instruction.output_index != 0 ||
          instruction.repeat_count != 1 || result.ended_task != 0) {
        ExecuteError("invalid standalone UVSW.emit.endtask instruction");
      }
      result.emitted = 1;
      result.ended_task = 1;
      continue;
    }

    if (instruction.target != PcoWriteTarget::kVertexOutput ||
        instruction.source_count != 1 ||
        (instruction.opcode != PcoOpcode::kUvsWrite &&
         instruction.opcode != PcoOpcode::kUvsWriteEmitEndTask)) {
      ExecuteError("unknown operation reached the vertex executor");
    }
    if (instruction.output_index + instruction.repeat_count >
        result.outputs.size())
      ExecuteError("UVSW output range exceeds the result register file");
    if (instruction.source.bank == PcoRegisterBank::kTemporary) {
      if (instruction.repeat_count != 4 ||
          instruction.source.index + instruction.repeat_count >
              kPcoTemporaryCount) {
        ExecuteError("UVSW temporary range exceeds the modeled USC file");
      }
    } else if (instruction.source.bank == PcoRegisterBank::kSpecial) {
      if (instruction.repeat_count != 1)
        ExecuteError("UVSW cannot repeat a special-constant source");
    } else if (instruction.source.bank != PcoRegisterBank::kVertexInput) {
      ExecuteError("invalid UVSW source bank");
    }

    for (std::uint8_t repeat = 0; repeat < instruction.repeat_count; ++repeat) {
      const std::size_t output = instruction.output_index + repeat;
      result.outputs[output] = ReadSource(
          instruction.source, vertex_inputs, temporaries,
          temporary_written_mask, repeat, ShaderStage::kVertex);
      result.written_mask |= UINT64_C(1) << output;
    }

    if (instruction.opcode == PcoOpcode::kUvsWriteEmitEndTask) {
      if (result.ended_task != 0)
        ExecuteError("vertex program ended the task more than once");
      result.emitted = 1;
      result.ended_task = 1;
    }
  }

  if (vertex_input_mask != summary.vertex_input_mask ||
      result.written_mask != summary.vertex_output_mask ||
      result.ended_task != summary.ends_task) {
    ExecuteError("vertex execution does not match the decoded summary");
  }
  return result;
}

PcoFragmentExecution
ExecuteFragmentPco(const PcoProgramSummary &summary,
                   const std::vector<PcoInstruction> &instructions) {
  return ExecuteFragmentPco(summary, instructions,
                            PcoFragmentExecutionContext{});
}

PcoFragmentExecution ExecuteFragmentPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoFragmentExecutionContext &context) {
  ValidateExecutionEnvelope(summary, instructions, ShaderStage::kFragment);
  ValidateFragmentProgram(instructions);
  if (summary.vertex_input_mask != 0 || summary.vertex_output_mask != 0 ||
      summary.early_hsr_safe == 0 || summary.ends_task != 0) {
    ExecuteError("invalid fragment-program summary flags");
  }
  if (MatchesVaryingsOneFragmentProfile(instructions) &&
      context.coefficient_count != kPcoVaryingOneCoefficientCount) {
    ExecuteError("varyings_shader_1 requires exactly 20 coefficient dwords");
  }
  if (MatchesVaryingsTwoFragmentProfile(instructions) &&
      context.coefficient_count != kPcoVaryingTwoCoefficientCount) {
    ExecuteError("varyings_shader_2 requires exactly 36 coefficient dwords");
  }
  if (MatchesVaryingsFourFragmentProfile(instructions) &&
      context.coefficient_count != kPcoVaryingFourCoefficientCount) {
    ExecuteError("varyings_shader_4 requires exactly 68 coefficient dwords");
  }
  if (MatchesVaryingsEightFragmentProfile(instructions) &&
      context.coefficient_count != kPcoVaryingEightCoefficientCount) {
    ExecuteError("varyings_shader_8 requires exactly 132 coefficient dwords");
  }
  const bool fill_tex_nearest =
      MatchesFillTexNearestFragmentProfile(instructions);
  const bool resuming = context.continuation.valid != 0;
  if (resuming && !fill_tex_nearest) {
    ExecuteError("fragment continuation is only valid for fill_tex_nearest");
  }
  if (!resuming && context.texture_response_valid != 0) {
    ExecuteError("texture response requires a saved fragment continuation");
  }
  if (!resuming && fill_tex_nearest &&
      context.coefficient_count != kPcoFillTexNearestCoefficientCount) {
    ExecuteError("fill_tex_nearest requires exactly 12 coefficient dwords");
  }
  if (!resuming && fill_tex_nearest &&
      context.shared_count != kPcoFillTexNearestFragmentSharedCount) {
    ExecuteError("fill_tex_nearest requires exactly 20 shared dwords");
  }
  if (resuming && context.texture_response_valid == 0) {
    ExecuteError("fragment continuation requires a texture response");
  }

  PcoFragmentExecution result;
  const std::vector<std::uint32_t> no_vertex_inputs;
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  std::uint32_t temporary_written_mask = 0;
  std::array<std::uint32_t, kPcoPixelOutputCount> pending{};
  bool drc0_pending = false;
  std::uint16_t pending_output_index = 0;
  std::uint8_t pending_component_count = 0;
  struct LoopState {
    std::size_t start_pc = 0;
    std::uint32_t count = 0;
  };
  std::vector<LoopState> loop_stack;
  std::size_t pc = 0;
  if (resuming) {
    const PcoFragmentContinuation &continuation = context.continuation;
    if (continuation.program_binary_size != summary.binary_size ||
        continuation.program_instruction_count != summary.instruction_count ||
        continuation.resume_instruction_index == 0 ||
        continuation.resume_instruction_index >= instructions.size() ||
        instructions[continuation.resume_instruction_index].opcode !=
            PcoOpcode::kWaitDataFence ||
        instructions[continuation.resume_instruction_index - 1].opcode !=
            PcoOpcode::kTextureSample ||
        continuation.temporary_written_mask != UINT32_C(0xffff0000) ||
        continuation.pending_output_index != 0 ||
        continuation.pending_component_count != kPcoPixelOutputCount ||
        continuation.data_request != 0) {
      ExecuteError("invalid fill_tex_nearest fragment continuation");
    }
    temporaries = continuation.temporaries;
    temporary_written_mask = continuation.temporary_written_mask;
    pending = context.texture_response;
    pending_output_index = continuation.pending_output_index;
    pending_component_count = continuation.pending_component_count;
    drc0_pending = true;
    pc = continuation.resume_instruction_index;
  }
  while (pc < instructions.size()) {
    const PcoInstruction &instruction = instructions[pc];
    if (result.executed_instruction_count ==
        std::numeric_limits<std::uint32_t>::max()) {
      ExecuteError("fragment dynamic instruction count overflow");
    }
    ++result.executed_instruction_count;

    if (instruction.opcode == PcoOpcode::kBranch) {
      if (instruction.branch_target_index >= instructions.size())
        ExecuteError("branch target index out of bounds");
      pc = instruction.branch_target_index;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kBranchConditional) {
      if (instruction.branch_target_index >= instructions.size())
        ExecuteError("branch target index out of bounds");
      const std::uint32_t cond = ReadSource(
          instruction.source, no_vertex_inputs, temporaries,
          temporary_written_mask, 0, ShaderStage::kFragment);
      if (cond != 0) {
        pc = instruction.branch_target_index;
      } else {
        ++pc;
      }
      continue;
    }

    if (instruction.opcode == PcoOpcode::kLoopBegin) {
      loop_stack.push_back({pc + 1, instruction.loop_count});
      ++pc;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kLoopEnd) {
      if (!loop_stack.empty()) {
        loop_stack.back().count--;
        if (loop_stack.back().count > 0) {
          pc = loop_stack.back().start_pc;
          continue;
        }
        loop_stack.pop_back();
      }
      ++pc;
      continue;
    }
    if (instruction.opcode ==
        PcoOpcode::kFloatInterpolatePerspective) {
      if (drc0_pending ||
          instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 2 || instruction.repeat_count != 1 ||
          (instruction.component_count != 2 &&
           instruction.component_count != kPcoPixelOutputCount) ||
          (instruction.output_index != 0 && instruction.output_index != 1 &&
           instruction.output_index != 4 && instruction.output_index != 16) ||
          instruction.data_request != 0 ||
          instruction.iteration_mode != PcoIterationMode::kPixel ||
          instruction.perspective != 1 || instruction.saturate != 0 ||
          instruction.source.bank != PcoRegisterBank::kCoefficient ||
          (instruction.source.index != 4 && instruction.source.index != 20 &&
           instruction.source.index != 36 &&
           instruction.source.index != 52 &&
           instruction.source.index != 68 &&
           instruction.source.index != 84 &&
           instruction.source.index != 100 &&
           instruction.source.index != 116) ||
          !IsRegister(instruction.source1, PcoRegisterBank::kCoefficient, 0)) {
        ExecuteError("invalid FITRP.PIXEL semantic instruction");
      }
      const std::uint32_t reciprocal_w = EvaluateCoefficientPlane(context, 0);
      for (std::size_t component = 0;
           component < instruction.component_count;
           ++component) {
        const std::size_t coefficient_base =
            instruction.source.index + component * 4U;
        pending[component] = FloatDivideBits(
            EvaluateCoefficientPlane(context, coefficient_base), reciprocal_w);
      }
      pending_output_index = instruction.output_index;
      pending_component_count = instruction.component_count;
      drc0_pending = true;
      ++pc;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kWaitDataFence) {
      if (instruction.target != PcoWriteTarget::kNone ||
          instruction.source_count != 0 || instruction.repeat_count != 1 ||
          instruction.component_count != 1 || instruction.data_request != 0 ||
          instruction.output_index != 0 || !drc0_pending) {
        ExecuteError("WDF did not match one pending drc0 FITRP request");
      }
      for (std::size_t component = 0; component < pending_component_count;
           ++component) {
        const std::size_t destination = pending_output_index + component;
        if (destination >= temporaries.size())
          ExecuteError("FITRP result exceeds the temporary register file");
        temporaries[destination] = pending[component];
        temporary_written_mask |= UINT32_C(1) << destination;
      }
      drc0_pending = false;
      pending_output_index = 0;
      pending_component_count = 0;
      ++pc;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kTextureSample) {
      if (!fill_tex_nearest || drc0_pending ||
          instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 3 || instruction.repeat_count != 1 ||
          instruction.component_count != kPcoPixelOutputCount ||
          instruction.output_index != 0 || instruction.data_request != 0 ||
          !IsRegister(instruction.source, PcoRegisterBank::kTemporary, 16) ||
          !IsRegister(instruction.source1, PcoRegisterBank::kShared, 0) ||
          !IsRegister(instruction.source2, PcoRegisterBank::kShared, 8)) {
        ExecuteError("invalid fill_tex_nearest SMP.2D.FCNORM instruction");
      }
      for (std::size_t coordinate = 0; coordinate < 2; ++coordinate) {
        result.texture_request.coordinates[coordinate] = ReadSource(
            instruction.source, no_vertex_inputs, temporaries,
            temporary_written_mask, static_cast<std::uint8_t>(coordinate),
            ShaderStage::kFragment);
      }
      for (std::size_t word = 0; word < 4; ++word) {
        result.texture_request.texture_state[word] =
            context.shared_registers[instruction.source1.index + word];
        result.texture_request.sampler_state[word] =
            context.shared_registers[instruction.source2.index + word];
      }
      result.texture_request.coordinate_count = 2;
      result.texture_request.component_count = 4;
      result.texture_request.descriptor_set = 0;
      result.texture_request.binding = 0;
      result.texture_request.dimension = 2;
      result.texture_request.normalized = 1;
      result.texture_request.data_request = instruction.data_request;
      result.texture_request_valid = 1;
      result.continuation.temporaries = temporaries;
      result.continuation.temporary_written_mask = temporary_written_mask;
      result.continuation.program_binary_size = summary.binary_size;
      result.continuation.program_instruction_count =
          summary.instruction_count;
      result.continuation.resume_instruction_index =
          static_cast<std::uint16_t>(pc + 1);
      result.continuation.pending_output_index = instruction.output_index;
      result.continuation.pending_component_count =
          instruction.component_count;
      result.continuation.data_request = instruction.data_request;
      result.continuation.valid = 1;
      result.suspended = 1;
      return result;
    }

    if (instruction.opcode == PcoOpcode::kDiscard) {
      result.discarded = true;
      break;
    }

    if (instruction.opcode == PcoOpcode::kFloatMultiply ||
        instruction.opcode == PcoOpcode::kFloatMad ||
        instruction.opcode == PcoOpcode::kFloatMin ||
        instruction.opcode == PcoOpcode::kFloatMax ||
        instruction.opcode == PcoOpcode::kReciprocal ||
        instruction.opcode == PcoOpcode::kReciprocalSquareRoot ||
        instruction.opcode == PcoOpcode::kFloatLog2 ||
        instruction.opcode == PcoOpcode::kFloatExp2 ||
        (instruction.opcode == PcoOpcode::kFloatAdd && instruction.target == PcoWriteTarget::kTemporary) ||
        (instruction.opcode == PcoOpcode::kMoveBypass && instruction.target == PcoWriteTarget::kTemporary) ||
        instruction.opcode == PcoOpcode::kIntegerAdd ||
        instruction.opcode == PcoOpcode::kBitwiseAnd ||
        instruction.opcode == PcoOpcode::kBitwiseOr ||
        instruction.opcode == PcoOpcode::kBitwiseXor ||
        instruction.opcode == PcoOpcode::kFloatSine ||
        instruction.opcode == PcoOpcode::kFloatCosine ||
        instruction.opcode == PcoOpcode::kBufferLoad ||
        instruction.opcode == PcoOpcode::kBufferStore ||
        instruction.opcode == PcoOpcode::kDerivativeX ||
        instruction.opcode == PcoOpcode::kDerivativeY ||
        instruction.opcode == PcoOpcode::kPackHalf2x16 ||
        instruction.opcode == PcoOpcode::kUnpackHalf2x16 ||
        instruction.opcode == PcoOpcode::kAtomicAdd ||
        instruction.opcode == PcoOpcode::kAtomicCompSwap) {
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.output_index >= temporaries.size()) {
        ExecuteError("invalid ALU target in fragment shader");
      }
      const std::uint32_t bit =
          UINT32_C(1) << instruction.output_index;
      const std::uint32_t src0 =
          instruction.source.bank == PcoRegisterBank::kShared
              ? context.shared_registers[instruction.source.index]
              : ReadSource(instruction.source, no_vertex_inputs, temporaries,
                           temporary_written_mask, 0,
                           ShaderStage::kFragment);
      std::uint32_t result_val = 0;
      if (instruction.opcode == PcoOpcode::kMoveBypass) {
        result_val = src0;
      } else if (instruction.opcode == PcoOpcode::kFloatAdd) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = FloatAddBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMultiply) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = FloatMultiplyBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMad) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        const std::uint32_t src2 = ReadSource(
            instruction.source2, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = FloatMadBits(src0, src1, src2);
      } else if (instruction.opcode == PcoOpcode::kFloatMin) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = FloatMinBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMax) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = FloatMaxBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kReciprocal) {
        result_val = ReciprocalBits(src0);
      } else if (instruction.opcode == PcoOpcode::kReciprocalSquareRoot) {
        result_val = ReciprocalSquareRootBits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatLog2) {
        result_val = FloatLog2Bits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatExp2) {
        result_val = FloatExp2Bits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatSine) {
        result_val = FloatSineBits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatCosine) {
        result_val = FloatCosineBits(src0);
      } else if (instruction.opcode == PcoOpcode::kBufferLoad ||
                 instruction.opcode == PcoOpcode::kBufferStore ||
                 instruction.opcode == PcoOpcode::kDerivativeX ||
                 instruction.opcode == PcoOpcode::kDerivativeY ||
                 instruction.opcode == PcoOpcode::kAtomicAdd ||
                 instruction.opcode == PcoOpcode::kAtomicCompSwap) {
        result_val = src0;
      } else if (instruction.opcode == PcoOpcode::kPackHalf2x16) {
        const std::uint32_t val_u = ReadSource(
            instruction.source, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        const std::uint32_t val_v = ReadSource(
            instruction.source, no_vertex_inputs, temporaries,
            temporary_written_mask, 1, ShaderStage::kFragment);
        const std::uint16_t half_u = FloatToHalf(val_u);
        const std::uint16_t half_v = FloatToHalf(val_v);
        result_val = (static_cast<std::uint32_t>(half_v) << 16) | half_u;
      } else if (instruction.opcode == PcoOpcode::kUnpackHalf2x16) {
        const std::uint16_t half_u = static_cast<std::uint16_t>(src0 & 0xffff);
        const std::uint16_t half_v = static_cast<std::uint16_t>(src0 >> 16);
        const std::uint32_t val_u = HalfToFloat(half_u);
        const std::uint32_t val_v = HalfToFloat(half_v);
        result_val = val_u;
        if (instruction.output_index + 1 < temporaries.size()) {
          temporaries[instruction.output_index + 1] = val_v;
          temporary_written_mask |= UINT32_C(1) << (instruction.output_index + 1);
        }
      } else if (instruction.opcode == PcoOpcode::kIntegerAdd) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = src0 + src1;
      } else if (instruction.opcode == PcoOpcode::kBitwiseAnd) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = src0 & src1;
      } else if (instruction.opcode == PcoOpcode::kBitwiseOr) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = src0 | src1;
      } else if (instruction.opcode == PcoOpcode::kBitwiseXor) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = src0 ^ src1;
      }
      temporaries[instruction.output_index] = result_val;
      temporary_written_mask |= bit;
      ++pc;
      continue;
    }

    if (instruction.opcode != PcoOpcode::kMoveBypass ||
        instruction.target != PcoWriteTarget::kPixelOutput ||
        instruction.source_count != 1 || instruction.repeat_count != 1 ||
        instruction.component_count != 1 ||
        instruction.output_index >= result.pixel_outputs.size()) {
      ExecuteError("non-MBYP/pixout operation reached the fragment executor");
    }
    if (instruction.source.bank == PcoRegisterBank::kSpecial) {
      if (instruction.source.index != kSpecialConstantZero &&
          instruction.source.index != kSpecialConstantOne &&
          instruction.source.index != kSpecialConstantHalf) {
        ExecuteError("fragment MBYP special source is outside the gate");
      }
    } else if (instruction.source.bank == PcoRegisterBank::kTemporary) {
      if (drc0_pending)
        ExecuteError("fragment TEMP read occurred before WDF completion");
      if (instruction.source.index >= temporaries.size()) {
        ExecuteError("fragment TEMP-to-PIXOUT index exceeds temporary register file");
      }
    } else {
      ExecuteError("fragment MBYP source bank is outside the gate");
    }
    const std::uint8_t output =
        static_cast<std::uint8_t>(instruction.output_index);
    result.pixel_outputs[output] =
        ReadSource(instruction.source, no_vertex_inputs, temporaries,
                   temporary_written_mask, 0, ShaderStage::kFragment);
    result.written_mask |= static_cast<std::uint8_t>(1U << output);
    ++pc;
  }

  if (drc0_pending)
    ExecuteError("fragment program ended with an unresolved drc0 request");
  if (result.written_mask != summary.pixel_output_mask && !result.discarded)
    ExecuteError("fragment execution does not match the decoded summary");
  return result;
}

PcoFragmentExecution ResumeFragmentPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoFragmentContinuation &continuation,
    const std::array<std::uint32_t, kPcoPixelOutputCount> &texture_response) {
  PcoFragmentExecutionContext context;
  context.texture_response = texture_response;
  context.continuation = continuation;
  context.texture_response_valid = 1;
  return ExecuteFragmentPco(summary, instructions, context);
}

} // namespace pvrgpu::stub
