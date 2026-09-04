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
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
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

/* GLBench conditionals, emitted by the pinned public Mesa PCO backend from
 * tools/pco-fixtures/generate_conditionals.c.  The VS reads the complete
 * mat4 from SH0..15. SHA-256:
 * b6b8b0e060c78b1b77084b714d93b3a3f33742b53bcdef5c179532c74bd13b88.
 */
const std::vector<std::uint8_t> kConditionalsVertexBinary = {
    0x35, 0x82, 0x00, 0x01, 0x80, 0xc0, 0x10, 0x00, 0x00, 0x40, 0x35, 0x82,
    0x00, 0x08, 0xc0, 0xc0, 0x08, 0x00, 0x00, 0x40, 0x35, 0x82, 0x00, 0x40,
    0x81, 0xe0, 0x04, 0x00, 0x00, 0x41, 0x35, 0x82, 0x00, 0x01, 0xc1, 0x80,
    0x00, 0x00, 0x42, 0xff, 0x35, 0x82, 0x00, 0x08, 0xc2, 0xa1, 0x00, 0x00,
    0x41, 0xff, 0x86, 0x92, 0x40, 0x13, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
    0x42, 0xff, 0x35, 0x82, 0x00, 0x40, 0xc2, 0xa0, 0x00, 0x00, 0x42, 0xff,
    0x35, 0x82, 0x00, 0x01, 0xc2, 0x80, 0x00, 0x00, 0x43, 0xff, 0x35, 0x82,
    0x00, 0x08, 0xc3, 0xa2, 0x00, 0x00, 0x42, 0xff, 0x8a, 0xd2, 0x00, 0xd3,
    0x3c, 0xec, 0x9c, 0x1e, 0x87, 0x87, 0xc0, 0xcf, 0x80, 0x11, 0x8b, 0x01,
    0x00, 0x20, 0x43, 0xff, 0x78, 0xd2, 0x00, 0xd1, 0x3c, 0xf0, 0xb0, 0x87,
    0x87, 0xc1, 0xe3, 0x80, 0x10, 0x42, 0x01, 0x41, 0x86, 0x92, 0x40, 0x13,
    0xcd, 0xcc, 0xcc, 0x3d, 0x00, 0x00, 0x42, 0xff, 0x35, 0x82, 0x00, 0x40,
    0xc2, 0xa1, 0x00, 0x00, 0x41, 0xff, 0x36, 0x82, 0x00, 0xc0, 0xc1, 0x60,
    0x00, 0x81, 0x00, 0x00, 0x40, 0xff, 0x35, 0x82, 0x00, 0x87, 0x80, 0x08,
    0x00, 0x00, 0x00, 0x41, 0x35, 0x82, 0x00, 0x87, 0x81, 0x08, 0x00, 0x00,
    0x00, 0x42, 0x35, 0x82, 0x00, 0x87, 0x82, 0x08, 0x00, 0x00, 0x00, 0x43,
    0x35, 0x82, 0x00, 0x87, 0x83, 0x08, 0x00, 0x00, 0x00, 0x44, 0x35, 0x82,
    0x00, 0x87, 0x84, 0x08, 0x00, 0x00, 0x00, 0x45, 0x35, 0x82, 0x00, 0x87,
    0x85, 0x08, 0x00, 0x00, 0x00, 0x46, 0x35, 0x82, 0x00, 0x87, 0x86, 0x08,
    0x00, 0x00, 0x00, 0x47, 0x35, 0x82, 0x00, 0x87, 0x87, 0x08, 0x00, 0x00,
    0x00, 0x48, 0x35, 0x82, 0x00, 0x40, 0xc5, 0xa0, 0x00, 0x00, 0x45, 0xff,
    0x35, 0x82, 0x00, 0x40, 0xc6, 0xa0, 0x00, 0x00, 0x46, 0xff, 0x35, 0x82,
    0x00, 0x40, 0xc7, 0xa0, 0x00, 0x00, 0x47, 0xff, 0x35, 0x82, 0x00, 0x40,
    0xc8, 0xa0, 0x00, 0x00, 0x40, 0xff, 0x36, 0x82, 0x00, 0xc0, 0xc1, 0x40,
    0x08, 0x45, 0x00, 0x00, 0x41, 0xff, 0x36, 0x82, 0x00, 0xc0, 0xc2, 0x40,
    0x08, 0x46, 0x00, 0x00, 0x42, 0xff, 0x36, 0x82, 0x00, 0xc0, 0xc3, 0x40,
    0x08, 0x47, 0x00, 0x00, 0x43, 0xff, 0x36, 0x82, 0x00, 0xc0, 0xc4, 0x40,
    0x08, 0x40, 0x00, 0x00, 0x40, 0xff, 0x35, 0x82, 0x00, 0x87, 0x88, 0x08,
    0x00, 0x00, 0x00, 0x44, 0x35, 0x82, 0x00, 0x87, 0x89, 0x08, 0x00, 0x00,
    0x00, 0x45, 0x35, 0x82, 0x00, 0x87, 0x8a, 0x08, 0x00, 0x00, 0x00, 0x46,
    0x35, 0x82, 0x00, 0x87, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x47, 0x36, 0x82,
    0x00, 0xc0, 0xc4, 0x42, 0x08, 0x41, 0x00, 0x00, 0x41, 0xff, 0x36, 0x82,
    0x00, 0xc0, 0xc5, 0x42, 0x08, 0x42, 0x00, 0x00, 0x44, 0xff, 0x36, 0x82,
    0x00, 0xc0, 0xc6, 0x42, 0x08, 0x43, 0x00, 0x00, 0x45, 0xff, 0x36, 0x82,
    0x00, 0xc0, 0xc7, 0x42, 0x08, 0x40, 0x00, 0x00, 0x46, 0xff, 0x35, 0x82,
    0x00, 0x87, 0x8c, 0x08, 0x00, 0x00, 0x00, 0x40, 0x35, 0x82, 0x00, 0x87,
    0x8d, 0x08, 0x00, 0x00, 0x00, 0x47, 0x35, 0x82, 0x00, 0x87, 0x8e, 0x08,
    0x00, 0x00, 0x00, 0x48, 0x35, 0x82, 0x00, 0x87, 0x8f, 0x08, 0x00, 0x00,
    0x00, 0x49, 0x35, 0x82, 0x00, 0x00, 0xc1, 0xa0, 0x00, 0x00, 0x40, 0xff,
    0x35, 0x82, 0x00, 0x00, 0xc4, 0xa7, 0x00, 0x00, 0x41, 0xff, 0x35, 0x82,
    0x00, 0x00, 0xc5, 0xa8, 0x00, 0x00, 0x42, 0xff, 0x35, 0x82, 0x00, 0x00,
    0xc6, 0xa9, 0x00, 0x00, 0x43, 0xff, 0x57, 0xa0, 0x06, 0x08, 0x00, 0xc0,
    0x00, 0x00, 0x00, 0x30, 0xf2, 0xff, 0xff, 0xff, 0x44, 0xa0, 0x80, 0x05,
    0x00, 0x00, 0x00, 0xff,
};

/* Same compiler/profile, FS reads the four-word gl_FbWposYTransform block at
 * SH0..3 and explicitly round-trips mediump values through binary16.
 * SHA-256:
 * 3887e069335b8394d2517eeaf071ee26de7a4070650dc9b038eb4e07eb324887.
 */
const std::vector<std::uint8_t> kConditionalsFragmentBinary = {
    0x57, 0x92, 0x00, 0x9c, 0x10, 0x80, 0x40, 0xa0, 0x00, 0x30, 0x00, 0x2c,
    0x40, 0xff, 0x55, 0xd2, 0x40, 0x01, 0x02, 0x80, 0xa0, 0x80, 0x81, 0x40,
    0x99, 0xc2, 0x00, 0xd3, 0x3c, 0xf0, 0xa0, 0x9c, 0x1e, 0x87, 0xc0, 0xcf,
    0x80, 0x11, 0x00, 0x20, 0x40, 0xff, 0x56, 0x92, 0x00, 0xd3, 0x3f, 0x80,
    0xc1, 0x83, 0x10, 0x00, 0x30, 0x41, 0x56, 0x92, 0x00, 0xd3, 0x3f, 0x80,
    0xc2, 0x83, 0x10, 0x00, 0x30, 0x42, 0x78, 0xd2, 0x00, 0xd0, 0x3c, 0xf2,
    0xb0, 0x87, 0x87, 0xc1, 0xe0, 0x80, 0x10, 0x42, 0x11, 0x40, 0x57, 0x92,
    0x00, 0x9c, 0x10, 0x80, 0x40, 0xa0, 0x00, 0x30, 0x00, 0x2c, 0x41, 0xff,
    0x55, 0xd2, 0x40, 0x01, 0x02, 0x80, 0xa1, 0x80, 0x81, 0x41, 0x99, 0xc2,
    0x00, 0xd3, 0x3c, 0xf0, 0xa0, 0x9c, 0x1e, 0x87, 0xc1, 0xcf, 0x80, 0x11,
    0x00, 0x20, 0x41, 0xff, 0x56, 0x92, 0x00, 0xd3, 0x3f, 0x80, 0xc4, 0x83,
    0x10, 0x00, 0x30, 0x42, 0x56, 0x92, 0x00, 0xd3, 0x3f, 0x80, 0xc5, 0x83,
    0x10, 0x00, 0x30, 0x43, 0x78, 0xd2, 0x00, 0xd0, 0x3c, 0xf2, 0xb0, 0x87,
    0x87, 0xc2, 0xe1, 0x80, 0x10, 0x43, 0x11, 0x41, 0x35, 0x82, 0x00, 0x87,
    0x80, 0x08, 0x00, 0x00, 0x00, 0x42, 0x35, 0x82, 0x00, 0x87, 0x81, 0x08,
    0x00, 0x00, 0x00, 0x43, 0x36, 0x82, 0x00, 0xc0, 0xc1, 0x62, 0x00, 0x43,
    0x00, 0x00, 0x41, 0xff, 0x86, 0x92, 0x40, 0x13, 0x17, 0xb7, 0xd1, 0x38,
    0x00, 0x00, 0x42, 0xff, 0x35, 0x82, 0x00, 0x40, 0xc0, 0xa2, 0x00, 0x00,
    0x40, 0xff, 0x35, 0x82, 0x00, 0x40, 0xc0, 0xa1, 0x00, 0x00, 0x40, 0xff,
    0x35, 0x82, 0x00, 0x01, 0xc0, 0x80, 0x00, 0x00, 0x41, 0xff, 0x35, 0x82,
    0x00, 0x08, 0xc1, 0xa0, 0x00, 0x00, 0x40, 0xff, 0x57, 0x92, 0x00, 0x9c,
    0x0e, 0x80, 0x40, 0xa0, 0x40, 0x10, 0x00, 0x2c, 0x40, 0xff, 0x35, 0x82,
    0x00, 0x9c, 0x0e, 0x40, 0x00, 0x00, 0x40, 0xff, 0x35, 0x82, 0x00, 0x40,
    0xc0, 0xc1, 0x02, 0x00, 0x00, 0x41, 0x57, 0x92, 0x00, 0x9c, 0x0e, 0x80,
    0x40, 0xa0, 0x41, 0x10, 0x00, 0x2c, 0x41, 0xff, 0x35, 0x82, 0x00, 0x9c,
    0x0e, 0x41, 0x00, 0x00, 0x41, 0xff, 0x35, 0x82, 0x00, 0x01, 0xc1, 0x80,
    0x00, 0x00, 0x42, 0xff, 0x35, 0x82, 0x00, 0x08, 0xc2, 0xa1, 0x00, 0x00,
    0x41, 0xff, 0x57, 0x92, 0x00, 0x9c, 0x0e, 0x80, 0x40, 0xa0, 0x41, 0x10,
    0x00, 0x2c, 0x41, 0xff, 0x86, 0x92, 0x40, 0x13, 0x00, 0x00, 0x40, 0x40,
    0x00, 0x00, 0x42, 0xff, 0x35, 0x82, 0x00, 0x40, 0xc0, 0xa2, 0x00, 0x00,
    0x42, 0xff, 0x57, 0x92, 0x00, 0x9c, 0x0e, 0x80, 0x40, 0xa0, 0x42, 0x10,
    0x00, 0x2c, 0x42, 0xff, 0x35, 0x82, 0x00, 0x9c, 0x0e, 0x42, 0x00, 0x00,
    0x42, 0xff, 0x35, 0x82, 0x00, 0x01, 0xc2, 0x80, 0x00, 0x00, 0x43, 0xff,
    0x35, 0x82, 0x00, 0x08, 0xc3, 0xa2, 0x00, 0x00, 0x42, 0xff, 0x57, 0x92,
    0x00, 0x9c, 0x0e, 0x80, 0x40, 0xa0, 0x42, 0x10, 0x00, 0x2c, 0x42, 0xff,
    0x35, 0x82, 0x00, 0x9c, 0x0e, 0x41, 0x00, 0x00, 0x41, 0xff, 0x35, 0x82,
    0x00, 0x9c, 0x0e, 0x42, 0x00, 0x00, 0x42, 0xff, 0x8a, 0xd2, 0x00, 0xd3,
    0x3c, 0xec, 0x9c, 0x1e, 0x87, 0x87, 0xc0, 0xcf, 0x80, 0x11, 0x8b, 0x01,
    0x00, 0x20, 0x40, 0xff, 0x78, 0xd2, 0x00, 0xd1, 0x3c, 0xf0, 0xb0, 0x87,
    0x87, 0xc1, 0xe0, 0x80, 0x10, 0x42, 0x01, 0x40, 0x34, 0x8a, 0x00, 0x87,
    0x40, 0x00, 0x00, 0x20, 0x34, 0x8a, 0x00, 0x87, 0x40, 0x00, 0x00, 0x21,
    0x36, 0x8a, 0x00, 0x87, 0x40, 0x00, 0x00, 0x22, 0xf2, 0xff, 0xff, 0xff,
    0x38, 0x8a, 0x80, 0x87, 0x80, 0x01, 0x00, 0x00, 0x00, 0x23, 0xf3, 0xff,
    0xff, 0xff, 0xff, 0xff,
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
inline constexpr std::uint16_t kSpecialConstantOneOver256 = 82;
inline constexpr std::uint16_t kSpecialConstantOneThird = 152;
inline constexpr std::uint16_t kSpecialConstantOneSixth = 153;
inline constexpr std::size_t kAttributeFetchTemporaryCount = 6;
inline constexpr std::uint8_t kControlOpWdf = 0b0101;

bool IsSupportedSpecialConstant(std::uint16_t index) {
  return index <= 31 || index == kSpecialConstantOne ||
         index == kSpecialConstantTwo || index == kSpecialConstantFour ||
         index == kSpecialConstantEight || index == kSpecialConstantHalf ||
         index == kSpecialConstantQuarter ||
         index == kSpecialConstantEighth ||
         index == kSpecialConstantOneOver256 ||
         index == kSpecialConstantOneThird ||
         index == kSpecialConstantOneSixth;
}

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
  bool bitwise = false;
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
  const bool bitwise = alu_type == 0x02U;
  if (!control && !bitwise && alu_type != 0)
    DecodeError(offset + 2, "unsupported ALU group type");
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
  header.bitwise = bitwise;
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
    source.bank = static_cast<PcoRegisterBank>(bank);
    source.index = index;
    cursor += 3;
  }

  if (source.bank == PcoRegisterBank::kSpecial) {
    if (!IsSupportedSpecialConstant(source.index)) {
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
    if (source.index >= kPcoMaximumSharedCount)
      DecodeError(source_offset,
                  "shared register exceeds the modeled USC file");
  } else if (source.bank == PcoRegisterBank::kCoefficient) {
    /*
     * Reading a coefficient register directly is how a flat or otherwise
     * uninterpolated varying arrives: the plane's C term is its value, and no
     * FITRP is emitted for it.  The runtime bound is the coefficient count the
     * draw supplied; this is the file the encoding can name.
     */
    if (source.index >= kPcoMaximumVaryingCoefficientCount)
      DecodeError(source_offset,
                  "coefficient register exceeds the modeled USC file");
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

void ValidateGenericSource(PcoRegisterRef source, std::size_t offset) {
  switch (source.bank) {
  case PcoRegisterBank::kSpecial:
    if (!IsSupportedSpecialConstant(source.index))
      DecodeError(offset, "unsupported public special-constant register");
    return;
  case PcoRegisterBank::kTemporary:
    if (source.index >= kPcoTemporaryCount)
      DecodeError(offset, "temporary register exceeds the modeled USC file");
    return;
  case PcoRegisterBank::kVertexInput:
    if (source.index >= kPcoVertexInputCount)
      DecodeError(offset, "vertex-input register exceeds the modeled USC file");
    return;
  case PcoRegisterBank::kShared:
    if (source.index >= kPcoMaximumSharedCount)
      DecodeError(offset, "shared register exceeds the modeled USC file");
    return;
  default:
    DecodeError(offset, "source register bank is outside this PCO subset");
  }
}

TwoLowerSources DecodeTwoLowerSources(
    const std::vector<std::uint8_t> &binary, std::size_t group_end,
    std::size_t &cursor, bool expect_is0_source1 = false,
    bool allow_internal_true_source1 = false) {
  const std::size_t source_offset = cursor;
  if (group_end - cursor < 2)
    DecodeError(cursor, "truncated two-source lower encoding");

  const std::uint8_t byte0 = binary[cursor];
  const std::uint8_t byte1 = binary[cursor + 1];
  const bool ext1 = (byte1 & 0x40U) != 0;
  if (ext1 && group_end - cursor < 3)
    DecodeError(cursor, "truncated extended two-source lower encoding");
  const std::uint8_t byte2 = ext1 ? binary[cursor + 2] : 0;
  const bool ext2 = ext1 && (byte2 & 0x80U) != 0;
  if (ext2 && group_end - cursor < 4)
    DecodeError(cursor, "truncated long two-source lower encoding");
  const std::uint8_t byte3 = ext2 ? binary[cursor + 3] : 0;

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
  if (ext1 && (byte2 & 0x60U) != 0)
    DecodeError(source_offset + 2, "FADD lower-source mux is unsupported");
  const std::uint8_t expected_long_control =
      expect_is0_source1 ? UINT8_C(0x10) : UINT8_C(0);
  if (ext2 && (byte3 & 0x14U) != expected_long_control)
    DecodeError(source_offset + 3,
                "long two-source reserved/mux bits are not canonical");
  if (expect_is0_source1 && !ext2)
    DecodeError(source_offset,
                "two-source is0=s1 selector requires the long encoding");

  const std::uint8_t bank0 = static_cast<std::uint8_t>(
      ((byte0 >> 6U) & 1U) | (((byte2 >> 4U) & 1U) << 1U) |
      (ext2 ? (((byte3 >> 3U) & 1U) << 2U) : 0U));
  const std::uint16_t index0 = static_cast<std::uint16_t>(
      (byte0 & 0x3fU) | (((byte2 >> 2U) & 1U) << 6U) |
      (ext2 ? (((byte3 >> 1U) & 1U) << 7U) |
                  (((byte3 >> 5U) & 7U) << 8U)
            : 0U));
  const std::uint8_t bank1 = static_cast<std::uint8_t>(
      ((byte1 >> 5U) & 1U) | (((byte2 >> 3U) & 1U) << 1U));
  const std::uint16_t index1 = static_cast<std::uint16_t>(
      (byte1 & 0x1fU) | ((byte2 & 0x03U) << 5U) |
      (ext2 ? ((byte3 & 1U) << 7U) : 0U));

  const PcoRegisterBank source0_bank = static_cast<PcoRegisterBank>(bank0);
  const PcoRegisterBank source1_bank = static_cast<PcoRegisterBank>(bank1);
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
      if (!IsSupportedSpecialConstant(index)) {
        DecodeError(offset, "unsupported two-source special constant");
      }
      return;
    }
    if (bank == PcoRegisterBank::kShared) {
      if (index >= kPcoMaximumSharedCount)
        DecodeError(offset, "two-source shared register exceeds USC file");
      return;
    }
    if (bank == PcoRegisterBank::kCoefficient) {
      /* An uninterpolated varying is read straight out of its plane. */
      if (index >= kPcoMaximumVaryingCoefficientCount)
        DecodeError(offset, "two-source coefficient exceeds the USC file");
      return;
    }
    DecodeError(offset, "two-source register bank is outside this PCO subset");
  };
  validate_source(source0_bank, index0, source_offset);
  if (allow_internal_true_source1 &&
      source1_bank == PcoRegisterBank::kSpecial && index1 == 143) {
    /* sc143 is the public all-bits-one Boolean feed used internally by BCMP.
     * It is consumed inside the group and is never exposed as a generic USC
     * source operand. */
  } else {
    validate_source(source1_bank, index1, source_offset + 1);
  }

  cursor += ext2 ? 4 : ext1 ? 3 : 2;
  return {
      {source0_bank, index0},
      {source1_bank, index1},
  };
}

struct ThreeLowerSources {
  PcoRegisterRef source0{};
  PcoRegisterRef source1{};
  PcoRegisterRef source2{};
  std::uint8_t input_selector = 0;
};

ThreeLowerSources DecodeThreeLowerSources(
    const std::vector<std::uint8_t> &binary, std::size_t group_end,
    std::size_t &cursor) {
  const std::size_t source_offset = cursor;
  if (group_end - cursor < 4)
    DecodeError(cursor, "truncated three-source lower encoding");

  const std::uint8_t byte0 = binary[cursor];
  const std::uint8_t byte1 = binary[cursor + 1];
  const std::uint8_t byte2 = binary[cursor + 2];
  const std::uint8_t byte3 = binary[cursor + 3];
  if ((byte0 & 0x80U) == 0 || (byte1 & 0xc0U) != 0x40U)
    DecodeError(source_offset, "invalid three-source lower selector");

  const bool ext2 = (byte2 & 0x80U) != 0;
  std::size_t encoded_bytes = 4;
  std::uint8_t byte4 = 0;
  std::uint8_t byte5 = 0;
  if (ext2) {
    if (group_end - cursor < 5)
      DecodeError(cursor, "truncated extended three-source lower encoding");
    byte4 = binary[cursor + 4];
    encoded_bytes = 5;
    if ((byte4 & 0x04U) != 0) {
      if (group_end - cursor < 6)
        DecodeError(cursor, "truncated long three-source lower encoding");
      byte5 = binary[cursor + 5];
      encoded_bytes = 6;
      if ((byte5 & 0xc0U) != 0)
        DecodeError(source_offset + 5,
                    "long three-source reserved bits are non-zero");
    }
  }

  PcoRegisterRef source0;
  PcoRegisterRef source1;
  PcoRegisterRef source2;
  source0.bank = static_cast<PcoRegisterBank>(
      ((byte0 >> 6U) & 1U) | (((byte2 >> 4U) & 1U) << 1U) |
      (ext2 ? (((byte4 >> 3U) & 1U) << 2U) : 0U));
  source0.index = static_cast<std::uint16_t>(
      (byte0 & 0x3fU) | (((byte2 >> 2U) & 1U) << 6U) |
      (ext2 ? (((byte4 >> 1U) & 1U) << 7U) : 0U) |
      (encoded_bytes == 6 ? ((byte5 & 7U) << 8U) : 0U));
  source1.bank = static_cast<PcoRegisterBank>(
      ((byte1 >> 5U) & 1U) | (((byte2 >> 3U) & 1U) << 1U));
  source1.index = static_cast<std::uint16_t>(
      (byte1 & 0x1fU) | ((byte2 & 3U) << 5U) |
      (ext2 ? ((byte4 & 1U) << 7U) : 0U));
  source2.bank = static_cast<PcoRegisterBank>(
      ((byte3 >> 6U) & 3U) |
      (ext2 ? (((byte4 >> 7U) & 1U) << 2U) : 0U));
  source2.index = static_cast<std::uint16_t>(
      (byte3 & 0x3fU) |
      (ext2 ? (((byte4 >> 5U) & 3U) << 6U) : 0U) |
      (encoded_bytes == 6 ? (((byte5 >> 3U) & 7U) << 8U) : 0U));

  ValidateGenericSource(source0, source_offset);
  ValidateGenericSource(source1, source_offset + 1);
  ValidateGenericSource(source2, source_offset + 3);
  const std::uint8_t input_selector = static_cast<std::uint8_t>(
      ((byte2 >> 5U) & 3U) |
      (encoded_bytes >= 5 && (byte4 & 0x10U) != 0 ? 4U : 0U));
  if (input_selector > 5)
    DecodeError(source_offset + 2, "invalid embedded lower-source selector");
  cursor += encoded_bytes;
  return {source0, source1, source2, input_selector};
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

struct DecodedDestination {
  PcoWriteTarget target = PcoWriteTarget::kNone;
  std::uint16_t index = 0;
};

DecodedDestination DecodeGenericDestination(
    const std::vector<std::uint8_t> &binary, std::size_t group_end,
    std::size_t &cursor) {
  if (cursor >= group_end)
    DecodeError(cursor, "missing ALU destination encoding");
  const std::size_t destination_offset = cursor;
  const std::uint8_t byte = binary[cursor++];
  if ((byte & 0x80U) != 0)
    DecodeError(destination_offset,
                "extended ALU destinations exceed modeled register files");
  const std::uint8_t bank = (byte >> 6U) & 1U;
  const std::uint16_t index = byte & 0x3fU;
  if (bank == static_cast<std::uint8_t>(PcoRegisterBank::kTemporary)) {
    if (index >= kPcoTemporaryCount)
      DecodeError(destination_offset,
                  "ALU destination exceeds the temporary register file");
    return {PcoWriteTarget::kTemporary, index};
  }
  if (index >= kPixelOutput0SpecialIndex &&
      index < kPixelOutput0SpecialIndex + kPcoPixelOutputCount) {
    return {PcoWriteTarget::kPixelOutput,
            static_cast<std::uint16_t>(index - kPixelOutput0SpecialIndex)};
  }
  DecodeError(destination_offset,
              "ALU destination is neither TEMP nor PIXOUT");
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
  /*
   * PIXEL and CENTROID iteration both name a position inside the pixel to
   * evaluate the coefficient planes at.  With one sample per pixel the
   * centroid of the covered area is the pixel centre, so the two coincide and
   * the executor evaluates either at the sample it was given.  SAMPLE
   * iteration picks a different position per sample and stays fail-closed
   * until the model rasterizes more than one.
   */
  const bool supported_iteration =
      iteration_mode == 0 || iteration_mode == 2;
  if (backend_op != kBackendOpFitr || !perspective || drc != 0 ||
      reserved0 != 0 || !supported_iteration || reserved1 != 0 || saturate ||
      component_count < 1 || component_count > 4) {
    DecodeError(header.offset + 3,
                "FITRP must be perspective PIXEL or CENTROID with "
                "count1..4/drc0/no-saturate"
                " [op=" + std::to_string(backend_op) +
                " persp=" + std::to_string(perspective ? 1 : 0) +
                " drc=" + std::to_string(drc) +
                " r0=" + std::to_string(reserved0) +
                " mode=" + std::to_string(iteration_mode) +
                " r1=" + std::to_string(reserved1) +
                " sat=" + std::to_string(saturate ? 1 : 0) +
                " count=" + std::to_string(component_count) + "]");
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
  /*
   * Bit 7 of the coefficient high byte is the extended-encoding flag the
   * other source forms use, and it inserts one more byte before the
   * destination.  The compiler reaches for it once a program addresses enough
   * registers; the field it carries is pinned to the single value observed so
   * an unfamiliar one still fails closed rather than being misread.
   */
  const bool extended_source = (coefficient_high_byte & 0x80U) != 0;
  if (extended_source) {
    if (group_end - cursor < 3)
      DecodeError(cursor, "truncated extended FITRP source encoding");
    const std::uint8_t extended_byte = binary[cursor++];
    if (extended_byte != 0x02U) {
      DecodeError(cursor - 1,
                  "FITRP extended source control is not canonical [" +
                      std::to_string(extended_byte) + "]");
    }
  }
  const std::uint16_t coefficient_index = static_cast<std::uint16_t>(
      (coefficient_byte & 0x3fU) |
      ((coefficient_high_byte & 0x04U) != 0 ? 0x40U : 0x00U));
  if ((coefficient_byte & 0xc0U) != 0xc0U ||
      coefficient_index < 4 || (coefficient_index & 3U) != 0 ||
      static_cast<std::size_t>(coefficient_index) + component_count * 4U >
          kPcoMaximumVaryingCoefficientCount ||
      source1_byte != 0x40U ||
      (coefficient_high_byte & ~0x84U) != 0x10U ||
      source3_byte != 0xc0U) {
    DecodeError(cursor - 1,
                "FITRP coefficient source encoding changed [cf=" +
                    std::to_string(coefficient_byte) +
                    " s1=" + std::to_string(source1_byte) +
                    " cfhi=" + std::to_string(coefficient_high_byte) +
                    " s3=" + std::to_string(source3_byte) +
                    " index=" + std::to_string(coefficient_index) +
                    " count=" + std::to_string(component_count) +
                    " next=" +
                    std::to_string(cursor < group_end ? binary[cursor] : 999) +
                    "," +
                    std::to_string(cursor + 1 < group_end ? binary[cursor + 1]
                                                          : 999) +
                    "," +
                    std::to_string(cursor + 2 < group_end ? binary[cursor + 2]
                                                          : 999) +
                    " total=" + std::to_string(header.total_bytes) + "]");
  }
  const std::uint8_t destination_byte = binary[cursor++];
  const std::uint16_t destination_index = destination_byte & 0x3fU;
  if ((destination_byte & 0xc0U) != 0x40U ||
      static_cast<std::size_t>(destination_index) + component_count >
          kPcoTemporaryCount ||
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
  instruction.iteration_mode = iteration_mode == 2
                                   ? PcoIterationMode::kCentroid
                                   : PcoIterationMode::kPixel;
  instruction.perspective = 1;
  instruction.saturate = 0;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeWdfGroup(
    const std::vector<std::uint8_t> &binary, const GroupHeader &header,
    std::uint16_t group_index) {
  if (!header.control || header.da != 0 || header.total_bytes != 4 ||
      header.operation_origin != 0 || header.output_load_check ||
      header.write0_present || header.write1_present ||
      header.control_op != kControlOpWdf || header.control_misc != 0 ||
      header.repeat_count != 1 || header.end) {
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

[[maybe_unused]] PcoInstruction DecodeFragmentMoveGroup(
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

[[maybe_unused]] PcoInstruction DecodeFragmentFloatAddGroup(
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

[[maybe_unused]] PcoInstruction DecodeFragmentTemporaryMoveGroup(
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

PcoInstruction DecodeTextureSampleGroup(
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

  /* The three lower sources are the four-word texture state, two normalized
   * coordinates, and four-word sampler state.  Mesa register allocation is
   * free to choose the coordinate and response TEMP bases (the original
   * fixture used TEMP16/TEMP0 while glmark2 texture uses TEMP3/TEMP3), so
   * decode the public register fields rather than pinning those allocations.
   */
  const ThreeLowerSources sources =
      DecodeThreeLowerSources(binary, group_end, cursor);
  if (sources.input_selector != 0 ||
      sources.source0.bank != PcoRegisterBank::kShared ||
      static_cast<std::size_t>(sources.source0.index) + 4U >
          kPcoMaximumSharedCount ||
      sources.source1.bank != PcoRegisterBank::kTemporary ||
      static_cast<std::size_t>(sources.source1.index) + 2U >
          kPcoTemporaryCount ||
      sources.source2.bank != PcoRegisterBank::kShared ||
      static_cast<std::size_t>(sources.source2.index) + 4U >
          kPcoMaximumSharedCount ||
      sources.source0.index % kPcoTextureDescriptorDwordCount != 0 ||
      sources.source2.index !=
          sources.source0.index + 8U ||
      sources.source0.index / kPcoTextureDescriptorDwordCount >=
          kPcoMaximumTextureDescriptorSets ||
      static_cast<std::size_t>(sources.source0.index) +
              kPcoTextureDescriptorDwordCount >
          kPcoMaximumSharedCount) {
    DecodeError(header.offset + 5,
                "SMP source registers exceed the public 2D layout");
  }

  if (group_end - cursor < 3 || binary[cursor++] != 0x80U)
    DecodeError(cursor - 1, "unsupported SMP upper-source selector");
  const std::uint8_t response = binary[cursor++];
  if ((response & 0xe0U) != 0xa0U)
    DecodeError(cursor - 1, "SMP response is not a temporary range");
  const std::uint16_t response_base = response & 0x1fU;
  if (static_cast<std::size_t>(response_base) + 4U > kPcoTemporaryCount)
    DecodeError(cursor - 1, "SMP response exceeds the temporary file");
  if (binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported SMP ISS selection");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kTextureSample;
  instruction.target = PcoWriteTarget::kTemporary;
  instruction.source = sources.source1;
  instruction.source1 = sources.source0;
  instruction.source2 = sources.source2;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = response_base;
  instruction.component_count = 4;
  instruction.data_request = 0;
  instruction.source_count = 3;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericSimpleAluGroup(
    ShaderStage stage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  if (header.control || header.bitwise || header.da != 3 ||
      header.operation_origin != 0 || !header.write0_present ||
      header.write1_present || header.repeat_count != 1) {
    DecodeError(header.offset, "unsupported scalar ALU instruction-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (cursor >= group_end)
    DecodeError(cursor, "missing scalar ALU operation");

  PcoOpcode opcode = PcoOpcode::kInternal;
  std::uint8_t source_count = 0;
  std::uint8_t source0_floor = 0;
  std::uint8_t source0_absolute = 0;
  std::uint8_t source1_absolute = 0;
  std::uint8_t saturate = 0;
  const std::uint8_t main = binary[cursor++];
  switch (main) {
  case 0x00:
    opcode = PcoOpcode::kFloatAdd;
    source_count = 2;
    break;
  case 0x01:
    opcode = PcoOpcode::kFloatAdd;
    source_count = 2;
    source0_floor = 1;
    break;
  case 0x04:
    opcode = PcoOpcode::kFloatAdd;
    source_count = 2;
    source0_absolute = 1;
    break;
  case 0x08:
    opcode = PcoOpcode::kFloatAddNegateSource0;
    source_count = 2;
    break;
  case 0x10:
    opcode = PcoOpcode::kFloatAdd;
    source_count = 2;
    saturate = 1;
    break;
  case 0x40:
    opcode = PcoOpcode::kFloatMultiply;
    source_count = 2;
    break;
  case 0x41:
    opcode = PcoOpcode::kFloatMultiply;
    source_count = 2;
    source0_floor = 1;
    break;
  case 0x42:
    opcode = PcoOpcode::kFloatMultiply;
    source_count = 2;
    source1_absolute = 1;
    break;
  case 0x80:
    opcode = PcoOpcode::kReciprocal;
    source_count = 1;
    break;
  case 0x81:
    opcode = PcoOpcode::kReciprocalSquareRoot;
    source_count = 1;
    break;
  case 0x82:
    opcode = PcoOpcode::kFloatLog2;
    source_count = 1;
    break;
  case 0x83:
    opcode = PcoOpcode::kFloatExp2;
    source_count = 1;
    break;
  case 0x87:
    opcode = PcoOpcode::kMoveBypass;
    source_count = 1;
    break;
  case 0x97:
    if (cursor >= group_end)
      DecodeError(cursor, "missing scalar MBYP source modifier");
    if (binary[cursor] == 0x01U)
      opcode = PcoOpcode::kFloatAbs;
    else if (binary[cursor] == 0x02U)
      opcode = PcoOpcode::kFloatNegate;
    else
      DecodeError(cursor, "unsupported scalar MBYP source modifier");
    ++cursor;
    source_count = 1;
    break;
  case 0xc0:
    opcode = PcoOpcode::kFloatMad;
    source_count = 3;
    break;
  case 0xc2:
    opcode = PcoOpcode::kFloatMadNegateSource2;
    source_count = 3;
    break;
  case 0xc8:
    opcode = PcoOpcode::kFloatMadNegateSource0;
    source_count = 3;
    break;
  case 0xca:
    opcode = PcoOpcode::kFloatMadNegateSource0Source2;
    source_count = 3;
    break;
  case 0x9c: {
    /*
     * UNPCK's format selector, as the compiler's own PCO_PCK_FORMAT_* values:
     * 6 is U32, 7 is S32 and 14 is F16F16.  A whole-word format makes the
     * unpack an integer-to-float conversion; the half format takes the low
     * sixteen bits.  Every other format stays fail-closed by number.
     */
    if (cursor >= group_end)
      DecodeError(cursor, "truncated scalar PCK/UNPCK format");
    const std::uint8_t pck_format = binary[cursor++];
    if (pck_format == 0x06U)
      opcode = PcoOpcode::kUnpackUnsignedToFloat;
    else if (pck_format == 0x07U)
      opcode = PcoOpcode::kUnpackSignedToFloat;
    else if (pck_format == 0x0eU)
      opcode = PcoOpcode::kFloatUnpackHalf;
    else
      DecodeError(cursor - 1, "unsupported scalar PCK/UNPCK format [" +
                                  std::to_string(pck_format) + "]");
    source_count = 1;
    break;
  }
  default:
    DecodeError(cursor - 1,
                "unsupported public scalar ALU operation [" +
                    std::to_string(main) + "]");
  }

  PcoRegisterRef source0{};
  PcoRegisterRef source1{};
  PcoRegisterRef source2{};
  if (source_count == 1) {
    source0 = DecodeOneLowerSource(binary, group_end, cursor);
  } else if (source_count == 2) {
    const TwoLowerSources sources =
        DecodeTwoLowerSources(binary, group_end, cursor);
    source0 = sources.source0;
    source1 = sources.source1;
  } else {
    const ThreeLowerSources sources =
        DecodeThreeLowerSources(binary, group_end, cursor);
    if (sources.input_selector != 0)
      DecodeError(header.offset,
                  "P0 scalar ALU requires embedded is0=s0");
    source0 = sources.source0;
    source1 = sources.source1;
    source2 = sources.source2;
  }
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported scalar ALU upper-source encoding");
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported scalar ALU ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (header.output_load_check !=
      (destination.target == PcoWriteTarget::kPixelOutput)) {
    DecodeError(header.offset,
                "output-load-check does not match the ALU destination");
  }
  if (stage == ShaderStage::kVertex &&
      destination.target != PcoWriteTarget::kTemporary) {
    DecodeError(header.offset, "vertex scalar ALU cannot write PIXOUT");
  }
  if (destination.target == PcoWriteTarget::kPixelOutput &&
      opcode != PcoOpcode::kMoveBypass) {
    DecodeError(header.offset, "only MBYP may write a modeled PIXOUT");
  }
  if (destination.target == PcoWriteTarget::kPixelOutput &&
      source0.bank == PcoRegisterBank::kSpecial &&
      source0.index != kSpecialConstantZero &&
      source0.index != kSpecialConstantOne &&
      source0.index != kSpecialConstantHalf) {
    DecodeError(header.offset,
                "PIXOUT special source is outside the public color gate");
  }
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = opcode;
  instruction.target = destination.target;
  instruction.source = source0;
  instruction.source1 = source1;
  instruction.source2 = source2;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source0_floor = source0_floor;
  instruction.source0_absolute = source0_absolute;
  instruction.source1_absolute = source1_absolute;
  instruction.saturate = saturate;
  instruction.source_count = source_count;
  instruction.repeat_count = 1;
  instruction.end_group = header.end ? 1U : 0U;
  return instruction;
}

PcoInstruction DecodeGenericPackHalfGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  if (header.control || header.bitwise || header.da != 5 ||
      header.operation_origin != 1 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1) {
    DecodeError(header.offset, "unsupported scalar PCK instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (group_end - cursor < 2 || binary[cursor++] != 0x9cU)
    DecodeError(header.offset + 3, "expected scalar PCK.F16F16 operation");
  const std::uint8_t rounding = binary[cursor++];
  PcoOpcode opcode = PcoOpcode::kInternal;
  if (rounding == 0x0eU)
    opcode = PcoOpcode::kFloatPackHalfRtne;
  else if (rounding == 0x4eU)
    opcode = PcoOpcode::kFloatPackHalfRtz;
  else
    DecodeError(header.offset + 4,
                "unsupported scalar PCK.F16F16 rounding mode");
  const ThreeLowerSources sources =
      DecodeThreeLowerSources(binary, group_end, cursor);
  if (sources.input_selector != 5)
    DecodeError(header.offset, "PCK requires embedded is0=s2");
  if (sources.source0.bank != PcoRegisterBank::kSpecial ||
      sources.source0.index != kSpecialConstantZero ||
      sources.source1.bank != PcoRegisterBank::kSpecial ||
      sources.source1.index != kSpecialConstantZero) {
    DecodeError(header.offset, "PCK unused lower sources are not canonical");
  }
  if (cursor >= group_end || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "unsupported PCK upper-source encoding");
  if (cursor >= group_end || binary[cursor++] != 0x2cU)
    DecodeError(cursor - 1, "unsupported PCK ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "PCK destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = opcode;
  instruction.target = destination.target;
  instruction.source = sources.source2;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 1;
  instruction.repeat_count = 1;
  instruction.end_group = header.end ? 1U : 0U;
  return instruction;
}

PcoInstruction DecodeGenericFloatMaxGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  if (header.control || header.bitwise || header.da != 7 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1) {
    DecodeError(header.offset, "unsupported FMAX instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {0xd0, 0x3c, 0xfa,
                                               0x10, 0x87, 0x87};
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected)
      DecodeError(cursor - 1, "unsupported TST/MOVC FMAX phase sequence");
  }
  const PcoRegisterRef source0 =
      DecodeOneLowerSource(binary, group_end, cursor);
  const PcoRegisterRef source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x10U)
    DecodeError(cursor - 1, "unsupported FMAX ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "FMAX destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatMax;
  instruction.target = destination.target;
  instruction.source = source0;
  instruction.source1 = source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = header.end ? 1U : 0U;
  return instruction;
}

PcoInstruction DecodeGenericFloatMinGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  /* Public FMIN lowers to an ordered TST.L and MOVC: P0 carries source0,
   * P1 carries source1, true selects P0, and false selects P1.  The false
   * source routing is observable for unordered inputs and equal signed zero,
   * so retain this exact phase form rather than treating it as host min(). */
  if (header.control || header.bitwise || header.da != 7 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1 || header.end) {
    DecodeError(header.offset, "unsupported FMIN instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {
      0xd0, 0x3c, 0xf0, 0x11, 0x87, 0x87,
  };
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected)
      DecodeError(cursor - 1, "unsupported TST/MOVC FMIN phase sequence");
  }
  const PcoRegisterRef source0 =
      DecodeOneLowerSource(binary, group_end, cursor);
  const PcoRegisterRef source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x10U)
    DecodeError(cursor - 1, "unsupported FMIN ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "FMIN destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatMin;
  instruction.target = destination.target;
  instruction.source = source0;
  instruction.source1 = source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericConditionalSelectGreaterZeroGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  if (header.control || header.bitwise || header.da != 7 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1 || header.end) {
    DecodeError(header.offset,
                "unsupported CSEL.F32.GZ instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {
      0xd0, 0x3c, 0xf2, 0x10, 0x87, 0x87,
  };
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected) {
      DecodeError(cursor - 1,
                  "unsupported CSEL.F32.GZ TST/MOVC phase sequence");
    }
  }

  /* P0 supplies the true value, is0/s1 supplies the floating-point
   * condition, and P1 supplies the false value.  The long lower-source
   * encoding is mandatory because it carries the exact is0=s1 selector. */
  const TwoLowerSources lower =
      DecodeTwoLowerSources(binary, group_end, cursor, true, false);
  const PcoRegisterRef false_source =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x11U) {
    DecodeError(cursor - 1,
                "unsupported CSEL.F32.GZ ISS selection");
  }
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary) {
    DecodeError(header.offset,
                "CSEL.F32.GZ destination must be temporary");
  }
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kConditionalSelectGreaterZero;
  instruction.target = destination.target;
  instruction.source = lower.source1;
  instruction.source1 = lower.source0;
  instruction.source2 = false_source;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 3;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericFloatEqualGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  if (header.control || header.bitwise || header.da != 8 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1 || header.end) {
    DecodeError(header.offset,
                "unsupported BCMP.F32.E instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {
      0xd3, 0x3c, 0xe8, 0x9c, 0x1e, 0x87, 0x87,
  };
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected) {
      DecodeError(cursor - 1,
                  "unsupported BCMP.F32.E TST/PCK/MOVC phase sequence");
    }
  }

  const TwoLowerSources lower =
      DecodeTwoLowerSources(binary, group_end, cursor, true, true);
  if (lower.source1.bank != PcoRegisterBank::kSpecial ||
      lower.source1.index != 143) {
    DecodeError(header.offset,
                "BCMP.F32.E requires canonical internal sc143 true source");
  }
  const PcoRegisterRef compare_source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x20U)
    DecodeError(cursor - 1, "unsupported BCMP.F32.E ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "BCMP.F32.E destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatEqual;
  instruction.target = destination.target;
  instruction.source = lower.source0;
  instruction.source1 = compare_source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericFloatLessGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  /* Public Mesa BCMP.F32.L is a five-phase group.  P0/P1 move the ordered
   * compare operands, P2 creates zero solely for MOVC's false value, TST.L
   * produces the predicate, and MOVC selects canonical sc143 or zero. */
  if (header.control || header.bitwise || header.da != 9 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1 || header.end) {
    DecodeError(header.offset,
                "unsupported BCMP.F32.L instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {
      0xd3, 0x3c, 0xf0, 0x01, 0x9c, 0x1e, 0x87, 0x87,
  };
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected) {
      DecodeError(cursor - 1,
                  "unsupported BCMP.F32.L TST/PCK/MOVC phase sequence");
    }
  }

  const TwoLowerSources lower =
      DecodeTwoLowerSources(binary, group_end, cursor, true, true);
  if (lower.source1.bank != PcoRegisterBank::kSpecial ||
      lower.source1.index != 143) {
    DecodeError(header.offset,
                "BCMP.F32.L requires canonical internal sc143 true source");
  }
  const PcoRegisterRef compare_source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x20U)
    DecodeError(cursor - 1, "unsupported BCMP.F32.L ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "BCMP.F32.L destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatLess;
  instruction.target = destination.target;
  instruction.source = lower.source0;
  instruction.source1 = compare_source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericFloatGreaterEqualGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  /* Public Mesa BCMP.F32.GE uses the same canonical Boolean materialization
   * as BCMP.F32.E: P0/P1 move the ordered compare operands, P2 supplies the
   * zero false value, TST.GE forms the predicate, and MOVC selects sc143 or
   * zero.  Accept only the exact public five-phase shape. */
  if (header.control || header.bitwise || header.da != 8 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1 || header.end) {
    DecodeError(header.offset,
                "unsupported BCMP.F32.GE instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {
      0xd3, 0x3c, 0xec, 0x9c, 0x1e, 0x87, 0x87,
  };
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected) {
      DecodeError(cursor - 1,
                  "unsupported BCMP.F32.GE TST/PCK/MOVC phase sequence");
    }
  }

  const TwoLowerSources lower =
      DecodeTwoLowerSources(binary, group_end, cursor, true, true);
  if (lower.source1.bank != PcoRegisterBank::kSpecial ||
      lower.source1.index != 143) {
    DecodeError(
        header.offset,
        "BCMP.F32.GE requires canonical internal sc143 true source");
  }
  const PcoRegisterRef compare_source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x20U)
    DecodeError(cursor - 1, "unsupported BCMP.F32.GE ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "BCMP.F32.GE destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatGreaterEqual;
  instruction.target = destination.target;
  instruction.source = lower.source0;
  instruction.source1 = compare_source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericFloatGreaterEqualOneZeroGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  /* Terrain's public Mesa FS also uses a distinct BCMP.F32.GE form whose P2
   * PCK.ONE and MOVC materialize binary32 1.0/0.0 instead of sc143 Boolean
   * bits.  Keep this encoding separate from the canonical Boolean form. */
  if (header.control || header.bitwise || header.da != 8 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1 || header.end) {
    DecodeError(
        header.offset,
        "unsupported BCMP.F32.GE.ONE instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {
      0xd2, 0x3c, 0xec, 0x9c, 0x1f, 0x87, 0x87,
  };
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected) {
      DecodeError(
          cursor - 1,
          "unsupported BCMP.F32.GE.ONE TST/PCK/MOVC phase sequence");
    }
  }

  const TwoLowerSources lower =
      DecodeTwoLowerSources(binary, group_end, cursor, true, true);
  if (lower.source1.bank != PcoRegisterBank::kSpecial ||
      lower.source1.index != 0) {
    DecodeError(header.offset,
                "BCMP.F32.GE.ONE requires canonical internal sc0 source");
  }
  const PcoRegisterRef compare_source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x30U)
    DecodeError(cursor - 1, "unsupported BCMP.F32.GE.ONE ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary) {
    DecodeError(header.offset,
                "BCMP.F32.GE.ONE destination must be temporary");
  }
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kFloatGreaterEqual;
  instruction.target = destination.target;
  instruction.source = lower.source0;
  instruction.source1 = compare_source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.comparison_result_float_one = 1;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericConditionalSelectGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  if (header.control || header.bitwise || header.da != 7 ||
      header.operation_origin != 5 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1 || header.end) {
    DecodeError(header.offset, "unsupported BCSEL instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  constexpr std::uint8_t kCanonicalPhases[] = {
      0xd1, 0x3c, 0xf0, 0xb0, 0x87,
  };
  for (std::uint8_t expected : kCanonicalPhases) {
    if (cursor >= group_end || binary[cursor++] != expected)
      DecodeError(cursor - 1, "unsupported BCSEL TST/MOVC phase sequence");
  }
  PcoOpcode opcode = PcoOpcode::kConditionalSelect;
  if (cursor >= group_end)
    DecodeError(cursor, "missing BCSEL P0 MBYP operation");
  if (binary[cursor] == 0x87U) {
    ++cursor;
  } else if (binary[cursor] == 0x97U) {
    ++cursor;
    if (cursor >= group_end || binary[cursor++] != 0x02U)
      DecodeError(cursor - 1, "unsupported BCSEL true-source negate modifier");
    opcode = PcoOpcode::kConditionalSelectNegateTrue;
  } else {
    DecodeError(cursor, "unsupported BCSEL P0 MBYP operation");
  }

  const TwoLowerSources lower =
      DecodeTwoLowerSources(binary, group_end, cursor, true, false);
  const PcoRegisterRef false_source =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (cursor >= group_end || binary[cursor++] != 0x01U)
    DecodeError(cursor - 1, "unsupported BCSEL ISS selection");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "BCSEL destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = opcode;
  instruction.target = destination.target;
  instruction.source = lower.source1;
  instruction.source1 = lower.source0;
  instruction.source2 = false_source;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 3;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericPhase2Group(
    ShaderStage stage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  const std::size_t operation_offset = header.offset + 3;
  if (operation_offset >= header.offset + header.total_bytes)
    DecodeError(operation_offset, "missing phase-2 ALU operation");
  switch (binary[operation_offset]) {
  case 0xd0:
    if (operation_offset + 2 >= header.offset + header.total_bytes) {
      DecodeError(operation_offset,
                  "truncated FMAX/CSEL.F32.GZ phase-2 operation");
    }
    if (binary[operation_offset + 2] == 0xfaU)
      return DecodeGenericFloatMaxGroup(stage, binary, header, group_index);
    if (binary[operation_offset + 2] == 0xf0U)
      return DecodeGenericFloatMinGroup(stage, binary, header, group_index);
    if (binary[operation_offset + 2] == 0xf2U) {
      return DecodeGenericConditionalSelectGreaterZeroGroup(
          stage, binary, header, group_index);
    }
    DecodeError(operation_offset + 2,
                "unsupported FMAX/CSEL.F32.GZ TST operation");
  case 0xd1:
    return DecodeGenericConditionalSelectGroup(stage, binary, header,
                                                group_index);
  case 0xd2:
    if (operation_offset + 2 >= header.offset + header.total_bytes)
      DecodeError(operation_offset, "truncated BCMP.F32.GE.ONE operation");
    if (binary[operation_offset + 2] == 0xecU) {
      return DecodeGenericFloatGreaterEqualOneZeroGroup(
          stage, binary, header, group_index);
    }
    DecodeError(operation_offset + 2,
                "unsupported BCMP.F32.GE.ONE TST operation");
  case 0xd3:
    if (operation_offset + 2 >= header.offset + header.total_bytes)
      DecodeError(operation_offset, "truncated BCMP.F32 operation");
    if (binary[operation_offset + 2] == 0xe8U)
      return DecodeGenericFloatEqualGroup(stage, binary, header, group_index);
    if (binary[operation_offset + 2] == 0xecU) {
      return DecodeGenericFloatGreaterEqualGroup(stage, binary, header,
                                                 group_index);
    }
    if (binary[operation_offset + 2] == 0xf0U)
      return DecodeGenericFloatLessGroup(stage, binary, header, group_index);
    DecodeError(operation_offset + 2,
                "unsupported BCMP.F32 TST operation");
  default:
    DecodeError(operation_offset,
                "phase-2 ALU operation is outside the public subset");
  }
}

PcoInstruction DecodeGenericImmediateGroup(
    ShaderStage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  if (!header.bitwise || header.control || header.da != 8 ||
      header.operation_origin != 1 || header.output_load_check ||
      !header.write0_present || header.write1_present ||
      header.repeat_count != 1) {
    DecodeError(header.offset,
                "unsupported bitwise-immediate instruction-group header");
  }
  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (group_end - cursor < 8 || binary[cursor++] != 0x13U)
    DecodeError(header.offset + 3, "expected BBYP immediate32 phase");
  const std::uint32_t immediate =
      static_cast<std::uint32_t>(binary[cursor]) |
      (static_cast<std::uint32_t>(binary[cursor + 1]) << 8U) |
      (static_cast<std::uint32_t>(binary[cursor + 2]) << 16U) |
      (static_cast<std::uint32_t>(binary[cursor + 3]) << 24U);
  cursor += 4;
  if (binary[cursor++] != 0x00U || binary[cursor++] != 0x00U)
    DecodeError(cursor - 1, "bitwise immediate has non-zero unused sources");
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "bitwise immediate destination must be TEMP");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kMoveImmediate;
  instruction.target = destination.target;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.immediate = immediate;
  instruction.source_count = 0;
  instruction.repeat_count = 1;
  instruction.end_group = header.end ? 1U : 0U;
  return instruction;
}

PcoInstruction DecodeGenericBitwiseAndGroup(
    ShaderStage stage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  /* Ideas lighting emits this public two-phase form for Boolean conjunction:
   *   p0: bbyp0s1 ft2, ft3, s2
   *   p1: logical.and ft4, _, ft2, _, s3
   * The three leading lower-source bytes encode unused s0/s1 plus the s2
   * selector.  Keep every phase and selector exact while allowing the actual
   * register indices carried by the final lower and upper source bytes. */
  if (stage != ShaderStage::kFragment || !header.bitwise || header.control ||
      header.da != 5 || header.operation_origin != 3 ||
      header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end ||
      header.total_bytes != 12) {
    DecodeError(header.offset,
                "unsupported LOGICAL.AND instruction-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (group_end - cursor < 8 || binary[cursor++] != 0x41U)
    DecodeError(header.offset + 3, "expected LOGICAL.AND phase-1 operation");
  if (binary[cursor++] != 0x02U)
    DecodeError(header.offset + 4, "expected BBYP0S1 phase-0 operation");
  if (binary[cursor++] != 0x80U || binary[cursor++] != 0x40U ||
      binary[cursor++] != 0x00U) {
    DecodeError(header.offset + 5,
                "LOGICAL.AND lower-source selector is not canonical");
  }
  const PcoRegisterRef source0 =
      DecodeOneLowerSource(binary, group_end, cursor);
  const PcoRegisterRef source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "LOGICAL.AND destination must be temporary");
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kBitwiseAnd;
  instruction.target = destination.target;
  instruction.source = source0;
  instruction.source1 = source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeGenericBitwiseXnorGroup(
    ShaderStage stage, const std::vector<std::uint8_t> &binary,
    const GroupHeader &header, std::uint16_t group_index) {
  /* Terrain D1 emits the public two-phase logical-not shape:
   *   p0: bbyp0s1 ft2, ft3, s2
   *   p1: logical.xnor ft4, _, ft2, _, s3
   * Its observed s3 is canonical sc0, making the visible operation ~s2.
   * Retain a distinct opcode so the ISS and histogram do not mislabel XNOR
   * as AND/XOR, and accept no other logical phase operation. */
  if (stage != ShaderStage::kFragment || !header.bitwise || header.control ||
      header.da != 5 || header.operation_origin != 3 ||
      header.output_load_check || !header.write0_present ||
      header.write1_present || header.repeat_count != 1 || header.end ||
      header.total_bytes != 12) {
    DecodeError(header.offset,
                "unsupported LOGICAL.XNOR instruction-group header");
  }

  const std::size_t group_end = header.offset + header.total_bytes;
  std::size_t cursor = header.offset + 3;
  if (group_end - cursor < 8 || binary[cursor++] != 0x46U)
    DecodeError(header.offset + 3, "expected LOGICAL.XNOR phase-1 operation");
  if (binary[cursor++] != 0x02U)
    DecodeError(header.offset + 4, "expected BBYP0S1 phase-0 operation");
  if (binary[cursor++] != 0x80U || binary[cursor++] != 0x40U ||
      binary[cursor++] != 0x00U) {
    DecodeError(header.offset + 5,
                "LOGICAL.XNOR lower-source selector is not canonical");
  }
  const PcoRegisterRef source0 =
      DecodeOneLowerSource(binary, group_end, cursor);
  const PcoRegisterRef source1 =
      DecodeOneLowerSource(binary, group_end, cursor);
  if (source0.bank != PcoRegisterBank::kTemporary) {
    DecodeError(header.offset,
                "LOGICAL.XNOR source must be a temporary register");
  }
  if (source1.bank != PcoRegisterBank::kSpecial || source1.index != 0) {
    DecodeError(header.offset,
                "LOGICAL.XNOR requires the captured canonical sc0 source");
  }
  const DecodedDestination destination =
      DecodeGenericDestination(binary, group_end, cursor);
  if (destination.target != PcoWriteTarget::kTemporary)
    DecodeError(header.offset, "LOGICAL.XNOR destination must be temporary");
  if (destination.index != source0.index) {
    DecodeError(header.offset,
                "LOGICAL.XNOR captured form must update its source in place");
  }
  ValidateAlignmentPadding(binary, header.offset, cursor, group_end);

  PcoInstruction instruction;
  instruction.opcode = PcoOpcode::kBitwiseXnor;
  instruction.target = destination.target;
  instruction.source = source0;
  instruction.source1 = source1;
  instruction.binary_offset = CheckedU32(header.offset + 3, "PCO offset");
  instruction.group_index = group_index;
  instruction.output_index = destination.index;
  instruction.source_count = 2;
  instruction.repeat_count = 1;
  instruction.end_group = 0;
  return instruction;
}

PcoInstruction DecodeFragmentGroup(const std::vector<std::uint8_t> &binary,
                                   const GroupHeader &header,
                                   std::uint16_t group_index) {
  if (header.control)
    return DecodeWdfGroup(binary, header, group_index);
  if (header.bitwise) {
    if (header.operation_origin == 1)
      return DecodeGenericImmediateGroup(ShaderStage::kFragment, binary,
                                         header, group_index);
    if (header.operation_origin == 3) {
      const std::size_t operation_offset = header.offset + 3;
      if (operation_offset >= header.offset + header.total_bytes)
        DecodeError(operation_offset, "missing logical phase operation");
      if (binary[operation_offset] == 0x41U) {
        return DecodeGenericBitwiseAndGroup(ShaderStage::kFragment, binary,
                                            header, group_index);
      }
      if (binary[operation_offset] == 0x46U) {
        return DecodeGenericBitwiseXnorGroup(ShaderStage::kFragment, binary,
                                             header, group_index);
      }
      DecodeError(operation_offset,
                  "logical phase operation is outside the public subset");
    }
    DecodeError(header.offset,
                "bitwise operation is outside the fragment public subset");
  }
  if (header.operation_origin == 2) {
    if (binary[header.offset + 3] >> 5U == kBackendOpDma)
      return DecodeTextureSampleGroup(binary, header, group_index);
    return DecodeFragmentFitrpGroup(binary, header, group_index);
  }
  if (header.operation_origin == 0)
    return DecodeGenericSimpleAluGroup(ShaderStage::kFragment, binary, header,
                                      group_index);
  if (header.operation_origin == 1)
    return DecodeGenericPackHalfGroup(ShaderStage::kFragment, binary, header,
                                      group_index);
  if (header.operation_origin == 5)
    return DecodeGenericPhase2Group(ShaderStage::kFragment, binary, header,
                                    group_index);
  DecodeError(header.offset,
              "instruction origin is outside the fragment PCO subset");
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

[[maybe_unused]] PcoInstruction DecodeVertexMoveGroup(
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

[[maybe_unused]] PcoInstruction DecodeVertexFloatAddGroup(
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

[[maybe_unused]] PcoInstruction DecodeVertexFloatMultiplyGroup(
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
    if (source.index + header.repeat_count > kPcoTemporaryCount) {
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
  if (header.control)
    return DecodeWdfGroup(binary, header, group_index);
  if (header.bitwise)
    return DecodeGenericImmediateGroup(ShaderStage::kVertex, binary, header,
                                       group_index);
  if (header.operation_origin == 0)
    return DecodeGenericSimpleAluGroup(ShaderStage::kVertex, binary, header,
                                      group_index);
  if (header.operation_origin == 1)
    return DecodeGenericPackHalfGroup(ShaderStage::kVertex, binary, header,
                                      group_index);
  if (header.operation_origin == 2 &&
      binary[header.offset + 3] >> 5U == kBackendOpDma)
    return DecodeTextureSampleGroup(binary, header, group_index);
  if (header.operation_origin == 2)
    return DecodeVertexBackendGroup(binary, header, group_index);
  if (header.operation_origin == 5)
    return DecodeGenericPhase2Group(ShaderStage::kVertex, binary, header,
                                    group_index);
  DecodeError(header.offset,
              "instruction origin is outside the vertex PCO subset");
}

bool IsRegister(const PcoRegisterRef &reference, PcoRegisterBank bank,
                std::uint16_t index) {
  return reference.bank == bank && reference.index == index;
}

std::vector<PcoInstruction> BuildConditionalsInstructions(
    ShaderStage stage, const std::vector<std::uint8_t> &binary) {
  const bool vertex = stage == ShaderStage::kVertex;
  if ((vertex && binary != kConditionalsVertexBinary) ||
      (!vertex && binary != kConditionalsFragmentBinary)) {
    DecodeError(0, "conditionals semantic decoder requires the exact binary");
  }

  std::vector<GroupHeader> headers;
  std::size_t offset = 0;
  while (offset < binary.size()) {
    if (binary.size() - offset < 3)
      DecodeError(offset, "truncated conditionals group header");
    std::size_t words = binary[offset] & 0x0fU;
    if (words == 0)
      words = 16;
    GroupHeader header;
    header.offset = offset;
    header.total_bytes = words * 2U;
    header.end = (binary[offset + 2] & 0x80U) != 0;
    header.repeat_count = static_cast<std::uint8_t>(
        ((binary[offset + 2] >> 1U) & 3U) + 1U);
    if (header.total_bytes < 3 ||
        header.total_bytes > binary.size() - offset) {
      DecodeError(offset, "conditionals group length exceeds the binary");
    }
    headers.push_back(header);
    offset += headers.back().total_bytes;
  }
  const std::size_t expected_groups = vertex ? 48U : 43U;
  if (headers.size() != expected_groups || offset != binary.size() ||
      !headers.back().end) {
    DecodeError(0, "conditionals group envelope changed");
  }

  const PcoRegisterRef zero{};
  const auto temp = [](std::uint16_t index) {
    return PcoRegisterRef{PcoRegisterBank::kTemporary, index};
  };
  const auto vtxin = [](std::uint16_t index) {
    return PcoRegisterRef{PcoRegisterBank::kVertexInput, index};
  };
  const auto shared = [](std::uint16_t index) {
    return PcoRegisterRef{PcoRegisterBank::kShared, index};
  };
  const auto special = [](std::uint16_t index) {
    return PcoRegisterRef{PcoRegisterBank::kSpecial, index};
  };

  std::vector<PcoInstruction> instructions;
  instructions.reserve(expected_groups);
  const auto append = [&](PcoOpcode opcode, PcoWriteTarget target,
                          PcoRegisterRef source0, PcoRegisterRef source1,
                          PcoRegisterRef source2, std::uint16_t output_index,
                          std::uint8_t source_count, std::uint32_t immediate,
                          std::uint8_t repeat_count) {
    if (instructions.size() >= headers.size())
      DecodeError(binary.size(), "conditionals semantic group overflow");
    const GroupHeader &header = headers[instructions.size()];
    if (header.repeat_count != repeat_count)
      DecodeError(header.offset, "conditionals repeat count changed");
    PcoInstruction instruction;
    instruction.opcode = opcode;
    instruction.target = target;
    instruction.source = source0;
    instruction.source1 = source1;
    instruction.source2 = source2;
    instruction.binary_offset =
        CheckedU32(header.offset + 3, "PCO conditionals offset");
    instruction.group_index =
        static_cast<std::uint16_t>(instructions.size());
    instruction.output_index = output_index;
    instruction.immediate = immediate;
    instruction.source_count = source_count;
    instruction.repeat_count = repeat_count;
    instruction.end_group = header.end ? 1U : 0U;
    instructions.push_back(instruction);
  };
  const auto internal = [&] {
    append(PcoOpcode::kInternal, PcoWriteTarget::kNone, zero, zero, zero, 0,
           0, 0, 1);
  };
  const auto unary = [&](PcoOpcode opcode, PcoRegisterRef source,
                         std::uint16_t output) {
    append(opcode, PcoWriteTarget::kTemporary, source, zero, zero, output, 1,
           0, 1);
  };
  const auto binary_op = [&](PcoOpcode opcode, PcoRegisterRef source0,
                             PcoRegisterRef source1, std::uint16_t output) {
    append(opcode, PcoWriteTarget::kTemporary, source0, source1, zero, output,
           2, 0, 1);
  };
  const auto ternary = [&](PcoOpcode opcode, PcoRegisterRef source0,
                           PcoRegisterRef source1, PcoRegisterRef source2,
                           std::uint16_t output) {
    append(opcode, PcoWriteTarget::kTemporary, source0, source1, source2,
           output, 3, 0, 1);
  };
  const auto immediate = [&](std::uint32_t value, std::uint16_t output) {
    append(PcoOpcode::kMoveImmediate, PcoWriteTarget::kTemporary, zero, zero,
           zero, output, 0, value, 1);
  };

  if (vertex) {
    unary(PcoOpcode::kFloatFloor, vtxin(0), 0);
    binary_op(PcoOpcode::kFloatSubtract, vtxin(0), temp(0), 0);
    binary_op(PcoOpcode::kFloatMultiply, temp(0),
              special(kSpecialConstantTwo), 1);
    unary(PcoOpcode::kFloatFloor, temp(1), 2);
    binary_op(PcoOpcode::kFloatSubtract, temp(1), temp(2), 1);
    immediate(UINT32_C(0x40400000), 2);
    binary_op(PcoOpcode::kFloatMultiply, temp(0), temp(2), 2);
    unary(PcoOpcode::kFloatFloor, temp(2), 3);
    binary_op(PcoOpcode::kFloatSubtract, temp(2), temp(3), 2);
    binary_op(PcoOpcode::kFloatGreaterEqual, temp(0),
              special(kSpecialConstantHalf), 3);
    ternary(PcoOpcode::kConditionalSelect, temp(3), temp(1), temp(2), 1);
    immediate(UINT32_C(0x3dcccccd), 2);
    binary_op(PcoOpcode::kFloatMultiply, temp(1), temp(2), 1);
    ternary(PcoOpcode::kFloatMad, temp(1), temp(0), vtxin(1), 0);
    for (std::uint16_t index = 0; index < 8; ++index)
      unary(PcoOpcode::kMoveBypass, shared(index), index + 1);
    for (std::uint16_t index = 0; index < 4; ++index) {
      const std::uint16_t source = static_cast<std::uint16_t>(index + 5);
      const std::uint16_t destination = index == 3 ? 0 : source;
      binary_op(PcoOpcode::kFloatMultiply, temp(source), temp(0),
                destination);
    }
    ternary(PcoOpcode::kFloatMad, temp(1), vtxin(0), temp(5), 1);
    ternary(PcoOpcode::kFloatMad, temp(2), vtxin(0), temp(6), 2);
    ternary(PcoOpcode::kFloatMad, temp(3), vtxin(0), temp(7), 3);
    ternary(PcoOpcode::kFloatMad, temp(4), vtxin(0), temp(0), 0);
    for (std::uint16_t index = 0; index < 4; ++index)
      unary(PcoOpcode::kMoveBypass, shared(index + 8), index + 4);
    ternary(PcoOpcode::kFloatMad, temp(4), vtxin(2), temp(1), 1);
    ternary(PcoOpcode::kFloatMad, temp(5), vtxin(2), temp(2), 4);
    ternary(PcoOpcode::kFloatMad, temp(6), vtxin(2), temp(3), 5);
    ternary(PcoOpcode::kFloatMad, temp(7), vtxin(2), temp(0), 6);
    unary(PcoOpcode::kMoveBypass, shared(12), 0);
    unary(PcoOpcode::kMoveBypass, shared(13), 7);
    unary(PcoOpcode::kMoveBypass, shared(14), 8);
    unary(PcoOpcode::kMoveBypass, shared(15), 9);
    binary_op(PcoOpcode::kFloatAdd, temp(1), temp(0), 0);
    binary_op(PcoOpcode::kFloatAdd, temp(4), temp(7), 1);
    binary_op(PcoOpcode::kFloatAdd, temp(5), temp(8), 2);
    binary_op(PcoOpcode::kFloatAdd, temp(6), temp(9), 3);
    append(PcoOpcode::kUvsWrite, PcoWriteTarget::kVertexOutput, temp(0), zero,
           zero, 0, 1, 0, 4);
    append(PcoOpcode::kUvsEmitEndTask, PcoWriteTarget::kNone, zero, zero, zero,
           0, 0, 0, 1);
  } else {
    for (unsigned index = 0; index < 5; ++index)
      internal();
    append(PcoOpcode::kFragmentCoordinate, PcoWriteTarget::kTemporary, zero,
           zero, zero, 0, 0, 0, 1);
    for (unsigned index = 0; index < 5; ++index)
      internal();
    append(PcoOpcode::kFragmentCoordinate, PcoWriteTarget::kTemporary, zero,
           zero, zero, 1, 0, 1, 1);
    unary(PcoOpcode::kMoveBypass, shared(0), 2);
    unary(PcoOpcode::kMoveBypass, shared(1), 3);
    ternary(PcoOpcode::kFloatMad, temp(1), temp(2), temp(3), 1);
    immediate(UINT32_C(0x38d1b717), 2);
    binary_op(PcoOpcode::kFloatMultiply, temp(0), temp(2), 0);
    binary_op(PcoOpcode::kFloatMultiply, temp(0), temp(1), 0);
    unary(PcoOpcode::kFloatFloor, temp(0), 1);
    binary_op(PcoOpcode::kFloatSubtract, temp(0), temp(1), 0);
    unary(PcoOpcode::kFloatPackHalfRtne, temp(0), 0);
    unary(PcoOpcode::kFloatUnpackHalf, temp(0), 0);
    binary_op(PcoOpcode::kFloatMultiply, temp(0),
              special(kSpecialConstantTwo), 1);
    unary(PcoOpcode::kFloatPackHalfRtne, temp(1), 1);
    unary(PcoOpcode::kFloatUnpackHalf, temp(1), 1);
    unary(PcoOpcode::kFloatFloor, temp(1), 2);
    binary_op(PcoOpcode::kFloatSubtract, temp(1), temp(2), 1);
    unary(PcoOpcode::kFloatPackHalfRtne, temp(1), 1);
    immediate(UINT32_C(0x40400000), 2);
    binary_op(PcoOpcode::kFloatMultiply, temp(0), temp(2), 2);
    unary(PcoOpcode::kFloatPackHalfRtne, temp(2), 2);
    unary(PcoOpcode::kFloatUnpackHalf, temp(2), 2);
    unary(PcoOpcode::kFloatFloor, temp(2), 3);
    binary_op(PcoOpcode::kFloatSubtract, temp(2), temp(3), 2);
    unary(PcoOpcode::kFloatPackHalfRtne, temp(2), 2);
    unary(PcoOpcode::kFloatUnpackHalf, temp(1), 1);
    unary(PcoOpcode::kFloatUnpackHalf, temp(2), 2);
    binary_op(PcoOpcode::kFloatGreaterEqual, temp(0),
              special(kSpecialConstantHalf), 0);
    ternary(PcoOpcode::kConditionalSelect, temp(0), temp(1), temp(2), 0);
    for (std::uint16_t output = 0; output < 3; ++output) {
      append(PcoOpcode::kMoveBypass, PcoWriteTarget::kPixelOutput, temp(0),
             zero, zero, output, 1, 0, 1);
    }
    append(PcoOpcode::kMoveBypass, PcoWriteTarget::kPixelOutput,
           special(kSpecialConstantOne), zero, zero, 3, 1, 0, 1);
  }

  if (instructions.size() != expected_groups)
    DecodeError(0, "conditionals semantic group count is inconsistent");
  return instructions;
}

bool SameConditionalsInstruction(const PcoInstruction &left,
                                 const PcoInstruction &right) {
  return left.opcode == right.opcode && left.target == right.target &&
         left.source.bank == right.source.bank &&
         left.source.index == right.source.index &&
         left.source1.bank == right.source1.bank &&
         left.source1.index == right.source1.index &&
         left.source2.bank == right.source2.bank &&
         left.source2.index == right.source2.index &&
         left.binary_offset == right.binary_offset &&
         left.group_index == right.group_index &&
         left.output_index == right.output_index &&
         left.branch_target_index == right.branch_target_index &&
         left.loop_count == right.loop_count &&
         left.immediate == right.immediate &&
         left.component_count == right.component_count &&
         left.data_request == right.data_request &&
         left.iteration_mode == right.iteration_mode &&
         left.perspective == right.perspective &&
         left.saturate == right.saturate &&
         left.source0_floor == right.source0_floor &&
         left.source0_absolute == right.source0_absolute &&
         left.source1_absolute == right.source1_absolute &&
         left.comparison_result_float_one ==
             right.comparison_result_float_one &&
         left.source_count == right.source_count &&
         left.repeat_count == right.repeat_count &&
         left.end_group == right.end_group;
}

bool MatchesConditionalsProfile(
    ShaderStage stage, const std::vector<PcoInstruction> &instructions) {
  /* Execution calls this once per shader lane.  Build each immutable semantic
   * profile once instead of reparsing and allocating the real PCO binary for
   * every vertex or fragment. */
  static const std::vector<PcoInstruction> vertex_expected =
      BuildConditionalsInstructions(ShaderStage::kVertex,
                                    kConditionalsVertexBinary);
  static const std::vector<PcoInstruction> fragment_expected =
      BuildConditionalsInstructions(ShaderStage::kFragment,
                                    kConditionalsFragmentBinary);
  const std::vector<PcoInstruction> &expected =
      stage == ShaderStage::kVertex ? vertex_expected : fragment_expected;
  if (instructions.size() != expected.size())
    return false;
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    if (!SameConditionalsInstruction(instructions[index], expected[index]))
      return false;
  }
  return true;
}

bool HasDefaultNonFitrpFieldsExceptSource0ModifierAndSaturate(
    const PcoInstruction &instruction) {
  return instruction.component_count == 1 && instruction.data_request == 0 &&
         instruction.iteration_mode == PcoIterationMode::kPixel &&
         instruction.perspective == 0;
}

bool HasDefaultNonFitrpFieldsExceptSource0Modifier(
    const PcoInstruction &instruction) {
  return HasDefaultNonFitrpFieldsExceptSource0ModifierAndSaturate(
             instruction) &&
         instruction.saturate == 0;
}

bool HasDefaultNonFitrpFields(const PcoInstruction &instruction) {
  return HasDefaultNonFitrpFieldsExceptSource0Modifier(instruction) &&
         instruction.source0_floor == 0 &&
         instruction.source0_absolute == 0 &&
         instruction.source1_absolute == 0 &&
         instruction.comparison_result_float_one == 0;
}

bool HasCanonicalGenericSource0Modifier(
    const PcoInstruction &instruction) {
  return instruction.source0_floor == 0 ||
         (instruction.source0_floor == 1 &&
          (instruction.opcode == PcoOpcode::kFloatAdd ||
           instruction.opcode == PcoOpcode::kFloatMultiply));
}

bool HasCanonicalGenericComparisonResult(
    const PcoInstruction &instruction) {
  return instruction.comparison_result_float_one == 0 ||
         (instruction.comparison_result_float_one == 1 &&
          instruction.opcode == PcoOpcode::kFloatGreaterEqual);
}

bool HasCanonicalGenericSaturate(const PcoInstruction &instruction) {
  return instruction.saturate == 0 ||
         (instruction.saturate == 1 &&
          instruction.opcode == PcoOpcode::kFloatAdd);
}

bool HasCanonicalGenericAbsoluteModifiers(
    const PcoInstruction &instruction) {
  if (instruction.source0_absolute > 1 ||
      instruction.source1_absolute > 1 ||
      (instruction.source0_absolute != 0 &&
       instruction.source1_absolute != 0)) {
    return false;
  }
  if (instruction.source0_absolute != 0) {
    return instruction.opcode == PcoOpcode::kFloatAdd &&
           instruction.source0_floor == 0 && instruction.saturate == 0;
  }
  if (instruction.source1_absolute != 0) {
    return instruction.opcode == PcoOpcode::kFloatMultiply &&
           instruction.source0_floor == 0 && instruction.saturate == 0;
  }
  return true;
}

bool HasCanonicalLogicalXnorShape(const PcoInstruction &instruction) {
  return instruction.opcode != PcoOpcode::kBitwiseXnor ||
         (instruction.target == PcoWriteTarget::kTemporary &&
          instruction.source_count == 2 &&
          instruction.source.bank == PcoRegisterBank::kTemporary &&
          instruction.source1.bank == PcoRegisterBank::kSpecial &&
          instruction.source1.index == 0 &&
          instruction.output_index == instruction.source.index);
}

bool HasCanonicalGenericNonFitrpFields(
    const PcoInstruction &instruction) {
  return HasDefaultNonFitrpFieldsExceptSource0ModifierAndSaturate(
             instruction) &&
         HasCanonicalGenericSource0Modifier(instruction) &&
         HasCanonicalGenericComparisonResult(instruction) &&
         HasCanonicalGenericSaturate(instruction) &&
         HasCanonicalGenericAbsoluteModifiers(instruction) &&
         HasCanonicalLogicalXnorShape(instruction);
}

bool IsDefaultUnusedRegister(const PcoRegisterRef &reference) {
  return IsRegister(reference, PcoRegisterBank::kSpecial, 0);
}

bool HasCanonicalUnusedSources(const PcoInstruction &instruction) {
  return (instruction.source_count >= 1 ||
          IsDefaultUnusedRegister(instruction.source)) &&
         (instruction.source_count >= 2 ||
          IsDefaultUnusedRegister(instruction.source1)) &&
         (instruction.source_count >= 3 ||
          IsDefaultUnusedRegister(instruction.source2));
}

bool HasDefaultControlFields(const PcoInstruction &instruction) {
  return instruction.branch_target_index == 0 && instruction.loop_count == 0;
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
  std::uint64_t written_mask = 0;
  bool request_pending = false;
  std::uint16_t pending_output = 0;
  std::uint8_t pending_components = 0;
  for (const PcoInstruction &instruction : instructions) {
    if (request_pending && instruction.opcode != PcoOpcode::kWaitDataFence) {
      DecodeError(instruction.binary_offset,
                  "vertex SMP must be followed by its DRC0 WDF");
    }
    const auto require_written = [&](const PcoRegisterRef &source,
                                     std::uint8_t repeat_count) {
      if (source.bank != PcoRegisterBank::kTemporary)
        return;
      uses_temporary_program = true;
      for (std::uint8_t repeat = 0; repeat < repeat_count; ++repeat) {
        const std::size_t index = source.index + repeat;
        if (index >= kPcoTemporaryCount ||
            (written_mask & (UINT64_C(1) << index)) == 0) {
          DecodeError(instruction.binary_offset,
                      "temporary register is read before it is written");
        }
      }
    };
    if (instruction.source_count >= 1)
      require_written(instruction.source, instruction.repeat_count);
    if (instruction.source_count >= 2)
      require_written(instruction.source1, instruction.repeat_count);
    if (instruction.source_count == 3)
      require_written(instruction.source2, instruction.repeat_count);

    if (instruction.opcode == PcoOpcode::kTextureSample) {
      uses_temporary_program = true;
      request_pending = true;
      pending_output = instruction.output_index;
      pending_components = instruction.component_count;
      continue;
    }
    if (instruction.opcode == PcoOpcode::kWaitDataFence) {
      uses_temporary_program = true;
      if (!request_pending ||
          static_cast<std::size_t>(pending_output) + pending_components >
              kPcoTemporaryCount) {
        DecodeError(instruction.binary_offset,
                    "vertex WDF has no bounded DRC0 response");
      }
      for (std::uint8_t component = 0; component < pending_components;
           ++component) {
        written_mask |= UINT64_C(1) << (pending_output + component);
      }
      request_pending = false;
      pending_output = 0;
      pending_components = 0;
      continue;
    }
    if (instruction.target == PcoWriteTarget::kTemporary) {
      uses_temporary_program = true;
      if (instruction.output_index >= kPcoTemporaryCount)
        DecodeError(instruction.binary_offset,
                    "temporary destination exceeds the USC file");
      written_mask |= UINT64_C(1) << instruction.output_index;
    }
    if (instruction.opcode == PcoOpcode::kUvsEmitEndTask)
      uses_temporary_program = true;
  }
  if (request_pending) {
    DecodeError(instructions.back().binary_offset,
                "vertex program ends with an unresolved DRC0 request");
  }

  const bool exact_profile =
      MatchesOneAttributeProfile(instructions) ||
      MatchesTwoAttributeProfile(instructions) ||
      MatchesFourAttributeProfile(instructions) ||
      MatchesEightAttributeProfile(instructions) ||
      MatchesVaryingsOneProfile(instructions) ||
      MatchesVaryingsTwoProfile(instructions) ||
      MatchesVaryingsFourProfile(instructions) ||
      MatchesVaryingsEightProfile(instructions) ||
      MatchesFillTexNearestVertexProfile(instructions) ||
      MatchesConditionalsProfile(ShaderStage::kVertex, instructions);
  if (!uses_temporary_program || exact_profile)
    return;

  std::size_t texture_sample_count = 0;
  for (const PcoInstruction &instruction : instructions) {
    if (instruction.opcode == PcoOpcode::kTextureSample) {
      const std::size_t coordinate_base = instruction.source.index;
      const std::uint64_t coordinate_mask =
          coordinate_base + 2U <= kPcoTemporaryCount
              ? (UINT64_C(3) << coordinate_base)
              : UINT64_C(0);
      if (++texture_sample_count > kPcoMaximumTextureSampleInstructions ||
          instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 3 || instruction.repeat_count != 1 ||
          instruction.component_count != kPcoPixelOutputCount ||
          instruction.data_request != 0 ||
          instruction.iteration_mode != PcoIterationMode::kPixel ||
          instruction.perspective != 0 || instruction.saturate != 0 ||
          instruction.immediate != 0 ||
          !HasDefaultControlFields(instruction) || instruction.end_group != 0 ||
          instruction.source.bank != PcoRegisterBank::kTemporary ||
          coordinate_mask == 0 ||
          instruction.source1.bank != PcoRegisterBank::kShared ||
          instruction.source1.index % kPcoTextureDescriptorDwordCount != 0 ||
          instruction.source1.index / kPcoTextureDescriptorDwordCount >=
              kPcoMaximumTextureDescriptorSets ||
          static_cast<std::size_t>(instruction.source1.index) +
                  kPcoTextureDescriptorDwordCount >
              kPcoMaximumVertexSharedCount ||
          instruction.source2.bank != PcoRegisterBank::kShared ||
          instruction.source2.index != instruction.source1.index + 8U ||
          static_cast<std::size_t>(instruction.output_index) +
                  kPcoPixelOutputCount >
              kPcoTemporaryCount) {
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex 2D texture request");
      }
      continue;
    }
    if (instruction.opcode == PcoOpcode::kWaitDataFence) {
      if (instruction.target != PcoWriteTarget::kNone ||
          instruction.source_count != 0 || instruction.repeat_count != 1 ||
          instruction.output_index != 0 ||
          !HasDefaultNonFitrpFields(instruction) ||
          !HasCanonicalUnusedSources(instruction) ||
          instruction.immediate != 0 ||
          !HasDefaultControlFields(instruction) || instruction.end_group != 0) {
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex DRC0 WDF");
      }
      continue;
    }
    if (!HasCanonicalGenericNonFitrpFields(instruction) ||
        !HasCanonicalUnusedSources(instruction) ||
        !HasDefaultControlFields(instruction) ||
        (instruction.opcode != PcoOpcode::kMoveImmediate &&
         instruction.immediate != 0)) {
      DecodeError(instruction.binary_offset,
                  "generic vertex instruction has noncanonical metadata");
    }
    const auto require_source_range = [&](const PcoRegisterRef &source,
                                          std::uint8_t count) {
      if (source.bank == PcoRegisterBank::kSpecial) {
        if (count != 1 || !IsSupportedSpecialConstant(source.index))
          DecodeError(instruction.binary_offset,
                      "invalid generic vertex special source");
      } else if (source.bank == PcoRegisterBank::kTemporary) {
        if (static_cast<std::size_t>(source.index) + count >
            kPcoTemporaryCount)
          DecodeError(instruction.binary_offset,
                      "generic vertex TEMP source is out of bounds");
      } else if (source.bank == PcoRegisterBank::kVertexInput) {
        if (static_cast<std::size_t>(source.index) + count >
            kPcoVertexInputCount)
          DecodeError(instruction.binary_offset,
                      "generic vertex-input source is out of bounds");
      } else if (source.bank == PcoRegisterBank::kShared) {
        if (static_cast<std::size_t>(source.index) + count >
            kPcoMaximumVertexSharedCount)
          DecodeError(instruction.binary_offset,
                      "generic vertex shared source is out of bounds");
      } else {
        DecodeError(instruction.binary_offset,
                    "generic vertex source bank is unsupported");
      }
    };
    if (instruction.source_count >= 1)
      require_source_range(instruction.source, instruction.repeat_count);
    if (instruction.source_count >= 2)
      require_source_range(instruction.source1, instruction.repeat_count);
    if (instruction.source_count == 3)
      require_source_range(instruction.source2, instruction.repeat_count);

    switch (instruction.opcode) {
    case PcoOpcode::kMoveImmediate:
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 0 || instruction.repeat_count != 1)
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex immediate move");
      break;
    case PcoOpcode::kMoveBypass:
    case PcoOpcode::kFloatNegate:
    case PcoOpcode::kFloatAbs:
    case PcoOpcode::kReciprocal:
    case PcoOpcode::kReciprocalSquareRoot:
    case PcoOpcode::kFloatLog2:
    case PcoOpcode::kFloatExp2:
    case PcoOpcode::kFloatPackHalfRtne:
    case PcoOpcode::kFloatPackHalfRtz:
    case PcoOpcode::kFloatUnpackHalf:
    case PcoOpcode::kUnpackUnsignedToFloat:
    case PcoOpcode::kUnpackSignedToFloat:
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 1 || instruction.repeat_count != 1)
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex unary ALU operation");
      break;
    case PcoOpcode::kFloatAdd:
    case PcoOpcode::kFloatAddNegateSource0:
    case PcoOpcode::kFloatMultiply:
    case PcoOpcode::kFloatMin:
    case PcoOpcode::kFloatMax:
    case PcoOpcode::kFloatEqual:
    case PcoOpcode::kFloatGreaterEqual:
    case PcoOpcode::kFloatLess:
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 2 || instruction.repeat_count != 1)
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex binary ALU operation");
      break;
    case PcoOpcode::kFloatMad:
    case PcoOpcode::kFloatMadNegateSource2:
    case PcoOpcode::kFloatMadNegateSource0:
    case PcoOpcode::kFloatMadNegateSource0Source2:
    case PcoOpcode::kConditionalSelect:
    case PcoOpcode::kConditionalSelectNegateTrue:
    case PcoOpcode::kConditionalSelectGreaterZero:
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 3 || instruction.repeat_count != 1)
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex ternary ALU operation");
      break;
    case PcoOpcode::kUvsWrite:
    case PcoOpcode::kUvsWriteEmitEndTask:
      if (instruction.target != PcoWriteTarget::kVertexOutput ||
          instruction.source_count != 1 ||
          static_cast<std::size_t>(instruction.output_index) +
                  instruction.repeat_count >
              kPcoVertexOutputCount)
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex UVSW write");
      break;
    case PcoOpcode::kUvsEmitEndTask:
      if (instruction.target != PcoWriteTarget::kNone ||
          instruction.source_count != 0 || instruction.repeat_count != 1)
        DecodeError(instruction.binary_offset,
                    "invalid generic vertex end-task operation");
      break;
    default:
      DecodeError(instruction.binary_offset,
                  "vertex TEMP program is outside the public generic subset");
    }
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
      MatchesFillTexNearestFragmentProfile(instructions) ||
      MatchesConditionalsProfile(ShaderStage::kFragment, instructions))
    return;
  std::uint64_t written_mask = 0;
  bool request_pending = false;
  std::uint16_t pending_output = 0;
  std::uint8_t pending_components = 0;
  for (const PcoInstruction &instruction : instructions) {
    const auto require_source = [&](const PcoRegisterRef &source) {
      if (source.bank == PcoRegisterBank::kSpecial) {
        if (!IsSupportedSpecialConstant(source.index))
          DecodeError(instruction.binary_offset,
                      "invalid generic fragment special source");
        return;
      }
      if (source.bank == PcoRegisterBank::kTemporary) {
        if (source.index >= kPcoTemporaryCount ||
            (written_mask & (UINT64_C(1) << source.index)) == 0)
          DecodeError(instruction.binary_offset,
                      "generic fragment TEMP is read before write");
        return;
      }
      if (source.bank == PcoRegisterBank::kShared) {
        if (source.index >= kPcoMaximumFragmentSharedCount)
          DecodeError(instruction.binary_offset,
                      "generic fragment shared source is out of bounds");
        return;
      }
      if (source.bank == PcoRegisterBank::kCoefficient) {
        if (source.index >= kPcoMaximumVaryingCoefficientCount)
          DecodeError(instruction.binary_offset,
                      "generic fragment coefficient source is out of bounds");
        return;
      }
      DecodeError(instruction.binary_offset,
                  "generic fragment source bank is unsupported");
    };

    if (instruction.opcode == PcoOpcode::kFloatInterpolatePerspective) {
      if (request_pending || instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 2 || instruction.repeat_count != 1 ||
          instruction.component_count < 1 || instruction.component_count > 4 ||
          instruction.source.bank != PcoRegisterBank::kCoefficient ||
          instruction.source1.bank != PcoRegisterBank::kCoefficient ||
          instruction.source1.index != 0 ||
          !IsDefaultUnusedRegister(instruction.source2) ||
          instruction.data_request != 0 ||
          (instruction.iteration_mode != PcoIterationMode::kPixel &&
           instruction.iteration_mode != PcoIterationMode::kCentroid) ||
          instruction.perspective != 1 || instruction.saturate != 0 ||
          instruction.immediate != 0 ||
          !HasDefaultControlFields(instruction) ||
          static_cast<std::size_t>(instruction.source.index) +
                  instruction.component_count * 4U >
              kPcoMaximumVaryingCoefficientCount ||
          static_cast<std::size_t>(instruction.output_index) +
                  instruction.component_count >
              kPcoTemporaryCount) {
        DecodeError(instruction.binary_offset, "invalid generic FITRP request");
      }
      request_pending = true;
      pending_output = instruction.output_index;
      pending_components = instruction.component_count;
      continue;
    }
    if (instruction.opcode == PcoOpcode::kTextureSample) {
      const std::size_t coordinate_base = instruction.source.index;
      const std::uint64_t coordinate_mask =
          coordinate_base + 2U <= kPcoTemporaryCount
              ? (UINT64_C(3) << coordinate_base)
              : UINT64_C(0);
      if (request_pending || instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 3 || instruction.repeat_count != 1 ||
          instruction.component_count != 4 || instruction.data_request != 0 ||
          instruction.iteration_mode != PcoIterationMode::kPixel ||
          instruction.perspective != 0 || instruction.saturate != 0 ||
          instruction.immediate != 0 ||
          !HasDefaultControlFields(instruction) ||
          instruction.source.bank != PcoRegisterBank::kTemporary ||
          coordinate_mask == 0 ||
          (written_mask & coordinate_mask) != coordinate_mask ||
          instruction.source1.bank != PcoRegisterBank::kShared ||
          static_cast<std::size_t>(instruction.source1.index) + 4U >
              kPcoMaximumSharedCount ||
          instruction.source2.bank != PcoRegisterBank::kShared ||
          static_cast<std::size_t>(instruction.source2.index) + 4U >
              kPcoMaximumSharedCount ||
          instruction.source1.index % kPcoTextureDescriptorDwordCount != 0 ||
          instruction.source2.index != instruction.source1.index + 8U ||
          instruction.source1.index / kPcoTextureDescriptorDwordCount >=
              kPcoMaximumTextureDescriptorSets ||
          static_cast<std::size_t>(instruction.source1.index) +
                  kPcoTextureDescriptorDwordCount >
              kPcoMaximumFragmentSharedCount ||
          static_cast<std::size_t>(instruction.output_index) + 4U >
              kPcoTemporaryCount) {
        DecodeError(instruction.binary_offset,
                    "invalid generic 2D texture request");
      }
      request_pending = true;
      pending_output = instruction.output_index;
      pending_components = instruction.component_count;
      continue;
    }
    if (instruction.opcode == PcoOpcode::kWaitDataFence) {
      if (!request_pending || instruction.target != PcoWriteTarget::kNone ||
          instruction.source_count != 0 || instruction.repeat_count != 1 ||
          instruction.output_index != 0 ||
          !HasDefaultNonFitrpFields(instruction) ||
          !HasCanonicalUnusedSources(instruction) ||
          instruction.immediate != 0 ||
          !HasDefaultControlFields(instruction))
        DecodeError(instruction.binary_offset, "invalid generic DRC0 WDF");
      for (std::uint8_t component = 0; component < pending_components;
           ++component)
        written_mask |= UINT64_C(1) << (pending_output + component);
      request_pending = false;
      continue;
    }

    if (!HasCanonicalGenericNonFitrpFields(instruction) ||
        !HasCanonicalUnusedSources(instruction) ||
        !HasDefaultControlFields(instruction) ||
        (instruction.opcode != PcoOpcode::kMoveImmediate &&
         instruction.immediate != 0)) {
      DecodeError(instruction.binary_offset,
                  "generic fragment instruction has noncanonical metadata");
    }

    if (instruction.source_count >= 1)
      require_source(instruction.source);
    if (instruction.source_count >= 2)
      require_source(instruction.source1);
    if (instruction.source_count == 3)
      require_source(instruction.source2);

    bool writes_temporary = false;
    switch (instruction.opcode) {
    case PcoOpcode::kMoveImmediate:
      writes_temporary = instruction.target == PcoWriteTarget::kTemporary &&
                         instruction.source_count == 0;
      break;
    case PcoOpcode::kMoveBypass:
      if (instruction.target == PcoWriteTarget::kPixelOutput) {
        if (instruction.source_count != 1 || instruction.repeat_count != 1 ||
            instruction.output_index >= kPcoPixelOutputCount)
          DecodeError(instruction.binary_offset,
                      "invalid generic fragment PIXOUT move");
        continue;
      }
      writes_temporary = instruction.target == PcoWriteTarget::kTemporary &&
                         instruction.source_count == 1;
      break;
    case PcoOpcode::kFloatNegate:
    case PcoOpcode::kFloatAbs:
    case PcoOpcode::kReciprocal:
    case PcoOpcode::kReciprocalSquareRoot:
    case PcoOpcode::kFloatLog2:
    case PcoOpcode::kFloatExp2:
    case PcoOpcode::kFloatPackHalfRtne:
    case PcoOpcode::kFloatPackHalfRtz:
    case PcoOpcode::kFloatUnpackHalf:
    case PcoOpcode::kUnpackUnsignedToFloat:
    case PcoOpcode::kUnpackSignedToFloat:
      writes_temporary = instruction.target == PcoWriteTarget::kTemporary &&
                         instruction.source_count == 1;
      break;
    case PcoOpcode::kFloatAdd:
    case PcoOpcode::kFloatAddNegateSource0:
    case PcoOpcode::kFloatMultiply:
    case PcoOpcode::kFloatMin:
    case PcoOpcode::kFloatMax:
    case PcoOpcode::kFloatEqual:
    case PcoOpcode::kFloatGreaterEqual:
    case PcoOpcode::kFloatLess:
    case PcoOpcode::kBitwiseAnd:
    case PcoOpcode::kBitwiseXnor:
      writes_temporary = instruction.target == PcoWriteTarget::kTemporary &&
                         instruction.source_count == 2;
      break;
    case PcoOpcode::kFloatMad:
    case PcoOpcode::kFloatMadNegateSource2:
    case PcoOpcode::kFloatMadNegateSource0:
    case PcoOpcode::kFloatMadNegateSource0Source2:
    case PcoOpcode::kConditionalSelect:
    case PcoOpcode::kConditionalSelectNegateTrue:
    case PcoOpcode::kConditionalSelectGreaterZero:
      writes_temporary = instruction.target == PcoWriteTarget::kTemporary &&
                         instruction.source_count == 3;
      break;
    default:
      DecodeError(instruction.binary_offset,
                  "fragment program is outside the public generic subset");
    }
    if (!writes_temporary || instruction.repeat_count != 1 ||
        instruction.output_index >= kPcoTemporaryCount)
      DecodeError(instruction.binary_offset,
                  "invalid generic fragment ALU destination");
    written_mask |= UINT64_C(1) << instruction.output_index;
  }
  if (request_pending)
    DecodeError(instructions.back().binary_offset,
                "fragment program ends with an unresolved DRC0 request");
}

std::uint32_t ReadSource(const PcoRegisterRef &source,
                         const std::vector<std::uint32_t> &vertex_inputs,
                         const std::array<std::uint32_t, kPcoTemporaryCount>
                             &temporaries,
                         std::uint64_t temporary_written_mask,
                         std::uint8_t repeat_index, ShaderStage stage) {
  const std::size_t index =
      static_cast<std::size_t>(source.index) + repeat_index;
  switch (source.bank) {
  case PcoRegisterBank::kSpecial:
    if (repeat_index != 0)
      ExecuteError("special constants cannot be register-range repeated");
    if (source.index <= 31)
      return source.index;
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
    if (source.index == kSpecialConstantOneOver256)
      return UINT32_C(0x3b800000);
    if (source.index == kSpecialConstantOneThird)
      return UINT32_C(0x3eaaaaab);
    if (source.index == kSpecialConstantOneSixth)
      return UINT32_C(0x3e2aaaab);
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
    if ((temporary_written_mask & (UINT64_C(1) << index)) == 0)
      ExecuteError("temporary register was read before it was written");
    return temporaries[index];
  default:
    ExecuteError("source register bank is outside this PCO subset");
  }
}

/*
 * Integer to binary32 with the round-to-nearest-even the hardware and every
 * public driver use.  A 32-bit integer does not always fit the 24-bit
 * significand, so the rounding is explicit rather than left to a host cast.
 */
std::uint32_t FloatFromUnsigned(std::uint32_t value) {
  if (value == 0)
    return UINT32_C(0);
  std::uint32_t exponent = 31;
  while ((value & (UINT32_C(1) << exponent)) == 0)
    --exponent;
  std::uint32_t significand;
  if (exponent <= 23) {
    significand = (value << (23 - exponent)) & UINT32_C(0x7fffff);
  } else {
    const std::uint32_t shift = exponent - 23;
    significand = (value >> shift) & UINT32_C(0x7fffff);
    const std::uint32_t discarded = value & ((UINT32_C(1) << shift) - 1);
    const std::uint32_t halfway = UINT32_C(1) << (shift - 1);
    const bool round_up =
        discarded > halfway ||
        (discarded == halfway && (significand & UINT32_C(1)) != 0);
    if (round_up) {
      ++significand;
      if (significand > UINT32_C(0x7fffff)) {
        significand = 0;
        ++exponent;
      }
    }
  }
  return ((exponent + 127U) << 23U) | significand;
}

std::uint32_t FloatFromSigned(std::int32_t value) {
  if (value >= 0)
    return FloatFromUnsigned(static_cast<std::uint32_t>(value));
  const std::uint32_t magnitude =
      value == std::numeric_limits<std::int32_t>::min()
          ? UINT32_C(0x80000000)
          : static_cast<std::uint32_t>(-value);
  return FloatFromUnsigned(magnitude) | UINT32_C(0x80000000);
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
  const std::uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
  const std::uint32_t exponent_field =
      (bits >> 23U) & UINT32_C(0xff);
  const std::uint32_t fraction = bits & UINT32_C(0x007fffff);
  const auto round_right_to_nearest_even = [](std::uint32_t value,
                                               std::uint32_t distance) {
    const std::uint32_t quotient = value >> distance;
    const std::uint32_t remainder_mask =
        (UINT32_C(1) << distance) - UINT32_C(1);
    const std::uint32_t remainder = value & remainder_mask;
    const std::uint32_t halfway = UINT32_C(1) << (distance - 1U);
    return quotient +
           (remainder > halfway ||
                    (remainder == halfway && (quotient & UINT32_C(1)) != 0)
                ? UINT32_C(1)
                : UINT32_C(0));
  };

  if (exponent_field == UINT32_C(0xff)) {
    /* Preserve infinity and produce a quiet half NaN for every binary32 NaN. */
    return static_cast<std::uint16_t>(
        sign | UINT32_C(0x7c00) |
        (fraction != 0 ? UINT32_C(0x0200) : UINT32_C(0)));
  }
  if (exponent_field == 0) {
    /* Every binary32 subnormal is below half's minimum rounding midpoint. */
    return static_cast<std::uint16_t>(sign);
  }

  std::int32_t exponent =
      static_cast<std::int32_t>(exponent_field) - 127;
  if (exponent > 15)
    return static_cast<std::uint16_t>(sign | UINT32_C(0x7c00));
  if (exponent < -25)
    return static_cast<std::uint16_t>(sign);

  const std::uint32_t significand = UINT32_C(0x00800000) | fraction;
  if (exponent < -14) {
    const std::uint32_t distance =
        static_cast<std::uint32_t>(-exponent - 1);
    const std::uint32_t half_fraction =
        round_right_to_nearest_even(significand, distance);
    /* Rounding the largest subnormal can carry into the minimum normal. */
    return static_cast<std::uint16_t>(sign | half_fraction);
  }

  std::uint32_t half_significand =
      round_right_to_nearest_even(significand, 13U);
  if (half_significand == UINT32_C(0x0800)) {
    half_significand = UINT32_C(0x0400);
    ++exponent;
    if (exponent > 15)
      return static_cast<std::uint16_t>(sign | UINT32_C(0x7c00));
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent + 15) << 10U) |
      (half_significand & UINT32_C(0x03ff)));
}

std::uint16_t FloatToHalfRtz(std::uint32_t bits) {
  const std::uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
  const std::uint32_t exponent_field =
      (bits >> 23U) & UINT32_C(0xff);
  const std::uint32_t fraction = bits & UINT32_C(0x007fffff);

  if (exponent_field == UINT32_C(0xff)) {
    /* Rounding does not change infinity.  Match the RTNE path's deterministic
     * quiet-NaN representation so non-numeric payload policy stays stable. */
    return static_cast<std::uint16_t>(
        sign | UINT32_C(0x7c00) |
        (fraction != 0 ? UINT32_C(0x0200) : UINT32_C(0)));
  }
  if (exponent_field == 0)
    return static_cast<std::uint16_t>(sign);

  const std::int32_t exponent =
      static_cast<std::int32_t>(exponent_field) - 127;
  if (exponent > 15) {
    /* IEEE roundTowardZero overflow returns the largest finite value rather
     * than infinity, for either sign. */
    return static_cast<std::uint16_t>(sign | UINT32_C(0x7bff));
  }
  if (exponent < -24)
    return static_cast<std::uint16_t>(sign);

  const std::uint32_t significand = UINT32_C(0x00800000) | fraction;
  if (exponent < -14) {
    const std::uint32_t distance =
        static_cast<std::uint32_t>(-exponent - 1);
    return static_cast<std::uint16_t>(sign | (significand >> distance));
  }

  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent + 15) << 10U) |
      ((significand >> 13U) & UINT32_C(0x03ff)));
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
  const auto exponent_field = [](std::uint32_t bits) {
    return (bits >> 23U) & UINT32_C(0xff);
  };
  const auto native_add = [](std::uint32_t lhs_bits,
                             std::uint32_t rhs_bits) {
    float lhs = 0.0F;
    float rhs = 0.0F;
    std::memcpy(&lhs, &lhs_bits, sizeof(lhs));
    std::memcpy(&rhs, &rhs_bits, sizeof(rhs));
    const float value = lhs + rhs;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
        (bits & UINT32_C(0x007fffff)) != 0) {
      return UINT32_C(0x7fc00000);
    }
    return bits;
  };

  /* Keep the integer RNE implementation below for the normal finite path.
   * IEEE special values, subnormal operands, and cancellation at the minimum
   * normal exponent use the host's binary32 operation.  The compiler target
   * is IEEE-754 binary32/RNE, and this path is required for legal values
   * produced by FEXP in Mesa's lowered fpow sequence. */
  const std::uint32_t left_exponent = exponent_field(left_bits);
  const std::uint32_t right_exponent = exponent_field(right_bits);
  if (left_exponent == 0 || right_exponent == 0 ||
      left_exponent == UINT32_C(0xff) ||
      right_exponent == UINT32_C(0xff) || left_exponent == 1 ||
      right_exponent == 1) {
    return native_add(left_bits, right_bits);
  }

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
  if (exponent >= UINT32_C(0xff)) {
    return (result_sign ? UINT32_C(0xff800000)
                        : UINT32_C(0x7f800000));
  }

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
    if (exponent >= UINT32_C(0xff)) {
      return (result_sign ? UINT32_C(0xff800000)
                          : UINT32_C(0x7f800000));
    }
  }
  if (rounded_significand < UINT32_C(0x00800000)) {
    ExecuteError("FADD subnormal-result policy is not in the public ISA gate");
  }

  return (result_sign ? UINT32_C(0x80000000) : UINT32_C(0)) |
         (exponent << 23U) |
         (rounded_significand & UINT32_C(0x007fffff));
}

std::uint32_t FloatSaturateBits(std::uint32_t value_bits) {
  constexpr std::uint32_t kSign = UINT32_C(0x80000000);
  constexpr std::uint32_t kExponent = UINT32_C(0x7f800000);
  constexpr std::uint32_t kFraction = UINT32_C(0x007fffff);
  constexpr std::uint32_t kOne = UINT32_C(0x3f800000);
  /* Mesa NIR fsat is minimumNumber(maximumNumber(x,+0),1).  IEEE-754-2019
   * number operations choose +0 for NaN/-0 and the finite operand for one
   * NaN input.  Positive non-NaN bit patterns preserve float ordering. */
  if ((value_bits & kExponent) == kExponent &&
      (value_bits & kFraction) != 0) {
    return UINT32_C(0);
  }
  if ((value_bits & kSign) != 0)
    return UINT32_C(0);
  if (value_bits > kOne)
    return kOne;
  return value_bits;
}

std::uint32_t FloatFloorBits(std::uint32_t value_bits) {
  (void)DecodeFaddOperand(value_bits);
  float value = 0.0F;
  std::memcpy(&value, &value_bits, sizeof(value));
  const float floored = std::floor(value);
  std::uint32_t result = 0;
  std::memcpy(&result, &floored, sizeof(result));
  return result;
}

std::uint32_t FloatGreaterEqualBits(std::uint32_t left_bits,
                                    std::uint32_t right_bits) {
  const auto is_nan = [](std::uint32_t bits) {
    return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
           (bits & UINT32_C(0x007fffff)) != 0;
  };
  /* TST.F32.GE is ordered: either NaN makes the predicate false.  Preserve
   * the original binary32 values so signed zero and infinity follow IEEE
   * ordering without passing through the arithmetic-only FADD gate. */
  if (is_nan(left_bits) || is_nan(right_bits))
    return UINT32_C(0);
  float left = 0.0F;
  float right = 0.0F;
  std::memcpy(&left, &left_bits, sizeof(left));
  std::memcpy(&right, &right_bits, sizeof(right));
  return left >= right ? UINT32_MAX : UINT32_C(0);
}

std::uint32_t FloatGreaterEqualResultBits(std::uint32_t left_bits,
                                          std::uint32_t right_bits,
                                          std::uint8_t float_one_result) {
  const std::uint32_t predicate =
      FloatGreaterEqualBits(left_bits, right_bits);
  if (float_one_result == 0)
    return predicate;
  return predicate != 0 ? UINT32_C(0x3f800000) : UINT32_C(0);
}

std::uint32_t FloatEqualBits(std::uint32_t left_bits,
                             std::uint32_t right_bits) {
  (void)DecodeFaddOperand(left_bits);
  (void)DecodeFaddOperand(right_bits);
  float left = 0.0F;
  float right = 0.0F;
  std::memcpy(&left, &left_bits, sizeof(left));
  std::memcpy(&right, &right_bits, sizeof(right));
  return left == right ? UINT32_MAX : UINT32_C(0);
}

std::uint32_t FloatLessBits(std::uint32_t left_bits,
                            std::uint32_t right_bits) {
  const auto is_nan = [](std::uint32_t bits) {
    return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
           (bits & UINT32_C(0x007fffff)) != 0;
  };
  /* TST.F32.L is ordered: either NaN makes the predicate false.  Host IEEE
   * comparison then gives the required signed-zero and infinity ordering
   * without changing either operand's binary32 value. */
  if (is_nan(left_bits) || is_nan(right_bits))
    return UINT32_C(0);
  float left = 0.0F;
  float right = 0.0F;
  std::memcpy(&left, &left_bits, sizeof(left));
  std::memcpy(&right, &right_bits, sizeof(right));
  return left < right ? UINT32_MAX : UINT32_C(0);
}

bool FloatGreaterZero(std::uint32_t value_bits) {
  const Binary32Operand operand = DecodeFaddOperand(value_bits);
  return !operand.sign && !operand.zero;
}


std::uint32_t FloatMadBits(std::uint32_t a_bits, std::uint32_t b_bits,
                           std::uint32_t c_bits) {
  float a_val = 0.0f, b_val = 0.0f, c_val = 0.0f;
  std::memcpy(&a_val, &a_bits, sizeof(a_val));
  std::memcpy(&b_val, &b_bits, sizeof(b_val));
  std::memcpy(&c_val, &c_bits, sizeof(c_val));
  const float result_val = std::fma(a_val, b_val, c_val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  if ((result_bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
      (result_bits & UINT32_C(0x007fffff)) != 0) {
    return UINT32_C(0x7fc00000);
  }
  return result_bits;
}

std::uint32_t FloatMinBits(std::uint32_t left_bits, std::uint32_t right_bits) {
  float left_val = 0.0f, right_val = 0.0f;
  std::memcpy(&left_val, &left_bits, sizeof(left_val));
  std::memcpy(&right_val, &right_bits, sizeof(right_val));
  /* The decoded PCO form is ordered (left < right) and MOVC selects the
   * left source only when that predicate is true.  False—including NaN and
   * equal +/-0—selects the unmodified right source. */
  return left_val < right_val ? left_bits : right_bits;
}

std::uint32_t FloatMaxBits(std::uint32_t left_bits, std::uint32_t right_bits) {
  float left_val = 0.0f, right_val = 0.0f;
  std::memcpy(&left_val, &left_bits, sizeof(left_val));
  std::memcpy(&right_val, &right_bits, sizeof(right_val));
  /* Public FMAX has the symmetric ordered TST.G/MOVC routing: true selects
   * left, while false—including NaN and equal signed zero—selects right. */
  return left_val > right_val ? left_bits : right_bits;
}

std::uint32_t ReciprocalBits(std::uint32_t val_bits) {
  constexpr std::uint32_t kExponentMask = UINT32_C(0x7f800000);
  constexpr std::uint32_t kFractionMask = UINT32_C(0x007fffff);
  constexpr std::uint32_t kCanonicalQuietNan = UINT32_C(0x7fc00000);
  if ((val_bits & kExponentMask) == kExponentMask) {
    /* The Refract fsqrt lowering is FRSQ followed by FRCP.  A negative
     * finite radicand therefore reaches this operation as a quiet NaN and
     * must remain NaN until the following CSEL.GZ discards that path.
     * Reciprocal infinity is the correspondingly signed zero. */
    if ((val_bits & kFractionMask) != 0)
      return kCanonicalQuietNan;
    return val_bits & UINT32_C(0x80000000);
  }
  const Binary32Operand operand = DecodeFaddOperand(val_bits);
  if (operand.zero) {
    /* Public PCO FRCP follows IEEE binary32 division for signed zero:
     * 1/+0 -> +infinity and 1/-0 -> -infinity.  Handle this class explicitly
     * so the otherwise strict NaN/Inf gate cannot confuse a legitimate RCP
     * result with an unmodeled input or arithmetic overflow policy. */
    return (val_bits & UINT32_C(0x80000000)) | UINT32_C(0x7f800000);
  }
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  const float result_val = 1.0f / val;
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  (void)DecodeFaddOperand(result_bits);
  return result_bits;
}

std::uint32_t ReciprocalSquareRootBits(std::uint32_t val_bits) {
  constexpr std::uint32_t kExponentMask = UINT32_C(0x7f800000);
  constexpr std::uint32_t kFractionMask = UINT32_C(0x007fffff);
  constexpr std::uint32_t kCanonicalQuietNan = UINT32_C(0x7fc00000);
  if ((val_bits & kExponentMask) == kExponentMask) {
    /* Public PCO FRSQ follows binary32 1/sqrt(x).  In particular,
     * +infinity is a valid normalized-vector intermediate and maps to +0.
     * A NaN or -infinity has no real reciprocal square root; keep the ISS's
     * deterministic quiet-NaN policy for those classes. */
    if ((val_bits & kFractionMask) != 0 ||
        (val_bits & UINT32_C(0x80000000)) != 0) {
      return kCanonicalQuietNan;
    }
    return UINT32_C(0);
  }
  const Binary32Operand operand = DecodeFaddOperand(val_bits);
  if (operand.zero) {
    /* PCO FRSQ follows the binary32 1/sqrt operation for signed zero.
     * sqrt preserves the zero sign, so the reciprocal is the correspondingly
     * signed infinity. */
    return (val_bits & UINT32_C(0x80000000)) | UINT32_C(0x7f800000);
  }
  if (operand.sign) {
    /* A finite negative radicand has no real result.  Mesa's fsqrt lowering
     * relies on the canonical quiet NaN being carried until a later fcsel
     * discards the total-internal-reflection path. */
    return kCanonicalQuietNan;
  }
  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  const float result_val = 1.0f / std::sqrt(val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  (void)DecodeFaddOperand(result_bits);
  return result_bits;
}

std::uint32_t FloatLog2Bits(std::uint32_t val_bits) {
  constexpr std::uint32_t kPositiveInfinity = UINT32_C(0x7f800000);
  constexpr std::uint32_t kNegativeInfinity = UINT32_C(0xff800000);
  constexpr std::uint32_t kCanonicalQuietNan = UINT32_C(0x7fc00000);
  const bool sign = (val_bits & UINT32_C(0x80000000)) != 0;
  const std::uint32_t exponent =
      (val_bits >> 23U) & UINT32_C(0xff);
  const std::uint32_t fraction = val_bits & UINT32_C(0x007fffff);

  /* PCO FLOG implements NIR flog2, whose scalar reference operation is
   * log2f().  In particular, either signed zero maps to -infinity.  This is
   * required by Mesa's public fpow lowering:
   *
   *   exp2(log2(max(x, 0.0)) * positive_exponent)
   *
   * A zero base therefore remains a valid zero rather than an execution
   * fault.  Negative inputs follow log2f() and produce a quiet NaN. */
  if (exponent == UINT32_C(0xff)) {
    if (fraction != 0 || sign)
      return kCanonicalQuietNan;
    return kPositiveInfinity;
  }
  if (exponent == 0) {
    if (fraction == 0)
      return kNegativeInfinity;
  }
  if (sign)
    return kCanonicalQuietNan;

  float val = 0.0f;
  std::memcpy(&val, &val_bits, sizeof(val));
  const float result_val = std::log2(val);
  std::uint32_t result_bits = 0;
  std::memcpy(&result_bits, &result_val, sizeof(result_bits));
  return result_bits;
}

std::uint32_t FloatExp2Bits(std::uint32_t val_bits) {
  constexpr std::uint32_t kPositiveInfinity = UINT32_C(0x7f800000);
  constexpr std::uint32_t kCanonicalQuietNan = UINT32_C(0x7fc00000);
  const bool sign = (val_bits & UINT32_C(0x80000000)) != 0;
  const std::uint32_t exponent =
      (val_bits >> 23U) & UINT32_C(0xff);
  const std::uint32_t fraction = val_bits & UINT32_C(0x007fffff);
  if (exponent == UINT32_C(0xff)) {
    if (fraction != 0)
      return kCanonicalQuietNan;
    return sign ? UINT32_C(0) : kPositiveInfinity;
  }
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
  constexpr std::uint32_t kPositiveInfinity = UINT32_C(0x7f800000);
  constexpr std::uint32_t kCanonicalQuietNan = UINT32_C(0x7fc00000);
  const auto exponent_field = [](std::uint32_t bits) {
    return (bits >> 23U) & UINT32_C(0xff);
  };
  const auto fraction = [](std::uint32_t bits) {
    return bits & UINT32_C(0x007fffff);
  };
  const auto is_zero = [&](std::uint32_t bits) {
    return exponent_field(bits) == 0 && fraction(bits) == 0;
  };
  const bool left_special = exponent_field(left_bits) == UINT32_C(0xff);
  const bool right_special = exponent_field(right_bits) == UINT32_C(0xff);
  if (left_special || right_special) {
    const bool left_nan = left_special && fraction(left_bits) != 0;
    const bool right_nan = right_special && fraction(right_bits) != 0;
    if (left_nan || right_nan ||
        (left_special && is_zero(right_bits)) ||
        (right_special && is_zero(left_bits))) {
      return kCanonicalQuietNan;
    }
    const bool sign = ((left_bits ^ right_bits) & UINT32_C(0x80000000)) != 0;
    return (sign ? UINT32_C(0x80000000) : UINT32_C(0)) |
           kPositiveInfinity;
  }

  if ((exponent_field(left_bits) == 0 && fraction(left_bits) != 0) ||
      (exponent_field(right_bits) == 0 && fraction(right_bits) != 0)) {
    float left_value = 0.0F;
    float right_value = 0.0F;
    std::memcpy(&left_value, &left_bits, sizeof(left_value));
    std::memcpy(&right_value, &right_bits, sizeof(right_value));
    const float result_value = left_value * right_value;
    std::uint32_t result_bits = 0;
    std::memcpy(&result_bits, &result_value, sizeof(result_bits));
    return result_bits;
  }

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
  if (exponent <= 0 || exponent >= 255) {
    float left_value = 0.0F;
    float right_value = 0.0F;
    std::memcpy(&left_value, &left_bits, sizeof(left_value));
    std::memcpy(&right_value, &right_bits, sizeof(right_value));
    const float result_value = left_value * right_value;
    std::uint32_t result_bits = 0;
    std::memcpy(&result_bits, &result_value, sizeof(result_bits));
    return result_bits;
  }
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
  if (context.coefficient_count < 4 ||
      context.coefficient_count > kPcoMaximumVaryingCoefficientCount ||
      (context.coefficient_count & 3U) != 0 ||
      coefficient_base + 3 >= context.coefficient_count) {
    ExecuteError("FITRP coefficient range is absent or truncated");
  }
  /* llvmpipe's public setup/interpolation ABI evaluates a plane as two
   * ordered llvm.fmuladd operations:
   *
   *   value = fma(A, x, C);
   *   value = fma(B, y, value);
   *
   * The nesting is observable at nearest-texel and binary16 boundaries, so
   * do not reassociate the two fused operations. */
  return FloatMadBits(
      context.coefficients[coefficient_base + 1], context.sample_y,
      FloatMadBits(context.coefficients[coefficient_base], context.sample_x,
                   context.coefficients[coefficient_base + 2]));
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
    if (!HasCanonicalGenericSource0Modifier(instruction))
      ExecuteError("decoded source0 modifier is not canonical");
    if (!HasCanonicalGenericComparisonResult(instruction))
      ExecuteError("decoded comparison-result encoding is not canonical");
    if (!HasCanonicalGenericSaturate(instruction))
      ExecuteError("decoded saturate modifier is not canonical");
    if (!HasCanonicalGenericAbsoluteModifiers(instruction))
      ExecuteError("decoded absolute-value modifier is not canonical");
    if (!HasCanonicalLogicalXnorShape(instruction))
      ExecuteError("decoded LOGICAL.XNOR shape is not canonical");
    if (instruction.iteration_mode != PcoIterationMode::kPixel &&
        instruction.iteration_mode != PcoIterationMode::kSample &&
        instruction.iteration_mode != PcoIterationMode::kCentroid)
      ExecuteError("decoded iteration mode is outside the public enum");
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

const std::vector<std::uint8_t> &ConditionalsVertexPcoBinary() {
  return kConditionalsVertexBinary;
}

const std::vector<std::uint8_t> &ConditionalsFragmentPcoBinary() {
  return kConditionalsFragmentBinary;
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
    case PcoOpcode::kInternal:
    case PcoOpcode::kMoveBypass:
    case PcoOpcode::kFloatNegate:
    case PcoOpcode::kFloatAbs:
    case PcoOpcode::kMoveImmediate:
    case PcoOpcode::kFragmentCoordinate:
    case PcoOpcode::kFloatFloor:
    case PcoOpcode::kFloatSubtract:
    case PcoOpcode::kFloatGreaterEqual:
    case PcoOpcode::kFloatEqual:
    case PcoOpcode::kFloatLess:
    case PcoOpcode::kConditionalSelect:
    case PcoOpcode::kConditionalSelectNegateTrue:
    case PcoOpcode::kConditionalSelectGreaterZero:
    case PcoOpcode::kFloatAdd:
    case PcoOpcode::kFloatAddNegateSource0:
    case PcoOpcode::kFloatMultiply:
    case PcoOpcode::kFloatMad:
    case PcoOpcode::kFloatMadNegateSource2:
    case PcoOpcode::kFloatMadNegateSource0:
    case PcoOpcode::kFloatMadNegateSource0Source2:
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
    case PcoOpcode::kBitwiseXnor:
    case PcoOpcode::kFloatSine:
    case PcoOpcode::kFloatCosine:
    case PcoOpcode::kDerivativeX:
    case PcoOpcode::kDerivativeY:
    case PcoOpcode::kPackHalf2x16:
    case PcoOpcode::kUnpackHalf2x16:
    case PcoOpcode::kFloatPackHalfRtne:
    case PcoOpcode::kFloatPackHalfRtz:
    case PcoOpcode::kFloatUnpackHalf:
    case PcoOpcode::kUnpackUnsignedToFloat:
    case PcoOpcode::kUnpackSignedToFloat:
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

  const bool conditionals_vertex =
      stage == ShaderStage::kVertex && binary == kConditionalsVertexBinary;
  const bool conditionals_fragment =
      stage == ShaderStage::kFragment && binary == kConditionalsFragmentBinary;
  if ((stage == ShaderStage::kVertex && binary == kConditionalsFragmentBinary) ||
      (stage == ShaderStage::kFragment && binary == kConditionalsVertexBinary)) {
    DecodeError(0, "conditionals binary was submitted to the wrong stage");
  }
  if (conditionals_vertex || conditionals_fragment) {
    decoded.instructions = BuildConditionalsInstructions(stage, binary);
    decoded.summary.group_count =
        CheckedU32(decoded.instructions.size(), "PCO conditionals group count");
    decoded.summary.instruction_count = decoded.summary.group_count;
    if (conditionals_vertex) {
      decoded.summary.vertex_input_mask = UINT32_C(0x7);
      decoded.summary.vertex_output_mask = UINT64_C(0xf);
      decoded.summary.ends_task = 1;
      ValidateVertexTemporaryProgram(decoded.instructions);
    } else {
      decoded.summary.pixel_output_mask = 0x0f;
      ValidateFragmentProgram(decoded.instructions);
    }
    return decoded;
  }

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
      if (instruction.source_count >= 2)
        include_vertex_source(instruction.source1);
      if (instruction.source_count == 3)
        include_vertex_source(instruction.source2);
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
  const bool conditionals =
      MatchesConditionalsProfile(ShaderStage::kVertex, instructions);
  const bool generic_vertex =
      !conditionals && !MatchesOneAttributeProfile(instructions) &&
      !MatchesTwoAttributeProfile(instructions) &&
      !MatchesFourAttributeProfile(instructions) &&
      !MatchesEightAttributeProfile(instructions) &&
      !MatchesVaryingsOneProfile(instructions) &&
      !MatchesVaryingsTwoProfile(instructions) &&
      !MatchesVaryingsFourProfile(instructions) &&
      !MatchesVaryingsEightProfile(instructions) &&
      !MatchesFillTexNearestVertexProfile(instructions);
  const bool texture_program = std::any_of(
      instructions.begin(), instructions.end(), [](const auto &instruction) {
        return instruction.opcode == PcoOpcode::kTextureSample;
      });
  const bool resuming = context.continuation.valid != 0;
  if (context.texture_response_valid > 1)
    ExecuteError("vertex texture-response validity is not Boolean");
  if (resuming && !texture_program)
    ExecuteError("vertex continuation requires a texture-sampling program");
  if (!resuming && context.texture_response_valid != 0)
    ExecuteError("texture response requires a saved vertex continuation");
  if (resuming && context.texture_response_valid == 0)
    ExecuteError("vertex continuation requires a texture response");

  PcoVertexExecution result;
  std::vector<std::uint32_t> effective_vertex_inputs;
  std::array<std::uint32_t, kPcoMaximumVertexSharedCount> effective_shared{};
  std::uint8_t effective_shared_count = 0;
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  std::uint64_t temporary_written_mask = 0;
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
  }
  if (vertex_input_mask != summary.vertex_input_mask)
    ExecuteError("vertex inputs do not match the decoded summary");

  std::array<std::uint32_t, kPcoPixelOutputCount> pending{};
  bool drc0_pending = false;
  std::uint16_t pending_output_index = 0;
  std::uint8_t pending_component_count = 0;
  std::size_t pc = 0;
  if (!resuming) {
    if (vertex_inputs.size() > kPcoVertexInputCount)
      ExecuteError("vertex input span exceeds the modeled USC file");
    if (context.shared_count > kPcoMaximumVertexSharedCount) {
      ExecuteError("vertex shared-register count exceeds the modeled USC file");
    }
    effective_vertex_inputs = vertex_inputs;
    effective_shared_count = context.shared_count;
    std::copy_n(context.shared_registers.begin(), effective_shared_count,
                effective_shared.begin());
  } else {
    const PcoVertexContinuation &continuation = context.continuation;
    if (continuation.valid != 1 ||
        continuation.program_binary_size != summary.binary_size ||
        continuation.program_instruction_count != summary.instruction_count ||
        continuation.resume_instruction_index == 0 ||
        continuation.resume_instruction_index >= instructions.size() ||
        instructions[continuation.resume_instruction_index].opcode !=
            PcoOpcode::kWaitDataFence ||
        instructions[continuation.resume_instruction_index - 1].opcode !=
            PcoOpcode::kTextureSample) {
      ExecuteError("invalid texture vertex continuation location");
    }
    const PcoInstruction &sample =
        instructions[continuation.resume_instruction_index - 1];
    if (continuation.pending_output_index != sample.output_index ||
        continuation.pending_component_count != sample.component_count ||
        continuation.data_request != sample.data_request ||
        continuation.pending_component_count != kPcoPixelOutputCount ||
        static_cast<std::size_t>(continuation.pending_output_index) +
                continuation.pending_component_count >
            kPcoTemporaryCount ||
        continuation.vertex_input_count > kPcoVertexInputCount ||
        continuation.shared_count > kPcoMaximumVertexSharedCount ||
        continuation.emitted > 1 || continuation.ended_task > 1) {
      ExecuteError("invalid texture vertex continuation state");
    }
    if (!vertex_inputs.empty() &&
        (vertex_inputs.size() != continuation.vertex_input_count ||
         !std::equal(vertex_inputs.begin(), vertex_inputs.end(),
                     continuation.vertex_inputs.begin()))) {
      ExecuteError("resumed vertex inputs differ from the saved lane");
    }
    if (context.shared_count != 0 &&
        (context.shared_count != continuation.shared_count ||
         !std::equal(context.shared_registers.begin(),
                     context.shared_registers.begin() + context.shared_count,
                     continuation.shared_registers.begin()))) {
      ExecuteError("resumed vertex shared registers differ from the saved lane");
    }

    std::uint64_t expected_temporary_mask = 0;
    std::uint64_t expected_output_mask = 0;
    bool expected_request_pending = false;
    std::uint16_t expected_pending_output = 0;
    std::uint8_t expected_pending_components = 0;
    std::uint8_t expected_emitted = 0;
    std::uint8_t expected_ended_task = 0;
    for (std::size_t index = 0;
         index + 1 < continuation.resume_instruction_index; ++index) {
      const PcoInstruction &prior = instructions[index];
      if (prior.opcode == PcoOpcode::kTextureSample) {
        if (expected_request_pending)
          ExecuteError("overlapping DRC0 requests precede vertex continuation");
        expected_request_pending = true;
        expected_pending_output = prior.output_index;
        expected_pending_components = prior.component_count;
        continue;
      }
      if (prior.opcode == PcoOpcode::kWaitDataFence) {
        if (!expected_request_pending)
          ExecuteError("unmatched WDF precedes vertex continuation");
        for (std::uint8_t component = 0;
             component < expected_pending_components; ++component) {
          expected_temporary_mask |=
              UINT64_C(1) << (expected_pending_output + component);
        }
        expected_request_pending = false;
        continue;
      }
      if (prior.target == PcoWriteTarget::kTemporary)
        expected_temporary_mask |= UINT64_C(1) << prior.output_index;
      if (prior.target == PcoWriteTarget::kVertexOutput) {
        for (std::uint8_t repeat = 0; repeat < prior.repeat_count; ++repeat)
          expected_output_mask |= UINT64_C(1)
                                  << (prior.output_index + repeat);
      }
      if (prior.opcode == PcoOpcode::kUvsWriteEmitEndTask ||
          prior.opcode == PcoOpcode::kUvsEmitEndTask) {
        expected_emitted = 1;
        expected_ended_task = 1;
      }
    }
    if (expected_request_pending ||
        continuation.temporary_written_mask != expected_temporary_mask ||
        continuation.output_written_mask != expected_output_mask ||
        continuation.emitted != expected_emitted ||
        continuation.ended_task != expected_ended_task) {
      ExecuteError("texture vertex continuation masks are inconsistent");
    }
    for (std::size_t index = continuation.vertex_input_count;
         index < continuation.vertex_inputs.size(); ++index) {
      if (continuation.vertex_inputs[index] != 0)
        ExecuteError("vertex continuation has data beyond its input span");
    }
    for (std::size_t index = continuation.shared_count;
         index < continuation.shared_registers.size(); ++index) {
      if (continuation.shared_registers[index] != 0)
        ExecuteError("vertex continuation has data beyond its shared span");
    }
    for (std::size_t index = 0; index < continuation.temporaries.size();
         ++index) {
      if ((continuation.temporary_written_mask & (UINT64_C(1) << index)) == 0 &&
          continuation.temporaries[index] != 0) {
        ExecuteError("vertex continuation has an unwritten TEMP value");
      }
    }
    for (std::size_t index = 0; index < continuation.outputs.size(); ++index) {
      if ((continuation.output_written_mask & (UINT64_C(1) << index)) == 0 &&
          continuation.outputs[index] != 0) {
        ExecuteError("vertex continuation has an unwritten output value");
      }
    }

    effective_vertex_inputs.assign(
        continuation.vertex_inputs.begin(),
        continuation.vertex_inputs.begin() + continuation.vertex_input_count);
    effective_shared_count = continuation.shared_count;
    std::copy_n(continuation.shared_registers.begin(), effective_shared_count,
                effective_shared.begin());
    temporaries = continuation.temporaries;
    temporary_written_mask = continuation.temporary_written_mask;
    result.outputs = continuation.outputs;
    result.written_mask = continuation.output_written_mask;
    result.emitted = continuation.emitted;
    result.ended_task = continuation.ended_task;
    pending = context.texture_response;
    pending_output_index = continuation.pending_output_index;
    pending_component_count = continuation.pending_component_count;
    drc0_pending = true;
    pc = continuation.resume_instruction_index;
  }
  if (conditionals &&
      effective_shared_count != kPcoConditionalsVertexSharedCount) {
    ExecuteError("conditionals VS requires exactly 16 shared dwords");
  }

  for (; pc < instructions.size(); ++pc) {
    const PcoInstruction &instruction = instructions[pc];
    if (result.executed_instruction_count ==
        std::numeric_limits<std::uint32_t>::max()) {
      ExecuteError("vertex dynamic instruction count overflow");
    }
    ++result.executed_instruction_count;

    if (instruction.opcode == PcoOpcode::kTextureSample) {
      const std::size_t coordinate_base = instruction.source.index;
      const std::uint64_t coordinate_mask =
          coordinate_base + 2U <= kPcoTemporaryCount
              ? (UINT64_C(3) << coordinate_base)
              : UINT64_C(0);
      if (drc0_pending ||
          instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 3 || instruction.repeat_count != 1 ||
          instruction.component_count != kPcoPixelOutputCount ||
          instruction.data_request != 0 ||
          instruction.source.bank != PcoRegisterBank::kTemporary ||
          coordinate_mask == 0 ||
          (temporary_written_mask & coordinate_mask) != coordinate_mask ||
          instruction.source1.bank != PcoRegisterBank::kShared ||
          instruction.source1.index % kPcoTextureDescriptorDwordCount != 0 ||
          instruction.source1.index / kPcoTextureDescriptorDwordCount >=
              kPcoMaximumTextureDescriptorSets ||
          static_cast<std::size_t>(instruction.source1.index) +
                  kPcoTextureDescriptorDwordCount >
              effective_shared_count ||
          instruction.source2.bank != PcoRegisterBank::kShared ||
          instruction.source2.index != instruction.source1.index + 8U ||
          static_cast<std::size_t>(instruction.source2.index) + 4U >
              effective_shared_count ||
          static_cast<std::size_t>(instruction.output_index) +
                  kPcoPixelOutputCount >
              kPcoTemporaryCount) {
        ExecuteError("invalid generic vertex SMP.2D.FCNORM instruction");
      }
      for (std::size_t coordinate = 0; coordinate < 2; ++coordinate) {
        result.texture_request.coordinates[coordinate] = ReadSource(
            instruction.source, effective_vertex_inputs, temporaries,
            temporary_written_mask, static_cast<std::uint8_t>(coordinate),
            ShaderStage::kVertex);
      }
      for (std::size_t word = 0; word < 4; ++word) {
        result.texture_request.texture_state[word] =
            effective_shared[instruction.source1.index + word];
        result.texture_request.sampler_state[word] =
            effective_shared[instruction.source2.index + word];
      }
      result.texture_request.coordinate_count = 2;
      result.texture_request.component_count = kPcoPixelOutputCount;
      result.texture_request.descriptor_set = static_cast<std::uint8_t>(
          instruction.source1.index / kPcoTextureDescriptorDwordCount);
      result.texture_request.binding = 0;
      result.texture_request.dimension = 2;
      result.texture_request.normalized = 1;
      result.texture_request.data_request = instruction.data_request;
      result.texture_request_valid = 1;

      std::copy(effective_vertex_inputs.begin(),
                effective_vertex_inputs.end(),
                result.continuation.vertex_inputs.begin());
      std::copy_n(effective_shared.begin(), effective_shared_count,
                  result.continuation.shared_registers.begin());
      result.continuation.temporaries = temporaries;
      result.continuation.outputs = result.outputs;
      result.continuation.temporary_written_mask = temporary_written_mask;
      result.continuation.output_written_mask = result.written_mask;
      result.continuation.program_binary_size = summary.binary_size;
      result.continuation.program_instruction_count =
          summary.instruction_count;
      result.continuation.resume_instruction_index =
          static_cast<std::uint16_t>(pc + 1U);
      result.continuation.pending_output_index = instruction.output_index;
      result.continuation.pending_component_count =
          instruction.component_count;
      result.continuation.data_request = instruction.data_request;
      result.continuation.vertex_input_count = static_cast<std::uint8_t>(
          effective_vertex_inputs.size());
      result.continuation.shared_count = effective_shared_count;
      result.continuation.emitted = result.emitted;
      result.continuation.ended_task = result.ended_task;
      result.continuation.valid = 1;
      result.suspended = 1;
      return result;
    }

    if (instruction.opcode == PcoOpcode::kWaitDataFence) {
      if (instruction.target != PcoWriteTarget::kNone ||
          instruction.source_count != 0 || instruction.repeat_count != 1 ||
          instruction.component_count != 1 || instruction.data_request != 0 ||
          instruction.output_index != 0 || !drc0_pending ||
          pending_component_count != kPcoPixelOutputCount ||
          static_cast<std::size_t>(pending_output_index) +
                  pending_component_count >
              kPcoTemporaryCount) {
        ExecuteError("WDF did not match one pending vertex drc0 SMP request");
      }
      for (std::size_t component = 0; component < pending_component_count;
           ++component) {
        const std::size_t destination = pending_output_index + component;
        temporaries[destination] = pending[component];
        temporary_written_mask |= UINT64_C(1) << destination;
      }
      drc0_pending = false;
      pending_output_index = 0;
      pending_component_count = 0;
      continue;
    }

    if (conditionals && instruction.target == PcoWriteTarget::kTemporary) {
      if (instruction.output_index >= temporaries.size())
        ExecuteError("conditionals VS temporary destination is out of range");
      const auto read = [&](const PcoRegisterRef &source) {
        if (source.bank == PcoRegisterBank::kShared) {
          if (source.index >= effective_shared_count)
            ExecuteError("conditionals VS shared source is out of range");
          return effective_shared[source.index];
        }
        return ReadSource(source, effective_vertex_inputs, temporaries,
                          temporary_written_mask, 0, ShaderStage::kVertex);
      };
      std::uint32_t value = 0;
      switch (instruction.opcode) {
      case PcoOpcode::kMoveBypass:
        value = read(instruction.source);
        break;
      case PcoOpcode::kFloatNegate:
        value = read(instruction.source) ^ UINT32_C(0x80000000);
        break;
      case PcoOpcode::kFloatAbs:
        value = read(instruction.source) & UINT32_C(0x7fffffff);
        break;
      case PcoOpcode::kMoveImmediate:
        value = instruction.immediate;
        break;
      case PcoOpcode::kFloatFloor:
        value = FloatFloorBits(read(instruction.source));
        break;
      case PcoOpcode::kFloatSubtract:
        value = FloatAddBits(read(instruction.source),
                             read(instruction.source1) ^ UINT32_C(0x80000000));
        break;
      case PcoOpcode::kFloatGreaterEqual:
        value = FloatGreaterEqualResultBits(
            read(instruction.source), read(instruction.source1),
            instruction.comparison_result_float_one);
        break;
      case PcoOpcode::kConditionalSelect:
        value = read(instruction.source) != 0 ? read(instruction.source1)
                                               : read(instruction.source2);
        break;
      case PcoOpcode::kConditionalSelectNegateTrue:
        value = read(instruction.source) != 0
                    ? (read(instruction.source1) ^ UINT32_C(0x80000000))
                    : read(instruction.source2);
        break;
      case PcoOpcode::kConditionalSelectGreaterZero:
        value = FloatGreaterZero(read(instruction.source))
                    ? read(instruction.source1)
                    : read(instruction.source2);
        break;
      case PcoOpcode::kFloatAdd:
        value = FloatAddBits(read(instruction.source),
                             read(instruction.source1));
        break;
      case PcoOpcode::kFloatAddNegateSource0:
        value = FloatAddBits(read(instruction.source) ^ UINT32_C(0x80000000),
                             read(instruction.source1));
        break;
      case PcoOpcode::kFloatMultiply:
        value = FloatMultiplyBits(read(instruction.source),
                                  read(instruction.source1));
        break;
      case PcoOpcode::kFloatMad:
        value = FloatMadBits(read(instruction.source),
                             read(instruction.source1),
                             read(instruction.source2));
        break;
      case PcoOpcode::kFloatMadNegateSource2:
        value = FloatMadBits(read(instruction.source),
                             read(instruction.source1),
                             read(instruction.source2) ^ UINT32_C(0x80000000));
        break;
      case PcoOpcode::kFloatMadNegateSource0:
        value = FloatMadBits(read(instruction.source) ^ UINT32_C(0x80000000),
                             read(instruction.source1),
                             read(instruction.source2));
        break;
      case PcoOpcode::kFloatMadNegateSource0Source2:
        value = FloatMadBits(read(instruction.source) ^ UINT32_C(0x80000000),
                             read(instruction.source1),
                             read(instruction.source2) ^ UINT32_C(0x80000000));
        break;
      default:
        ExecuteError("unexpected conditionals VS semantic operation");
      }
      temporaries[instruction.output_index] = value;
      temporary_written_mask |= UINT64_C(1) << instruction.output_index;
      continue;
    }

    if (generic_vertex &&
        instruction.target == PcoWriteTarget::kTemporary) {
      if (instruction.output_index >= temporaries.size() ||
          instruction.repeat_count != 1)
        ExecuteError("generic vertex ALU destination is out of range");
      const auto read = [&](const PcoRegisterRef &source) {
        if (source.bank == PcoRegisterBank::kShared) {
          if (source.index >= effective_shared_count)
            ExecuteError("generic vertex shared source is absent");
          return effective_shared[source.index];
        }
        return ReadSource(source, effective_vertex_inputs, temporaries,
                          temporary_written_mask, 0,
                          ShaderStage::kVertex);
      };
      const auto read_source0 = [&]() {
        std::uint32_t bits = read(instruction.source);
        if (instruction.source0_floor != 0)
          bits = FloatFloorBits(bits);
        if (instruction.source0_absolute != 0)
          bits &= UINT32_C(0x7fffffff);
        return bits;
      };
      const auto read_source1 = [&]() {
        std::uint32_t bits = read(instruction.source1);
        if (instruction.source1_absolute != 0)
          bits &= UINT32_C(0x7fffffff);
        return bits;
      };
      std::uint32_t value = 0;
      switch (instruction.opcode) {
      case PcoOpcode::kMoveImmediate:
        value = instruction.immediate;
        break;
      case PcoOpcode::kMoveBypass:
        value = read(instruction.source);
        break;
      case PcoOpcode::kFloatNegate:
        value = read(instruction.source) ^ UINT32_C(0x80000000);
        break;
      case PcoOpcode::kFloatAbs:
        value = read(instruction.source) & UINT32_C(0x7fffffff);
        break;
      case PcoOpcode::kFloatAdd:
        value = FloatAddBits(read_source0(), read_source1());
        if (instruction.saturate != 0)
          value = FloatSaturateBits(value);
        break;
      case PcoOpcode::kFloatAddNegateSource0:
        value = FloatAddBits(read(instruction.source) ^ UINT32_C(0x80000000),
                             read(instruction.source1));
        break;
      case PcoOpcode::kFloatMultiply:
        value = FloatMultiplyBits(read_source0(), read_source1());
        break;
      case PcoOpcode::kFloatMad:
        value = FloatMadBits(read(instruction.source),
                             read(instruction.source1),
                             read(instruction.source2));
        break;
      case PcoOpcode::kFloatMadNegateSource2:
        value = FloatMadBits(read(instruction.source),
                             read(instruction.source1),
                             read(instruction.source2) ^ UINT32_C(0x80000000));
        break;
      case PcoOpcode::kFloatMadNegateSource0:
        value = FloatMadBits(read(instruction.source) ^ UINT32_C(0x80000000),
                             read(instruction.source1),
                             read(instruction.source2));
        break;
      case PcoOpcode::kFloatMadNegateSource0Source2:
        value = FloatMadBits(read(instruction.source) ^ UINT32_C(0x80000000),
                             read(instruction.source1),
                             read(instruction.source2) ^ UINT32_C(0x80000000));
        break;
      case PcoOpcode::kFloatMin:
        value = FloatMinBits(read(instruction.source),
                             read(instruction.source1));
        break;
      case PcoOpcode::kFloatMax:
        value = FloatMaxBits(read(instruction.source),
                             read(instruction.source1));
        break;
      case PcoOpcode::kFloatEqual:
        value = FloatEqualBits(read(instruction.source),
                               read(instruction.source1));
        break;
      case PcoOpcode::kFloatGreaterEqual:
        value = FloatGreaterEqualResultBits(
            read(instruction.source), read(instruction.source1),
            instruction.comparison_result_float_one);
        break;
      case PcoOpcode::kFloatLess:
        value = FloatLessBits(read(instruction.source),
                              read(instruction.source1));
        break;
      case PcoOpcode::kConditionalSelect:
        value = read(instruction.source) != 0 ? read(instruction.source1)
                                               : read(instruction.source2);
        break;
      case PcoOpcode::kConditionalSelectNegateTrue:
        value = read(instruction.source) != 0
                    ? (read(instruction.source1) ^ UINT32_C(0x80000000))
                    : read(instruction.source2);
        break;
      case PcoOpcode::kConditionalSelectGreaterZero:
        value = FloatGreaterZero(read(instruction.source))
                    ? read(instruction.source1)
                    : read(instruction.source2);
        break;
      case PcoOpcode::kReciprocal:
        value = ReciprocalBits(read(instruction.source));
        break;
      case PcoOpcode::kReciprocalSquareRoot:
        value = ReciprocalSquareRootBits(read(instruction.source));
        break;
      case PcoOpcode::kFloatLog2:
        value = FloatLog2Bits(read(instruction.source));
        break;
      case PcoOpcode::kFloatExp2:
        value = FloatExp2Bits(read(instruction.source));
        break;
      case PcoOpcode::kFloatPackHalfRtne:
        value = FloatToHalf(read(instruction.source));
        break;
      case PcoOpcode::kFloatPackHalfRtz:
        value = FloatToHalfRtz(read(instruction.source));
        break;
      case PcoOpcode::kFloatUnpackHalf:
        value = HalfToFloat(static_cast<std::uint16_t>(
            read(instruction.source) & UINT32_C(0xffff)));
        break;
      default:
        ExecuteError("unknown generic vertex ALU operation");
      }
      temporaries[instruction.output_index] = value;
      temporary_written_mask |= UINT64_C(1) << instruction.output_index;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kFloatNegate ||
        instruction.opcode == PcoOpcode::kFloatAbs ||
        instruction.opcode == PcoOpcode::kFloatAddNegateSource0 ||
        instruction.opcode == PcoOpcode::kFloatMultiply ||
        instruction.opcode == PcoOpcode::kFloatMad ||
        instruction.opcode == PcoOpcode::kFloatMadNegateSource2 ||
        instruction.opcode == PcoOpcode::kFloatMadNegateSource0 ||
        instruction.opcode == PcoOpcode::kFloatMadNegateSource0Source2 ||
        instruction.opcode == PcoOpcode::kFloatMin ||
        instruction.opcode == PcoOpcode::kFloatMax ||
        instruction.opcode == PcoOpcode::kFloatEqual ||
        instruction.opcode == PcoOpcode::kFloatGreaterEqual ||
        instruction.opcode == PcoOpcode::kFloatLess ||
        instruction.opcode == PcoOpcode::kConditionalSelect ||
        instruction.opcode == PcoOpcode::kConditionalSelectNegateTrue ||
        instruction.opcode == PcoOpcode::kConditionalSelectGreaterZero ||
        instruction.opcode == PcoOpcode::kReciprocal ||
        instruction.opcode == PcoOpcode::kReciprocalSquareRoot ||
        instruction.opcode == PcoOpcode::kFloatLog2 ||
        instruction.opcode == PcoOpcode::kFloatExp2) {
      if (instruction.target != PcoWriteTarget::kTemporary ||
          instruction.output_index >= temporaries.size()) {
        ExecuteError("invalid ALU target in vertex shader");
      }
      const std::uint64_t bit =
          UINT64_C(1) << instruction.output_index;
      const std::uint32_t src0 = ReadSource(
          instruction.source, effective_vertex_inputs, temporaries,
          temporary_written_mask, 0, ShaderStage::kVertex);
      std::uint32_t result_val = 0;
      if (instruction.opcode == PcoOpcode::kFloatMultiply) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, effective_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        result_val = FloatMultiplyBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMad) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, effective_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        const std::uint32_t src2 = ReadSource(
            instruction.source2, effective_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        result_val = FloatMadBits(src0, src1, src2);
      } else if (instruction.opcode == PcoOpcode::kFloatMin) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, effective_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kVertex);
        result_val = FloatMinBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMax) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, effective_vertex_inputs, temporaries,
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
      const std::uint64_t bit =
          UINT64_C(1) << instruction.output_index;
      const std::uint32_t left =
          ReadSource(instruction.source, effective_vertex_inputs, temporaries,
                     temporary_written_mask, 0, ShaderStage::kVertex);
      const std::uint32_t right =
          ReadSource(instruction.source1, effective_vertex_inputs, temporaries,
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
            effective_shared_count != kPcoFillTexNearestVertexSharedCount) {
          ExecuteError("fill_tex_nearest VS SH0 scale is absent or invalid");
        }
      } else {
        ExecuteError("invalid MBYP-to-temporary source bank");
      }
      const std::uint64_t bit =
          UINT64_C(1) << instruction.output_index;
      temporaries[instruction.output_index] =
          instruction.source.bank == PcoRegisterBank::kShared
              ? effective_shared[instruction.source.index]
              : ReadSource(instruction.source, effective_vertex_inputs,
                           temporaries,
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
      if (instruction.source.index + instruction.repeat_count >
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
          instruction.source, effective_vertex_inputs, temporaries,
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

  if (drc0_pending)
    ExecuteError("vertex program ended with an unresolved drc0 request");
  if (vertex_input_mask != summary.vertex_input_mask ||
      result.written_mask != summary.vertex_output_mask ||
      result.ended_task != summary.ends_task) {
    ExecuteError("vertex execution does not match the decoded summary");
  }
  return result;
}

PcoVertexExecution ResumeVertexPco(
    const PcoProgramSummary &summary,
    const std::vector<PcoInstruction> &instructions,
    const PcoVertexContinuation &continuation,
    const std::array<std::uint32_t, kPcoPixelOutputCount> &texture_response) {
  PcoVertexExecutionContext context;
  context.texture_response = texture_response;
  context.continuation = continuation;
  context.texture_response_valid = 1;
  return ExecuteVertexPco(summary, instructions, {}, context);
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
  if (context.shared_count > kPcoMaximumFragmentSharedCount)
    ExecuteError("fragment shared-register count exceeds modeled USC file");
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
  const bool texture_program = std::any_of(
      instructions.begin(), instructions.end(), [](const auto &instruction) {
        return instruction.opcode == PcoOpcode::kTextureSample;
      });
  const bool conditionals =
      MatchesConditionalsProfile(ShaderStage::kFragment, instructions);
  if (conditionals &&
      (context.shared_count != kPcoConditionalsFragmentSharedCount ||
       context.coefficient_count != 0)) {
    ExecuteError(
        "conditionals FS requires four shared dwords and no coefficients");
  }
  const bool resuming = context.continuation.valid != 0;
  if (resuming && !texture_program) {
    ExecuteError("fragment continuation requires a texture-sampling program");
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

  /* Temporary, opt-in instruction trace used to compare one native PCO lane
   * with Mesa's compiler IR.  It is deliberately keyed by binary size and
   * exact sample coordinates so a replay does not emit every fragment. */
  const auto trace_coordinate = [](const char *name,
                                   std::uint32_t fallback) {
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0')
      return fallback;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return end != text && *end == '\0' &&
                   value <= std::numeric_limits<std::uint32_t>::max()
               ? static_cast<std::uint32_t>(value)
               : fallback;
  };
  const auto float_bits = [](float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  const char *trace_flag = std::getenv("PVRGPU_PCO_TRACE_FRAGMENT");
  const std::uint32_t trace_binary = trace_coordinate(
      "PVRGPU_PCO_TRACE_BINARY_SIZE", summary.binary_size);
  const std::uint32_t trace_x =
      trace_coordinate("PVRGPU_PCO_TRACE_X", UINT32_MAX);
  const std::uint32_t trace_y =
      trace_coordinate("PVRGPU_PCO_TRACE_Y", UINT32_MAX);
  const bool trace = trace_flag != nullptr && *trace_flag != '\0' &&
                     summary.binary_size == trace_binary &&
                     context.sample_x == float_bits(static_cast<float>(trace_x)) &&
                     context.sample_y == float_bits(static_cast<float>(trace_y));
  const auto opcode_name = [](PcoOpcode opcode) {
    switch (opcode) {
    case PcoOpcode::kMoveBypass: return "MBYP";
    case PcoOpcode::kFloatNegate: return "FNEG";
    case PcoOpcode::kFloatAbs: return "FABS";
    case PcoOpcode::kMoveImmediate: return "MOVI";
    case PcoOpcode::kFloatFloor: return "FLR";
    case PcoOpcode::kFloatSubtract: return "FSUB";
    case PcoOpcode::kFloatGreaterEqual: return "FGE";
    case PcoOpcode::kFloatEqual: return "FEQ";
    case PcoOpcode::kFloatLess: return "FLT";
    case PcoOpcode::kConditionalSelect: return "CSEL";
    case PcoOpcode::kConditionalSelectNegateTrue: return "CSEL.NEG";
    case PcoOpcode::kConditionalSelectGreaterZero: return "CSEL.GZ";
    case PcoOpcode::kFloatAdd: return "FADD";
    case PcoOpcode::kFloatAddNegateSource0: return "FADD.NEG0";
    case PcoOpcode::kFloatMultiply: return "FMUL";
    case PcoOpcode::kFloatMad: return "FMAD";
    case PcoOpcode::kFloatMadNegateSource2: return "FMAD.NEG2";
    case PcoOpcode::kFloatMadNegateSource0: return "FMAD.NEG0";
    case PcoOpcode::kFloatMadNegateSource0Source2: return "FMAD.NEG02";
    case PcoOpcode::kFloatMin: return "FMIN";
    case PcoOpcode::kFloatMax: return "FMAX";
    case PcoOpcode::kReciprocal: return "FRCP";
    case PcoOpcode::kReciprocalSquareRoot: return "FRSQ";
    case PcoOpcode::kFloatLog2: return "FLOG2";
    case PcoOpcode::kFloatExp2: return "FEXP2";
    case PcoOpcode::kIntegerAdd: return "IADD";
    case PcoOpcode::kBitwiseAnd: return "AND";
    case PcoOpcode::kBitwiseOr: return "OR";
    case PcoOpcode::kBitwiseXor: return "XOR";
    case PcoOpcode::kBitwiseXnor: return "XNOR";
    case PcoOpcode::kFloatInterpolatePerspective: return "FITRP";
    case PcoOpcode::kWaitDataFence: return "WDF";
    default: return "OTHER";
    }
  };
  const auto trace_value = [](std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    std::cerr << "0x" << std::hex << std::setw(8) << std::setfill('0')
              << bits << std::dec << std::setfill(' ') << '(' <<
                 std::setprecision(9) << value << ')';
  };
  if (trace) {
    std::cerr << "pco-fragment-trace begin binary=" << summary.binary_size
              << " instructions=" << instructions.size() << " sample=";
    trace_value(context.sample_x);
    std::cerr << ',';
    trace_value(context.sample_y);
    std::cerr << " coeffs=" << static_cast<unsigned>(context.coefficient_count)
              << '\n';
  }

  PcoFragmentExecution result;
  const std::vector<std::uint32_t> no_vertex_inputs;
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  std::uint64_t temporary_written_mask = 0;
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
            PcoOpcode::kTextureSample) {
      ExecuteError("invalid texture fragment continuation location");
    }
    const PcoInstruction &sample =
        instructions[continuation.resume_instruction_index - 1];
    if (continuation.pending_output_index != sample.output_index ||
        continuation.pending_component_count != sample.component_count ||
        continuation.data_request != sample.data_request ||
        continuation.pending_component_count != kPcoPixelOutputCount ||
        static_cast<std::size_t>(continuation.pending_output_index) +
                continuation.pending_component_count >
            kPcoTemporaryCount) {
      ExecuteError("invalid texture fragment continuation response range");
    }

    std::uint64_t expected_written_mask = 0;
    bool expected_request_pending = false;
    std::uint16_t expected_pending_output = 0;
    std::uint8_t expected_pending_components = 0;
    for (std::size_t index = 0;
         index + 1 < continuation.resume_instruction_index; ++index) {
      const PcoInstruction &prior = instructions[index];
      if (prior.opcode == PcoOpcode::kFloatInterpolatePerspective ||
          prior.opcode == PcoOpcode::kTextureSample) {
        if (expected_request_pending)
          ExecuteError("overlapping DRC0 requests precede continuation");
        expected_request_pending = true;
        expected_pending_output = prior.output_index;
        expected_pending_components = prior.component_count;
        continue;
      }
      if (prior.opcode == PcoOpcode::kWaitDataFence) {
        if (!expected_request_pending)
          ExecuteError("unmatched WDF precedes texture continuation");
        for (std::uint8_t component = 0;
             component < expected_pending_components; ++component) {
          expected_written_mask |=
              UINT64_C(1) << (expected_pending_output + component);
        }
        expected_request_pending = false;
        continue;
      }
      if (prior.target == PcoWriteTarget::kTemporary)
        expected_written_mask |= UINT64_C(1) << prior.output_index;
    }
    if (expected_request_pending ||
        continuation.temporary_written_mask != expected_written_mask) {
      ExecuteError("texture fragment continuation TEMP state is inconsistent");
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

    if (conditionals && instruction.opcode == PcoOpcode::kInternal) {
      ++pc;
      continue;
    }

    if (conditionals && instruction.target == PcoWriteTarget::kTemporary) {
      if (instruction.output_index >= temporaries.size())
        ExecuteError("conditionals FS temporary destination is out of range");
      const auto read = [&](const PcoRegisterRef &source) {
        if (source.bank == PcoRegisterBank::kShared) {
          if (source.index >= context.shared_count)
            ExecuteError("conditionals FS shared source is out of range");
          return context.shared_registers[source.index];
        }
        return ReadSource(source, no_vertex_inputs, temporaries,
                          temporary_written_mask, 0,
                          ShaderStage::kFragment);
      };
      std::uint32_t value = 0;
      switch (instruction.opcode) {
      case PcoOpcode::kMoveBypass:
        value = read(instruction.source);
        break;
      case PcoOpcode::kMoveImmediate:
        value = instruction.immediate;
        break;
      case PcoOpcode::kFragmentCoordinate:
        value = instruction.immediate == 0 ? context.sample_x
                                           : context.sample_y;
        break;
      case PcoOpcode::kFloatFloor:
        value = FloatFloorBits(read(instruction.source));
        break;
      case PcoOpcode::kFloatSubtract:
        value = FloatAddBits(read(instruction.source),
                             read(instruction.source1) ^ UINT32_C(0x80000000));
        break;
      case PcoOpcode::kFloatGreaterEqual:
        value = FloatGreaterEqualResultBits(
            read(instruction.source), read(instruction.source1),
            instruction.comparison_result_float_one);
        break;
      case PcoOpcode::kConditionalSelect:
        value = read(instruction.source) != 0 ? read(instruction.source1)
                                               : read(instruction.source2);
        break;
      case PcoOpcode::kFloatAdd:
        value = FloatAddBits(read(instruction.source),
                             read(instruction.source1));
        break;
      case PcoOpcode::kFloatMultiply:
        value = FloatMultiplyBits(read(instruction.source),
                                  read(instruction.source1));
        break;
      case PcoOpcode::kFloatMad:
        value = FloatMadBits(read(instruction.source),
                             read(instruction.source1),
                             read(instruction.source2));
        break;
      case PcoOpcode::kFloatPackHalfRtne:
        value = FloatToHalf(read(instruction.source));
        break;
      case PcoOpcode::kFloatUnpackHalf:
        value = HalfToFloat(static_cast<std::uint16_t>(
            read(instruction.source) & UINT32_C(0xffff)));
        break;
      default:
        ExecuteError("unexpected conditionals FS semantic operation");
      }
      temporaries[instruction.output_index] = value;
      temporary_written_mask |= UINT64_C(1) << instruction.output_index;
      ++pc;
      continue;
    }

    if (instruction.opcode ==
        PcoOpcode::kFloatInterpolatePerspective) {
      if (drc0_pending ||
          instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 2 || instruction.repeat_count != 1 ||
          instruction.component_count < 1 ||
          instruction.component_count > kPcoPixelOutputCount ||
          static_cast<std::size_t>(instruction.output_index) +
                  instruction.component_count >
              kPcoTemporaryCount ||
          instruction.data_request != 0 ||
          instruction.iteration_mode != PcoIterationMode::kPixel ||
          instruction.perspective != 1 || instruction.saturate != 0 ||
          instruction.source.bank != PcoRegisterBank::kCoefficient ||
          instruction.source.index < 4 ||
          (instruction.source.index & 3U) != 0 ||
          static_cast<std::size_t>(instruction.source.index) +
                  instruction.component_count * 4U >
              context.coefficient_count ||
          !IsRegister(instruction.source1, PcoRegisterBank::kCoefficient, 0)) {
        ExecuteError("invalid FITRP.PIXEL semantic instruction");
      }
      const std::uint32_t reciprocal_w = EvaluateCoefficientPlane(context, 0);
      /* llvmpipe lowers perspective interpolation as two independently
       * rounded binary32 operations: oow = 1.0F / w, followed by
       * numerator * oow.  A single numerator / w rounds only once and can
       * select the adjacent normalized texture coordinate at a nearest-texel
       * boundary.  Preserve the public driver ABI's operation sequence. */
      const std::uint32_t one_over_w =
          FloatDivideBits(UINT32_C(0x3f800000), reciprocal_w);
      for (std::size_t component = 0;
           component < instruction.component_count;
           ++component) {
        const std::size_t coefficient_base =
            instruction.source.index + component * 4U;
        pending[component] = FloatMultiplyBits(
            EvaluateCoefficientPlane(context, coefficient_base), one_over_w);
      }
      pending_output_index = instruction.output_index;
      pending_component_count = instruction.component_count;
      drc0_pending = true;
      if (trace) {
        std::cerr << "pco-fragment-trace pc=" << pc << " off="
                  << instruction.binary_offset << " op=FITRP pending=";
        for (std::size_t component = 0;
             component < instruction.component_count; ++component) {
          if (component)
            std::cerr << ',';
          trace_value(pending[component]);
        }
        std::cerr << '\n';
      }
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
        temporary_written_mask |= UINT64_C(1) << destination;
      }
      drc0_pending = false;
      pending_output_index = 0;
      pending_component_count = 0;
      if (trace) {
        std::cerr << "pco-fragment-trace pc=" << pc << " off="
                  << instruction.binary_offset << " op=WDF mask=0x"
                  << std::hex << temporary_written_mask << std::dec << '\n';
      }
      ++pc;
      continue;
    }

    if (instruction.opcode == PcoOpcode::kTextureSample) {
      const std::size_t coordinate_base = instruction.source.index;
      const std::uint64_t coordinate_mask =
          coordinate_base + 2U <= kPcoTemporaryCount
              ? (UINT64_C(3) << coordinate_base)
              : UINT64_C(0);
      if (drc0_pending ||
          instruction.target != PcoWriteTarget::kTemporary ||
          instruction.source_count != 3 || instruction.repeat_count != 1 ||
          instruction.component_count != kPcoPixelOutputCount ||
          instruction.data_request != 0 ||
          instruction.source.bank != PcoRegisterBank::kTemporary ||
          coordinate_mask == 0 ||
          (temporary_written_mask & coordinate_mask) != coordinate_mask ||
          instruction.source1.bank != PcoRegisterBank::kShared ||
          static_cast<std::size_t>(instruction.source1.index) + 4U >
              context.shared_count ||
          instruction.source2.bank != PcoRegisterBank::kShared ||
          static_cast<std::size_t>(instruction.source2.index) + 4U >
              context.shared_count ||
          instruction.source1.index % kPcoTextureDescriptorDwordCount != 0 ||
          instruction.source2.index != instruction.source1.index + 8U ||
          static_cast<std::size_t>(instruction.source1.index) +
                  kPcoTextureDescriptorDwordCount >
              context.shared_count ||
          static_cast<std::size_t>(instruction.output_index) + 4U >
              kPcoTemporaryCount) {
        ExecuteError("invalid generic SMP.2D.FCNORM instruction");
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
      result.texture_request.descriptor_set = static_cast<std::uint8_t>(
          instruction.source1.index / kPcoTextureDescriptorDwordCount);
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

    if (instruction.opcode == PcoOpcode::kFloatNegate ||
        instruction.opcode == PcoOpcode::kFloatAbs ||
        instruction.opcode == PcoOpcode::kFloatAddNegateSource0 ||
        instruction.opcode == PcoOpcode::kFloatMultiply ||
        instruction.opcode == PcoOpcode::kFloatMad ||
        instruction.opcode == PcoOpcode::kFloatMadNegateSource2 ||
        instruction.opcode == PcoOpcode::kFloatMadNegateSource0 ||
        instruction.opcode == PcoOpcode::kFloatMadNegateSource0Source2 ||
        instruction.opcode == PcoOpcode::kFloatMin ||
        instruction.opcode == PcoOpcode::kFloatMax ||
        instruction.opcode == PcoOpcode::kFloatEqual ||
        instruction.opcode == PcoOpcode::kFloatGreaterEqual ||
        instruction.opcode == PcoOpcode::kFloatLess ||
        instruction.opcode == PcoOpcode::kConditionalSelect ||
        instruction.opcode == PcoOpcode::kConditionalSelectNegateTrue ||
        instruction.opcode == PcoOpcode::kConditionalSelectGreaterZero ||
        instruction.opcode == PcoOpcode::kReciprocal ||
        instruction.opcode == PcoOpcode::kReciprocalSquareRoot ||
        instruction.opcode == PcoOpcode::kFloatLog2 ||
        instruction.opcode == PcoOpcode::kFloatExp2 ||
        instruction.opcode == PcoOpcode::kMoveImmediate ||
        instruction.opcode == PcoOpcode::kFloatPackHalfRtne ||
        instruction.opcode == PcoOpcode::kFloatPackHalfRtz ||
        instruction.opcode == PcoOpcode::kFloatUnpackHalf ||
        (instruction.opcode == PcoOpcode::kFloatAdd && instruction.target == PcoWriteTarget::kTemporary) ||
        (instruction.opcode == PcoOpcode::kMoveBypass && instruction.target == PcoWriteTarget::kTemporary) ||
        instruction.opcode == PcoOpcode::kIntegerAdd ||
        instruction.opcode == PcoOpcode::kBitwiseAnd ||
        instruction.opcode == PcoOpcode::kBitwiseOr ||
        instruction.opcode == PcoOpcode::kBitwiseXor ||
        instruction.opcode == PcoOpcode::kBitwiseXnor ||
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
      const std::uint64_t bit =
          UINT64_C(1) << instruction.output_index;
      const auto read = [&](const PcoRegisterRef &source,
                            std::uint8_t repeat = 0) {
        if (source.bank == PcoRegisterBank::kShared) {
          const std::size_t index = source.index + repeat;
          if (index >= context.shared_count)
            ExecuteError("generic fragment shared source is absent");
          return context.shared_registers[index];
        }
        if (source.bank == PcoRegisterBank::kCoefficient) {
          // The stored plane term itself, not an interpolated value: that is
          // what an uninterpolated varying reads.
          const std::size_t index = source.index + repeat;
          if (index >= context.coefficient_count)
            ExecuteError("generic fragment coefficient source is absent");
          return context.coefficients[index];
        }
        return ReadSource(source, no_vertex_inputs, temporaries,
                          temporary_written_mask, repeat,
                          ShaderStage::kFragment);
      };
      std::uint32_t src0 =
          instruction.opcode == PcoOpcode::kMoveImmediate
              ? UINT32_C(0)
              : read(instruction.source);
      if (instruction.source0_floor != 0)
        src0 = FloatFloorBits(src0);
      if (instruction.source0_absolute != 0)
        src0 &= UINT32_C(0x7fffffff);
      std::uint32_t result_val = 0;
      if (instruction.opcode == PcoOpcode::kMoveImmediate) {
        result_val = instruction.immediate;
      } else if (instruction.opcode == PcoOpcode::kMoveBypass) {
        result_val = src0;
      } else if (instruction.opcode == PcoOpcode::kFloatNegate) {
        result_val = src0 ^ UINT32_C(0x80000000);
      } else if (instruction.opcode == PcoOpcode::kFloatAbs) {
        result_val = src0 & UINT32_C(0x7fffffff);
      } else if (instruction.opcode == PcoOpcode::kFloatAdd) {
        std::uint32_t src1 = read(instruction.source1);
        if (instruction.source1_absolute != 0)
          src1 &= UINT32_C(0x7fffffff);
        result_val = FloatAddBits(src0, src1);
        if (instruction.saturate != 0)
          result_val = FloatSaturateBits(result_val);
      } else if (instruction.opcode ==
                 PcoOpcode::kFloatAddNegateSource0) {
        const std::uint32_t src1 = read(instruction.source1);
        result_val = FloatAddBits(src0 ^ UINT32_C(0x80000000), src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMultiply) {
        std::uint32_t src1 = read(instruction.source1);
        if (instruction.source1_absolute != 0)
          src1 &= UINT32_C(0x7fffffff);
        result_val = FloatMultiplyBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMad) {
        const std::uint32_t src1 = read(instruction.source1);
        const std::uint32_t src2 = read(instruction.source2);
        result_val = FloatMadBits(src0, src1, src2);
      } else if (instruction.opcode ==
                 PcoOpcode::kFloatMadNegateSource2) {
        const std::uint32_t src1 = read(instruction.source1);
        const std::uint32_t src2 = read(instruction.source2);
        result_val = FloatMadBits(src0, src1,
                                  src2 ^ UINT32_C(0x80000000));
      } else if (instruction.opcode ==
                 PcoOpcode::kFloatMadNegateSource0) {
        const std::uint32_t src1 = read(instruction.source1);
        const std::uint32_t src2 = read(instruction.source2);
        result_val = FloatMadBits(src0 ^ UINT32_C(0x80000000), src1, src2);
      } else if (instruction.opcode ==
                 PcoOpcode::kFloatMadNegateSource0Source2) {
        const std::uint32_t src1 = read(instruction.source1);
        const std::uint32_t src2 = read(instruction.source2);
        result_val = FloatMadBits(src0 ^ UINT32_C(0x80000000), src1,
                                  src2 ^ UINT32_C(0x80000000));
      } else if (instruction.opcode == PcoOpcode::kFloatMin) {
        const std::uint32_t src1 = read(instruction.source1);
        result_val = FloatMinBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatMax) {
        const std::uint32_t src1 = read(instruction.source1);
        result_val = FloatMaxBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatEqual) {
        const std::uint32_t src1 = read(instruction.source1);
        result_val = FloatEqualBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kFloatGreaterEqual) {
        const std::uint32_t src1 = read(instruction.source1);
        result_val = FloatGreaterEqualResultBits(
            src0, src1, instruction.comparison_result_float_one);
      } else if (instruction.opcode == PcoOpcode::kFloatLess) {
        const std::uint32_t src1 = read(instruction.source1);
        result_val = FloatLessBits(src0, src1);
      } else if (instruction.opcode == PcoOpcode::kConditionalSelect) {
        result_val = src0 != 0 ? read(instruction.source1)
                               : read(instruction.source2);
      } else if (instruction.opcode ==
                 PcoOpcode::kConditionalSelectNegateTrue) {
        result_val = src0 != 0
                         ? (read(instruction.source1) ^ UINT32_C(0x80000000))
                         : read(instruction.source2);
      } else if (instruction.opcode ==
                 PcoOpcode::kConditionalSelectGreaterZero) {
        result_val = FloatGreaterZero(src0) ? read(instruction.source1)
                                            : read(instruction.source2);
      } else if (instruction.opcode == PcoOpcode::kReciprocal) {
        result_val = ReciprocalBits(src0);
      } else if (instruction.opcode == PcoOpcode::kReciprocalSquareRoot) {
        result_val = ReciprocalSquareRootBits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatLog2) {
        result_val = FloatLog2Bits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatExp2) {
        result_val = FloatExp2Bits(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatPackHalfRtne) {
        result_val = FloatToHalf(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatPackHalfRtz) {
        result_val = FloatToHalfRtz(src0);
      } else if (instruction.opcode == PcoOpcode::kFloatUnpackHalf) {
        result_val = HalfToFloat(
            static_cast<std::uint16_t>(src0 & UINT32_C(0xffff)));
      } else if (instruction.opcode == PcoOpcode::kUnpackUnsignedToFloat) {
        result_val = FloatFromUnsigned(src0);
      } else if (instruction.opcode == PcoOpcode::kUnpackSignedToFloat) {
        result_val = FloatFromSigned(static_cast<std::int32_t>(src0));
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
          temporary_written_mask |= UINT64_C(1) << (instruction.output_index + 1);
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
      } else if (instruction.opcode == PcoOpcode::kBitwiseXnor) {
        const std::uint32_t src1 = ReadSource(
            instruction.source1, no_vertex_inputs, temporaries,
            temporary_written_mask, 0, ShaderStage::kFragment);
        result_val = ~(src0 ^ src1);
      }
      temporaries[instruction.output_index] = result_val;
      temporary_written_mask |= bit;
      if (trace) {
        std::cerr << "pco-fragment-trace pc=" << pc << " off="
                  << instruction.binary_offset << " group="
                  << instruction.group_index << " op="
                  << opcode_name(instruction.opcode) << " dst=t"
                  << instruction.output_index << " src="
                  << static_cast<unsigned>(instruction.source.bank) << ':'
                  << instruction.source.index << ','
                  << static_cast<unsigned>(instruction.source1.bank) << ':'
                  << instruction.source1.index << ','
                  << static_cast<unsigned>(instruction.source2.bank) << ':'
                  << instruction.source2.index << " flags=floor"
                  << static_cast<unsigned>(instruction.source0_floor)
                  << "/abs" << static_cast<unsigned>(instruction.source0_absolute)
                  << static_cast<unsigned>(instruction.source1_absolute)
                  << "/sat" << static_cast<unsigned>(instruction.saturate)
                  << " value=";
        trace_value(result_val);
        std::cerr << '\n';
      }
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
    if (trace) {
      std::cerr << "pco-fragment-trace pc=" << pc << " off="
                << instruction.binary_offset << " op=PIXOUT" <<
                   static_cast<unsigned>(output) << " value=";
      trace_value(result.pixel_outputs[output]);
      std::cerr << '\n';
    }
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
