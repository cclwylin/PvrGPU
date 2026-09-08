# Third-party notices

## VK-GL-CTS / dEQP

The optional `pvrgpu-deqp` target builds the EGL, GLES2, GLES3, and GLES31
packages from the official
[VK-GL-CTS repository](https://github.com/KhronosGroup/VK-GL-CTS) at commit
`067e8832315e79817ede1c4863804e440f5d1c80`. The external checkout and build
tree are not vendored in this workspace. `src/deqp_runner/main.cpp` is derived
from the upstream `framework/platform/tcuMain.cpp` entry point and retains its
Android Open Source Project copyright and Apache License 2.0 header.

`third_party/vk-gl-cts.lock` records the source pin and the four linked package
targets. Preserve upstream notices and license texts when redistributing a
binary containing those packages or their adjacent test data.

## ChromeOS GLBench

The optional external GLBench runner is built from the official
[ChromeOS GLBench repository](https://chromium.googlesource.com/chromiumos/platform/glbench)
at commit `e99bc684272bffd68b06c998e272531c9c84330f`.

The source file headers identify ChromiumOS BSD-style licensing. The source is
not redistributed by this PvrGPU workspace. If it is vendored or packaged in a
future release, retain all upstream notices and include the canonical ChromeOS
license text after a license review.

The project-owned helpers
`src/systemc/common/glbench_triangle_fixture.h` and
`src/systemc/common/glbench_triangle_fixture.cpp` reproduce the pinned
GLBench input-data behavior needed by the built-in functional fixtures. Their
source provenance is `src/utils.cc` (`CreateLattice` and `CreateMesh`),
`src/trianglesetuptest.cc` (128×128 winding/cull cases), and
`src/attributefetchtest.cc` (cases 1/2/4/8: 64×64 lattice, one/two/four/eight bindings of the
same tightly packed float2 VBO), all at the commit above. These helpers do not make the built-in
path a Mesa/GLBench command-ingest implementation; they preserve the fixture
geometry and loop/order semantics for differential testing.

## Mesa / PowerVR PCO encoding

Compute SSBO atomic semantics follow `emit_atomic_mem` and
`lp_translate_atomic_op` in Mesa's
`src/gallium/auxiliary/gallivm/lp_bld_nir_soa.c` at commit
`da14d65e4499e66468094be52bff9ea0915a695e`: active-lane integer RMW,
signed/unsigned min/max, uint32 wrapping and return-old values. PvrGPU
implements these through its own serialized SystemC memory service, not
llvmpipe's LLVM JIT. This does not claim llvmpipe's out-of-bounds robustness:
PvrGPU rejects accesses outside an authorized binding view. The reference
file's notice is:

> Copyright 2019 Red Hat.
> All Rights Reserved.
>
> Permission is hereby granted, free of charge, to any person obtaining a
> copy of this software and associated documentation files (the "Software"),
> to deal in the Software without restriction, including without limitation
> the rights to use, copy, modify, merge, publish, distribute, sublicense,
> and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included
> in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
> OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
> THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Native compare-and-swap uses the same pinned PCO compiler's
`pco_nir_sync.c` and `usclib/sync.cl` lowering (Copyright 2025 Imagination
Technologies Ltd., MIT): MUTEX LOCK/RELEASE plus per-instance coherent LD/ST
selected by SR51 INST_NUM. The model executes those actual instructions;
it does not replace the sequence with a host CAS. Generated native compute
fixtures in `tests/pco_compute_fixtures.h` come from
`tests/pvrgpu_compute_compiler_test.c` and have individual hashes in
`third_party/mesa-pco.lock`. Test-only SUB encoding and ADD64 destination
boundary mutations are explicitly distinguished from compiler output.

`src/systemc/common/msaa.h` adapts the standard 1x/2x/4x/8x/16x
sample-position tables from Mesa's
`src/gallium/auxiliary/util/u_sample_positions.c`, copyright 2023 Alyssa
Rosenzweig, under the MIT license reproduced below. The 4x and 8x positions
also match llvmpipe's `lp_rast.c`. The driver resolve implementation follows
llvmpipe's use of `util_blitter`: normalized/floating-point samples are
averaged in linear color space and integer resolves select sample zero.
No llvmpipe shader or rasterizer implementation is linked into the model.

Multisample texture fetch follows `lp_build_sample_ms_offset` and
`lp_build_fetch_texel` in the same pinned Mesa's
`src/gallium/auxiliary/gallivm/lp_bld_sample_soa.c` (Copyright 2009 VMware,
Inc.; MIT notice below): select the requested sample, check bounds, and do
not filter or resolve. PvrGPU implements its own pixel-interleaved address
arithmetic instead of llvmpipe's sample-major storage. Native NNCOORDS/SNO,
the three-bit lookup sample field, SMPCNT and array-address lowering are
verified against Mesa `pco_nir_tex.c`, `pco_isa.py`, `rogue_hw_defs.h` and
`texstate.xml` at commit `da14d65e4499e66468094be52bff9ea0915a695e`.
`tools/pco-fixtures/generate_multisample_texture.c` emits the native test
binaries; these fixtures are test inputs, never runtime shader substitutes.
The 21 entries in `tests/pco_multisample_texture_fixtures.h` preserve that
compiler output byte-for-byte; their individual sizes and SHA-256 hashes
are recorded in `third_party/mesa-pco.lock`. They cover FS/VS multisample
fetch, descriptor-only NIR size/sample-count queries, and native fragment
position reads, plus ordinary 2D size with explicit LOD, using the PvrGPU
internal graphics export profile. The
NIR sample-count query fixtures establish compiler/ISA behavior, not the
availability of a `textureSamples` GLSL builtin in OpenGL ES.

`third_party/mesa-26.2.1-pco-stride-texture-size.patch` corrects that pinned
PCO `usclib/tex.cl` helper: STRIDE image-word bits 60:63 contain mip count,
not the normal IMAGE layout's base level. Only non-STRIDE queries add a
base level to the requested LOD. The patch preserves the upstream 2025
Imagination Technologies MIT notice and is applied equally to both Mesa
source trees. Fixture hashes include the patch identity; the original 20
unpatched hashes remain recorded separately for provenance, not as runtime
shader alternatives.

The MSAA alpha-to-coverage helpers in `src/systemc/common/msaa.h` and
per-sample late depth/stencil ordering adapt
`lp_build_alpha_to_coverage_dither`, `lp_build_sample_alpha_to_coverage` and
the alpha-to-one ordering in Mesa's
`src/gallium/drivers/llvmpipe/lp_state_fs.c` at commit
`da14d65e4499e66468094be52bff9ea0915a695e` (Mesa 26.2.1).
The functional C++ implementation uses the same thresholds and ordered 2x2
dither matrix, not LLVM code generation or llvmpipe host execution.
The original notice for this adaptation is:

> Copyright 2009 VMware, Inc.
> Copyright 2007 VMware, Inc.
> All Rights Reserved.
>
> Permission is hereby granted, free of charge, to any person obtaining a
> copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sub license, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice (including the
> next paragraph) shall be included in all copies or substantial portions
> of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
> OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
> IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
> ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

`src/gallium/drivers/pvrgpu/pvrgpu_msaa.h` adapts the sequential F32
resolve arithmetic in `util_make_fs_msaa_resolve` from Mesa's
`src/gallium/auxiliary/util/u_simple_shaders.c` at the same commit.
Its copyright (2008 VMware, Inc.; 2009 Marek Olšák) and full permission and
disclaimer are preserved in that header. Tests separately cover sRGB
decode/linear-average/encode, resolve-before-bilinear filtering and integer
sample selection. Mesa format utilities remain part of the external Mesa
build rather than being copied into the model.

The patches `third_party/mesa-26.2.1-drisw-imported-msaa.patch` and
`third_party/mesa-26.2.1-msaa-readpixels-resolve.patch` target that same Mesa
revision's DRI software frontend and renderbuffer readback, respectively.
They preserve upstream file notices and must be applied equally to the local
PvrGPU and llvmpipe Mesa builds for default-framebuffer MSAA comparisons.
`third_party/mesa-26.2.1-ms-texture-sample-query.patch` targets the same pinned
revision's `src/mesa/state_tracker/st_format.c`. It adds sampler support to
multisample texture sample-count queries, preserves renderbuffer queries,
and retains the upstream file notice. Apply it equally to both backends for
multisample-texture comparisons; it does not change shader execution or CTS.
`third_party/mesa-26.2.1-renderbuffer-optional-sampler.patch` preserves the
same revision's `src/mesa/main/renderbuffer.c` notice and requests optional
sampler binding only where supported, retaining render-only sample counts.
`third_party/mesa-26.2.1-pco-stride-texture-size.patch` corrects PCO usclib's
interpretation of STRIDE image descriptors: their high WORD1 field is a mip
count, not a base level. The native compiler still emits and executes the
descriptor-query instructions. Both patches are applied to both Mesa trees;
the fixture lock records compiler-patch provenance and regenerated hashes.

`third_party/mesa-26.2.1-failed-query-state.patch` preserves the same revision's
Mesa query-object and conditional-rendering file notices. It propagates an
already reported Gallium query failure without polling forever, reporting a
successful zero, or overwriting application result memory. It also balances
query lifetime bookkeeping and permits a fresh query interval after failure.

The current runtime points to an external Mesa 26.2.1 build with local
llvmpipe telemetry patches. Most Mesa source and binaries are not redistributed
by this workspace. The following source artifacts do, however, preserve a
small MIT-licensed public PowerVR PCO encoding subset and generated shader
bytes from Mesa commit `da14d65e4499e66468094be52bff9ea0915a695e`:

- `src/systemc/shader/pco_iss.cpp`
- `src/systemc/shader/pco_iss.h`
- `tools/pco-fixtures/generate_fill_solid_fs.c`
- `tools/pco-fixtures/generate_attribute_fetch_shader.c`
- `tools/pco-fixtures/generate_varying_shader.c`
- `tools/pco-fixtures/generate_fill_tex_nearest.c`
- `tools/pco-fixtures/generate_multisample_texture.c`
- `tools/pco-fixtures/generate_conditionals.c`
- `tools/pco-fixtures/generate_ubo_shader.c`
- `tools/pco-fixtures/generate_temp256_operands.c`
- `tests/pco_uniform_buffer_fixtures.h`
- `tests/pco_multisample_texture_fixtures.h`
- `tests/pco_temp256_fixtures.h`

The NIR shader generators under `tools/pco-fixtures/` are development-time
tools that call Mesa's public PCO backend. The SystemC runtime does not call
Mesa's compiler: its built-in fixture paths embed immutable raw USC binaries,
and its driver-command path strictly decodes supplied compiled binaries.
`tools/pco-fixtures/generate_temp256_operands.c` separately
calls the generated public `pco_isa.py` source/destination encoders to record
high-register operand bytes in `tests/pco_temp256_fixtures.h`. That test header
assembles ISA-level LD/WDF/export and extended ADD64_32 register-boundary
programs using those bytes and the existing UBO compiler prefix; it does not
claim the assembled variants are unmodified GLSL compiler output. The
TEMP256 generator, operand tables, and assembled variants are test-only,
never runtime answer maps.
The locked core fixture provenance is:

| Fixture | Generator/origin | Bytes | SHA-256 |
|---|---|---:|---|
| Fill.Solid passthrough VS | Mesa `VS_PASSTHROUGH_COMMON` precompiled data | 32 | `81aeeb897687ca7e9e5997c90378a4e094d15b9c47df24ee2d0092d47d78a3b3` |
| Fill.Solid red FS | `tools/pco-fixtures/generate_fill_solid_fs.c` | 48 | `731542be4e64da704e3576248a6d234f8ea56e999e1a9ab447a182e7a03eb3dd` |
| Red FS + SH0 depth feedback | `tools/pco-fixtures/generate_fill_solid_fs.c` (`red depth-feedback`) | 80 | `163561de2918bd44913dac20eb45c759eb69960004326a96b645b924132715bf` |
| UBO raw DWORD VS, burst 1 | `tools/pco-fixtures/generate_ubo_shader.c` (`vertex 1`) | 80 | `ac88ef87c909311920f74e812451d576bdd3d5f4af9690f35dc4f5fa089b0f92` |
| UBO raw DWORD VS, burst 2 | `tools/pco-fixtures/generate_ubo_shader.c` (`vertex 2`) | 88 | `165d39faea0beb79e406f61a336dba33dcad368356d95ad766175a15848ef86c` |
| UBO raw DWORD VS, burst 3 | `tools/pco-fixtures/generate_ubo_shader.c` (`vertex 3`) | 104 | `1a23ee188479ed3b8ee5410d90324f38bf04c07bbc86d9d76edc63566640fa1b` |
| UBO raw DWORD VS, burst 4 | `tools/pco-fixtures/generate_ubo_shader.c` (`vertex 4`) | 112 | `7d257e488cd9b1cfdbd68414c18a1edcad75d86051fcf4196057a0efd9564905` |
| UBO raw DWORD FS, burst 1 | `tools/pco-fixtures/generate_ubo_shader.c` (`fragment 1`) | 96 | `38df63312521290dd7fb6617a59907d7babd9b218811029055441b2b32a0e283` |
| UBO raw DWORD FS, burst 2 | `tools/pco-fixtures/generate_ubo_shader.c` (`fragment 2`) | 96 | `65ab6614e945f2dcb75b11057462c20faf4dafe2d6d2a37b90d0e0e6eb2ca0c3` |
| UBO raw DWORD FS, burst 3 | `tools/pco-fixtures/generate_ubo_shader.c` (`fragment 3`) | 96 | `2d12ab78c8297412f65c4aca61853cca79d09ca5dfe232b9c726bde87823e567` |
| UBO raw DWORD FS, burst 4 | `tools/pco-fixtures/generate_ubo_shader.c` (`fragment 4`) | 96 | `8b6963924cc6fda6e18955dd9b3e36fe20eb0a7f3b45bc45d994a816bb81d90b` |
| Attribute fetch case-1 VS | `tools/pco-fixtures/generate_attribute_fetch_shader.c` | 56 | `01fb08add3c710fb9062ed0033fecc15e5cfbce56a38a49ed17db4e43f2bf026` |
| Attribute fetch case-2 VS | `tools/pco-fixtures/generate_attribute_fetch_shader.c` | 56 | `a275bcd7b146f7243e995528c197a04ee24e17f11d313313c7a5bea78030b88f` |
| Attribute fetch case-4 VS | `tools/pco-fixtures/generate_attribute_fetch_shader.c` | 96 | `81b4bf2b412eb2ba35adcd1076d965918336ffb0ffb860e66547695ef4a6ae28` |
| Attribute fetch case-8 VS | `tools/pco-fixtures/generate_attribute_fetch_shader.c` | 176 | `877802fe53fd258bb114aa2cf5713c317405c986b3a43b4612a58b6db9f7eccb` |
| Attribute fetch gray FS (cases 1/2/4/8) | `tools/pco-fixtures/generate_attribute_fetch_shader.c` | 48 | `45a123bc247c1b37570721ad7a18894be4d6802dfb459de191ebfc3a32ec5959` |

The attribute generator records the public `gx6250` target used to produce
the case-1/case-2/case-4/case-8 raw USC binaries. It constructs one/two/four/eight
`PIPE_FORMAT_R32G32_FLOAT` inputs in consecutive PCO vertex-input ranges and
uses NIR `fadd` for the multi-attribute sums. The runtime embeds the immutable
outputs and has no Mesa compiler dependency. `third_party/mesa-pco.lock` is the machine-readable pin
for the exact Mesa commit, generator paths, sizes, and hashes.

Relevant upstream declarative encoding and binary code carry:

> Copyright © 2024 Imagination Technologies Ltd.  
> SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

A future packaged external Mesa build must also preserve the licenses and
notices for every Mesa component included in that build.

## Qt / PySide6 and SystemC

Qt/PySide6 and Accellera SystemC are external runtime/build dependencies and
are not redistributed by this workspace. Packaging must be reviewed against
the selected distribution form and applicable licenses.
