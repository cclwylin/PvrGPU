# Third-party notices

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

The current runtime points to an external Mesa 26.2.1 build with local
llvmpipe telemetry patches. Most Mesa source and binaries are not redistributed
by this workspace. The following source artifacts do, however, preserve a
small MIT-licensed public PowerVR PCO encoding subset and generated shader
bytes from Mesa commit `da14d65e4499e66468094be52bff9ea0915a695e`:

- `src/systemc/shader/pco_iss.cpp`
- `src/systemc/shader/pco_iss.h`
- `tools/pco-fixtures/generate_fill_solid_fs.c`
- `tools/pco-fixtures/generate_attribute_fetch_shader.c`

The two files under `tools/pco-fixtures/` are development-time generators that
build NIR and call Mesa's public PCO backend. The SystemC runtime does not call
Mesa's compiler; it embeds and strictly decodes the resulting immutable raw
USC binaries. The locked core fixture provenance is:

| Fixture | Generator/origin | Bytes | SHA-256 |
|---|---|---:|---|
| Fill.Solid passthrough VS | Mesa `VS_PASSTHROUGH_COMMON` precompiled data | 32 | `81aeeb897687ca7e9e5997c90378a4e094d15b9c47df24ee2d0092d47d78a3b3` |
| Fill.Solid red FS | `tools/pco-fixtures/generate_fill_solid_fs.c` | 48 | `731542be4e64da704e3576248a6d234f8ea56e999e1a9ab447a182e7a03eb3dd` |
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
