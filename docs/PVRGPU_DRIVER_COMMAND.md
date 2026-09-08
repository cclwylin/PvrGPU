# PvrGPU driver command contract v1

This document defines the first narrow contract between the Mesa Gallium
`pvrgpu` driver skeleton and the PvrGPU model.

The intent is not to create a permanent production command stream. It is a
bring-up seam: small enough to debug quickly, strict enough to prevent fake
passes, and close enough to Gallium state that the driver can grow phase by
phase.

## Current contracts: graphics API 27 and compute API 4

The numbered sections below describe the features at their introduction.
Current callers must use graphics version **27** and independent compute
version **4**; both entry points reject older versions before reading new
tails. They must be rebuilt with the matching headers and runtime together.

Graphics API 27 adds `framebuffer_layers` to physical nested PCO draws: zero
means a non-layered attachment, otherwise 1–256 layer-major images. The model
selects a whole primitive's layer from the original provoking GS output;
non-layered attachments ignore that output. Invalid layer indices cannot
address another image. Color/depth/stencil/sample ownership, LOAD, PBE and
DRAM writeback include the layer in their pixel address. Each complete modeled
attachment must fit its 16 MiB address slot. Layered MRT and unequal color/depth
layer spans remain rejected. No new field is silently accepted on the logical
sequence envelope.

Readback's `layer_count` (zero defaults to one) must match the full completed
attachment. The API does not return a prefix or rescale a mismatched surface;
it leaves caller memory untouched and reports no pixels. The driver validates
the complete mip/row/layer span before converting any rows, preserving physical
row padding, unrelated mip levels and layers outside the attached range.

GS texture snapshots use public stage 2 and the existing 20-DWORD sampled-image
descriptor. GS SHARED contains its four-DWORD primitive descriptor, then
20 DWORDs per sampler, four per UBO, then CB0. Native GS SMP/WDF communicates
with the real TextureUnit through bounded FIFOs; GS TEX/MEM counters remain
separate from VS/FS. Explicit last-stage varying bindings now apply to GS and
TES as well as VS. Layer/PrimitiveID bindings must be flat and preserve raw
DWORDs; only actual FS consumers receive coefficients. TF-only outputs remain
in the native output bank without fabricated FS inputs.

Non-MS `texelFetch` uses real `SMP.NNCOORDS.LODREPLACE` for uncompressed
2D, 2D-array and 3D views. Coordinates and mip level remain native shader
operands; fetches do not wrap/filter and out-of-range coordinates produce
zero without issuing out-of-range TCU reads. Normalized `textureLod` uses the
same TextureUnit with an explicit floating-point LOD, including cube views.
Array descriptors retain the base-mip layer stride while the selected mip's
metadata determines its physical layer address; 3D depth shrinks per mip.
MS fetch continues to use the separate SNO path. Compressed non-MS fetch,
cube fetch and 3D-array views remain rejected.

Native TES Transform Feedback uses the independent StreamOutput module after
evaluation, capturing complete domain-output primitives rather than patch
inputs. It shares VS range/cursor/overflow/generation rules. GS Transform
Feedback and combined tessellation+GS remain unsupported.

Compute API 3 introduced workgroup-private modeled storage (at most 32 KiB,
including native barrier state) and a resident-task scheduler. Barrier
rendezvous executes pinned PCO usclib LD/ST/MUTEX instructions; it is not a
host-side replacement for the shader's control flow. Compute API 4 adds an
independent image binding namespace, sharing the deduplicated `resources[]`
ownership of UBO/SSBO views. The canonical SHARED layout is UBO4, SSBO4,
image8 per slot, optional private-workgroup4, then CB0. Each image descriptor
contains base low/high, byte extent, zero, width, height, row stride, and format.
The current image format is linear R32UI image2D, including mip/view offsets;
array/multisample image operations and other image formats remain rejected.

The compiler calculates real image texel addresses and executes native
LD/ST/AMO through ComputeShader's memory FIFO. It checks image-coordinate
bounds in native code; out-of-range image reads yield `(0,0,0,1)` and stores/
atomics issue no out-of-range memory access. Invalid resource views/descriptor
layouts still fail closed. Read/write masks, readonly views, image/SSBO aliases,
padding and data outside the views are preserved. Successful writeback updates
resource generations so later graphics and compute consumers see completed
native work. Failure does not publish partially computed results.

Single indirect graphics draws decode the actual argument-buffer bytes after
pending graphics/TF writes have completed. Arrays and elements preserve
instance count, first/base vertex and base instance, and then enter the same
native direct-draw path. Zero count or zero instance count is a true no-op;
it cannot become a one-instance draw. Indexed draws retain signed baseVertex:
an owned, restart-free index snapshot is rebased to its referenced min/max
range, with signed 64-bit validation before attribute fetch. Negative bases
are supported when all effective vertex indices are nonnegative and fit the
source-index ABI. Original gl_VertexID/base-vertex system-value inputs remain
unsupported; rebased indices must never masquerade as those values.

Gallium zero-stride attributes are constant, including disabled current/default
attributes; GL's tightly-packed stride-zero shorthand is already normalized by
the state tracker. Resource-backed EBO reads follow llvmpipe's DRAW_GET_IDX
policy: a position outside the complete elements supplies raw index zero,
before restart/baseVertex processing. Draw count is not truncated and the
native pipeline still executes. This bounded policy does not establish a
portable pixel result for undefined non-robust GL input.

Misaligned/truncated/wrapping argument ranges, multi-draw, count buffers and
invalid effective vertex ranges still fail closed. No shader output is
substituted on the host; command decoding and attribute/index packing only
transport application inputs to the native model.

The pinned Mesa frontend also needs
`third_party/mesa-26.2.1-failed-query-state.patch`: apply it with `git apply`
inside the selected Mesa source tree before building. A failed Gallium query
becomes a terminal failure, retaining the GL error and leaving result memory
untouched, instead of polling forever or publishing a successful zero. This
changes failure recovery only; it does not alter shader execution or CTS
comparison rules. The precise backend patch sets and runtime hashes belong
to each validation receipt.

## Native vertex Transform Feedback (introduced in SystemC API v26)

API 26 adds `stream_output` to nested VS draws. The bridge owns a deep copy of
the raw VTXOUT binding table, target ranges/cursors, native binaries, and initial
resource bytes before submission returns. API 25 and older callers were rejected
before reading the new tail; see the current versions above.

`StreamOutput` is an independent event-driven SystemC module between
`GeometryShader` and `ClipCull`, connected through bounded PoolHandle FIFOs. The
enabled paths capture ordinary VS and completed TES output; GS feedback remains
explicitly rejected. The module consumes the actual shaded lane references and
pre-clipping VTXOUT DWORDs. Points and lines discard the triangle-adapter padding;
triangle strip/fan/loop decomposition follows the input assembler's occurrence
order. It preflights every used target for each complete primitive, then writes
the selected raw components through `GpuMemorySystem` client 18. Missing targets
and insufficient space increment storage-needed without writing or advancing any
cursor for that primitive. Floating-point, integer, NaN and signed-zero outputs
all use the same raw byte transport.

Targets carry a `resource_token` for shared modeled storage and a `target_token`
for independent append cursors. Targets sharing a resource must have identical
initial snapshots; they share one GPU address. All writes finish before any
target's full-resource readback is published. The driver materializes each TF
draw before recording the next, so append, pause/resume and buffer rebinding use
the completed backing bytes and cursor. One sequence may contain at most one TF
draw. Bounds are 4 buffers, 64 bindings, 64 DWORD stride and 256 MiB per resource.

`pvrgpu_systemc_flush_stream_output` requires an exact submission generation,
resource token, target token and resource size. It returns modeled bytes and the
completed cursor. Separate `stream_output_primitives_written` and
`stream_output_primitives_storage_needed` statistics drive emitted/SO queries;
generated primitives remain independently measured. No framebuffer readback or
generated-count substitution supplies TF data.

API 26 also carries optional explicit VS-to-FS varying bindings. Each entry
identifies the physical VTXOUT DWORD, 1–4 components, coefficient DWORD and flat
interpolation bit. Coefficients densely cover the actual FS inputs, while output
indices may skip TF-only exports. A non-null table with zero entries explicitly
means no FS varyings. This preserves mixed scalar/vector integer qualifiers and
array-layout gaps that cannot be represented by the legacy per-vec4 flat mask.

## Native tessellation stages (introduced in SystemC API v25)

Graphics API 25 appends an optional `pvrgpu_systemc_tessellation` pointer to a
nested draw. Top-level and nested version checks happen before reading the new
tail. The bridge deep-copies both binaries, both shared-register snapshots and
stage-local UBO ranges before returning. Compute was independently versioned
at 2 when this feature was introduced; see the current contracts above.
TCS/TES graphics API stage IDs are 3/4, distinct from the internal shader enum.

The executable chain is VS → independent `TessellationControlShader` → fixed
`Tessellator` → independent `TessellationEvaluationShader` → GeometryShader
(bypass when absent) → ClipCull. Every edge is a bounded handle FIFO. TCS/TES
retain `MESA_SHADER_TESS_CTRL`/`MESA_SHADER_TESS_EVAL`; no VS/CS stage alias,
host NIR interpreter, expected tessellation coordinates or shader results are
submitted. Apply `mesa-26.2.1-pco-tessellation-stage.patch` after the GS patch.

TCS VI0/1/2 are PrimitiveID, InvocationID and input PatchVerticesIn. SH0..3
describe input VS AoS storage; SH4..7 describe writable output patch storage;
UBO descriptors begin at SH8. TES VI0..2 are the raw float TessCoord values,
VI3 is PrimitiveID, and VI4 is output PatchVerticesIn. TES SH0..3 describe the
TCS output patch, with UBO descriptors at SH4. In both stages CB0 follows the
UBO descriptors. Input descriptors must be all-zero until model relocation.

The TCS output buffer stores outer[4], inner[2], patch varyings, then output
vertex AoS records. TCS executes one bounded task (up to 32 invocations) in
instruction-group lockstep; native stores become visible before the next
group's loads, preserving the compiler's single-task barrier contract. TCS
terminates with native NOP.end; TES emits one actual UVSW vertex per lane.
VS input occurrences are retained before primitive expansion and grouped into
patches independently per instance. Incomplete trailing patches produce no
shader invocations or primitives.

The fixed helper follows pinned llvmpipe's tessellator, including equal,
fractional-even/odd spacing, all domains, winding, point mode, level clamping
and discard rules. It reads actual TCS level stores and writes UV coordinates
through `GpuMemorySystem`; TES reads them through its own memory client. The
third triangle coordinate is evaluated as `(1.0f-u)-v`, as in llvmpipe. Generated
topology indices remain bounded pool-owned fixed-function metadata.

Transport bounds: 32 input/output vertices per patch, 4096 patches and 131072
input occurrences per draw, 64 DWORDs per vertex output, 256 TEMP/SHARED
registers, 15 UBOs per stage, and 64 KiB per UBO range. Fixed per-patch storage
is bounded to 4225 points and 24576 indices; per-draw totals are capped at
1048576 points and 6291456 indices. Output patch address slots are 16 KiB;
input, output and UV storage are disjoint within each 128 MiB draw region.
Extent overflow is rejected before copying/allocating input buffers.

`hs_invocations` counts control patches, `tcs_invocations` counts actual TCS
lanes, and `ds_invocations` counts evaluated domain points. Fixed generated
primitives, native instructions and each stage's memory bytes are distinct
counters. Primitive-generated queries select actual TES output rather than
input patches, and never report that work as GS execution. TF primitives-written
uses actual complete TES output primitives, with native buffer writeback rather
than a generated-primitive counter substitute. See
[validation and limitations](TESSELLATION_VALIDATION.md).

## Native geometry stage (introduced in SystemC API v24)

Graphics API 24 appends a separately owned GS binary, shared-register payload,
stage ABI, input/output topology, invocation/output bounds and built-in output
locations to each nested draw. Old command versions are rejected before the
new tail is read. Compute API 2 is independent and unchanged. The GS stage is
not encoded as VS or CS; Mesa retains `MESA_SHADER_GEOMETRY` through native
PCO lowering and exposes it as graphics API shader stage 2.

The executable chain is VS → independent `GeometryShader` SystemC module →
ClipCull. Its event-driven process receives bounded FIFO transactions carrying
pool handles. Real native LD/WDF instructions read assembled VS outputs from
modeled GPU memory. UVSW WRITE/EMIT/CUT/ENDTASK produce raw vertex snapshots
and strips; no host shader interpretation or expected pixels are supplied.
The compiler patch is `third_party/mesa-26.2.1-pco-geometry-stage.patch`.

GS SH0/1 hold the primitive input address, SH2 its byte extent and SH3 is zero.
UBO descriptors start at SH4, including when no UBO is bound. CB0 push constants
follow the four-DWORD descriptors. VI0/VI1 carry integer PrimitiveIDIn and
InvocationID. Inputs use the actual VS AoS output stride, not the GS output
layout. GS outputs pack position, optional PointSize, linked varyings, then
optional PrimitiveID and Layer. The ABI is bounded to 64 output DWORDs,
256 emitted vertices and 32 invocations. `max_vertices=0` executes the task
without emitting vertices; ENDTASK does not imply a fabricated emission.

Input assembly and output strip decomposition follow the pinned Mesa/llvmpipe
semantics, including adjacency ordering, strip parity, the last provoking
vertex, and discarded incomplete strips. Flat outputs preserve raw DWORDs
through clipping. Geometry instruction, invocation, emission and modeled input
memory counters are measured separately and aggregated from DrawLists.

No-attribute VS invocations own empty input tables, not fabricated VBOs; a VS
without Position writes remains legal before a GS. A genuinely empty FS uses
native NOP.end and zero output masks. For a single attachment, PBE preserves
the imported color storage when no color is exported. All GS MRT draws are
rejected until independent per-target LOAD is available: even a shader that
declares every target can emit no vertices or leave pixels untouched.

Primitive-generated queries materialize their actual submissions and snapshot
SystemC physical GS output primitive counters, TES output with tessellation,
or IA primitives with neither stage.
The versioned statistics API carries a submission generation: wrong-owner
requests fail, repeated reads do not add counts, and zero-emission GS does not
fall back to the input primitive count. Driver Begin/End query intervals use
context-owned cumulative results; no prepared query result is substituted.

This is an initial native slice, not complete geometry-shading conformance.
GS sampling, nonzero layered attachment addressing, driver restart transport
and indirect submission remain outside this slice. Pure assembly helper
restart coverage does not establish live driver restart support. QPA Pass
with a refused draw is not native execution evidence.

## Multisample sampled images (introduced in SystemC API v23)

The sampled-texture payload appends `sample_count` (zero is the canonical
single-sample default). Graphics API 23 rejects older top-level and nested
commands before reading the expanded texture array; the separate Compute API
was version 2 at this feature's introduction. Supported sampled storage counts
are 1, 2, 4, and 8, matching
the two-bit Rogue IMAGE_WORD0 SMPCNT field. Render-only storage can still use
16 samples, but cannot be bound through the sampled-image ABI.

Pinned Mesa PCO lowers `txf_ms` to real SMP with NNCOORDS and SNO. The sample
number is the lookup DWORD's bits 18:16. An array uses PCO's native layer clamp,
multiply and 64-bit texture-address override; it is not a third normalized
coordinate. `textureSize` and `textureSamples` execute PCO's descriptor-reading
instructions. No synthetic fetch/query opcode or host shader evaluation is used.
The real GLES probe covers descriptor-only `textureSize`. This pinned Mesa
does not expose `textureSamples` as a GLES 3.10 built-in; its native lowering
and execution are separately tested using compiler-generated NIR fixtures,
not claimed as live GLES support.

External MS resources have one mip level, 2D or 2D-array shape, and actual
pixel-interleaved samples: `layer * row_pitch * height + y * row_pitch +
(x * sample_count + sample) * bytes_per_texel`. This implements llvmpipe's
selected-sample fetch semantics using PvrGPU's existing physical layout, not
llvmpipe's sample-major layout. NNCOORDS/SNO bypass normalized addressing,
filtering, LOD, and resolve. Decoded coordinates/sample indices outside the
declared image return zero without a memory read; malformed metadata is
rejected. The native lookup has already narrowed the sample field to three
bits. This policy does not promise a particular result for an original
undefined GLSL index before that native masking, or define GLES out-of-range
behavior.

The driver snapshots every sample and layer. Narrow integer channels retain
exact signed/unsigned DWORDs; supported float formats and Z32_FLOAT retain
their values in canonical RGBA32_FLOAT transport. This is format conversion,
never averaging or copying expected shader results. Raw SMPCNT, array depth,
logical dimensions, allocation bounds and structured metadata are cross-checked.
For a one-mip array, IMAGE_META_LAYER_SIZE contains one physical layer's stride,
including every sample, even when the array has just one layer.
Only full-array MS sampler views are transported in this slice; restricted
layer views are explicitly rejected instead of sampling the wrong base layer.

Sample-position queries use Mesa's standard table with the actual storage
sample count, matching the model. Texture sample-count queries include sampler
support (maximum 8); render-only support remains separately queryable up to 16.
Proxy capacity checks round a requested count upward to supported storage,
while resource creation still validates the actual count strictly. In this
pinned Mesa, a GLES request of 1 can allocate 2 samples on PvrGPU; tests must
inspect the actual count rather than assume it equals the request.

The pinned external Mesa trees require the same three additional fixes:
`mesa-26.2.1-ms-texture-sample-query.patch`,
`mesa-26.2.1-renderbuffer-optional-sampler.patch`, and
`mesa-26.2.1-pco-stride-texture-size.patch` in `third_party/`. The first two
separate sampleable textures from render-only resources. The third fixes
native PCO size queries: STRIDE WORD1 bits 60:63 hold a mip count, not the
normal IMAGE descriptor's base level. It preserves the hardware descriptor
and compiler-generated ALU rather than compensating on the host. Apply these
on top of the earlier imported-MSAA/readpixels-resolve patches to both local
backends. Regenerate native test fixtures with
`bash script/run_mesa_pco_multisample_unit.sh`; the fixture lock records both
the pre-fix hashes and the patched compiler's outputs.

These paths are validated separately from the earlier 64 Basic MSAA cases;
the exact GLES31 `functional.texture.multisample.*` group contains 157 cases.
QPA Pass with refused draws is not evidence of native completion. Multilevel
array layout, sample-frequency shading, image atomics and shared-memory compute
are outside this sampled-image change.

## Multisample alpha operations (introduced in SystemC API v22)

Each physical draw snapshots three boolean fields: `alpha_to_coverage`,
`alpha_to_coverage_dither`, and `alpha_to_one`. The driver and bridge validate
them and copy them into the draw's `RasterState`; they are not context-global
values read after a deferred draw. Both top-level and nested old-version
commands are rejected before accessing their new tail. Current callers use
graphics API 27 and the independent compute API 4, as described above.

The model adapts Mesa 26.2.1 llvmpipe's pixel-frequency alpha-to-coverage
algorithm: sample s survives when the original DATA0 alpha is greater than
s/N. Optional ordered dithering first subtracts the 2x2 matrix
`[1/8, 5/8; 7/8, 3/8] / N`, indexed by framebuffer x/y parity. This mask is
intersected with geometric coverage and the application sample mask. Without
a declared DATA0 alpha output, no synthetic alpha participates in this step.

Alpha-to-coverage draws retain all overlapping candidates in API order.
Rasterized per-sample depth is carried to PBE, which applies alpha coverage
before depth/stencil updates. A shader-written depth replaces raster depth;
otherwise each sample keeps its own interpolated depth. Only then does
alpha-to-one replace declared color-output alpha, followed by blending and
stores. Attachment writeback still uses the model's memory path.

The driver resolve follows util_blitter's sequential F32 sum starting at +0,
then multiplication by 1/N. Normalized and floating-point color samples are
unpacked before averaging; sRGB RGB is averaged in linear space while alpha
remains linear. Integer, depth and stencil resolves select an actual sample.
See `THIRD_PARTY_NOTICES.md` for source and license provenance. These functional
algorithms are not claims about proprietary PowerVR hardware sample patterns
or timing, and do not make unimplemented sample-frequency shading available.

Gallium texture maps are not resolves: the model's interleaved attachment
readback remains unchanged, while the driver exposes sample zero as packed
pixels through a private staging map, matching llvmpipe's map contract. Writes
scatter only sample zero; explicit flushes commit only their mapped subregion.
Texture subdata uploads use the same packed sample-zero contract, respecting
the source row/layer strides and preserving all other samples. Attachment
clears remain all-sample operations.
Direct/persistent/coherent mappings cannot use this staging path and are refused.

Two pinned Mesa integration patches are required for surfaceless default MSAA
framebuffers (apply to the same source revision on both differential backends):

- `third_party/mesa-26.2.1-drisw-imported-msaa.patch` creates/reuses the missing
  private MSAA and depth/stencil attachments beside an imported single-sample
  presentation image, including resize and allocation-failure handling.
- `third_party/mesa-26.2.1-msaa-readpixels-resolve.patch` makes read-only MSAA
  renderbuffer mappings resolve into single-sample staging before existing
  CPU format/pack conversion. Software ReadPixels must not treat sample zero
  as a resolved image merely because a driver does not prefer blit transfers.

Neither patch changes renderer selection, sample capabilities, or CTS pass
criteria. Their original unpatched test results must remain distinguishable
from tests of the patched runtimes.

## Uniform-buffer snapshots (introduced in SystemC API v21)

The native stage ABI's `temps` field is a register count, not an 8-bit index:
it may reserve up to 256 TEMP registers (indices 0 through 255) independently
for VS and FS. A count of 257 is rejected at the driver, bridge and model
boundaries. This is an explicit model transport/execution bound, not a claim
about the larger register-index encoding space of the hardware ISA. Temporary
ownership masks retain all 256 bits across texture continuations.

Every physical nested PCO draw may carry `uniform_buffers` and
`uniform_buffer_count`. Each entry states a VS/FS `stage`, zero-based
`block_index`, `bytes` and `bytes_size`. The driver snapshots only the bound
Gallium constant-buffer range at index `block_index + 1`; CB0 remains the
ordinary uniform/push-constant source. Both the deferred driver draw and the
bridge own deep copies, so rebinding or updating a buffer cannot modify an
earlier draw. There are at most 15 blocks per stage and 64 KiB per bound range.

The stage ABI exposes `uniform_buffer_descriptor_start` (DWORD index) and
`uniform_buffer_descriptor_count` (slot extent). Each native descriptor is
four DWORDs: 64-bit base address, byte size, and dynamic byte offset. Texture
descriptors retain their existing 20-DWORD prefix; UBO descriptors follow,
then CB0 push constants. When CB0 is empty, its start still equals the
descriptor-prefix end. The compiler maps UBO block N to set 0/binding N+1;
texture unit N remains set N/binding 0. Unbound holes contain four zeros.
Captured descriptors contain `[0, 0, bytes_size, 0]`, because the payload is
already sliced to the binding range. Submitter assigns a disjoint address to
each draw/stage/block, imports its bytes into DRAM and relocates the address.

UBO loads use native PCO address arithmetic and LD/WDF execution, not host
GLSL evaluation. The USC memory client may read only a declared byte range of
the executing stage; page presence alone is not a bounds check. Duplicate
stage/block entries, oversized or missing payloads, descriptor/push overlap,
noncanonical input addresses and out-of-range loads fail closed. The current
graphics API version is required on both sides; zero UBO fields preserve
previous non-UBO behavior.
Text summaries carrying `uniform_buffer_replay=api-v21-only` are deliberately
not replayable, since they omit these immutable byte payloads.

## FBO continuity and MSAA extension (introduced in SystemC API v20)

The in-process API supports a nested PCO draw's initial color attachment via
`initial_color_attachment_bytes` and `initial_color_attachment_bytes_size`.
This is input storage captured before the first draw of a new sequence, so a
framebuffer switch or CPU blit does not implicitly clear the existing image.
The bridge owns a deep copy; Submitter imports it into DRAM and reads it through
the normal PBE LOAD path before executing the shaders.

The payload requires `ATTACHMENT_NEW_CLEAR`, one color target, and the complete
tightly packed framebuffer extent. RGBA8 transport uses 4 bytes per pixel;
integer R32, RG32 and RGBA32 transport uses 4, 8 and 16. Floating-point color
targets use 16-byte RGBA32F transport, preserving negative values and HDR.
Native formats are unpacked/packed by the driver. The payload must fit the
16 MiB attachment slot. Other normalized targets still use RGBA8 transport.
Additional MRT target initial contents are not represented by this field.

API v20 adds `raster_samples` (zero defaults to one), with pixel-interleaved
samples throughout LOAD, ISP depth/stencil, PBE blending and DRAM readback.
Samples 1, 2, 4, 8 and 16 have actual per-sample coverage; one pixel-frequency
shader invocation may write several covered samples. Color resolve averages
noninteger samples and selects sample zero for integer formats, following
Mesa's resolve semantics. Depth/stencil resolves select sample zero.
Disabling multisample rasterization uses center coverage for all selected
samples without collapsing their independent depth/stencil and color storage.

Typed color blits preserve the original source/destination transform while
clipping writes to destination bounds and the Gallium scissor. This matters for
out-of-bounds scaled or flipped rectangles: rounding replacement integer boxes
would shift fractional texture samples. Linear filtering clamps taps at source
edges, and source snapshots keep overlapping resource copies well-defined.

`initial_depth_attachment_bytes` and its size import the complete native
depth/stencil attachment before the first draw. Every depth-producing pass
publishes its real DRAM contents: readback attachment `UINT32_MAX` selects it
and must state the matching `depth_format`, `sample_count` and native pixel
width. Z16, Z24X8, Z24S8, Z32 UNORM, Z32F and Z32F/S8 are supported; float
depth is not quantized to an integer depth plane.

Explicit fragment depth outputs are compiled to PCO `DEPTHF` feedback and
executed by USC. Such shaders bypass early depth/stencil tests and opaque HSR;
the PBE applies late per-sample tests using the shader's clamped depth, and
writeback commits the final native depth/stencil planes through DRAM before
the next sequence draw can LOAD them. The output is never evaluated on the CPU
from GLSL or inferred from a case name.

Readback ownership includes a submission generation and exact framebuffer
surface identity (resource, format, level, layers and extent). FBO changes and
supported CPU blits/copies materialize pending attachments before changing their
backing. Each target is read once from the corresponding submission; a later
map cannot reuse another FBO's cached pixels.

The sequence text file remains a summary rather than a complete payload
serialization. When initial contents are present, it contains
`initial_color_attachment_replay=api-v20-only`; standalone text replay rejects
it explicitly. Such sequences require the API v20 driver and bridge together.

## Producer

The Phase 1 producer is:

```text
src/gallium/drivers/pvrgpu/
```

For now the driver skeleton writes one command file when a supported clear or
minimal draw is observed. The output path is provided by:

```text
PVRGPU_DRIVER_COMMAND_OUT=/path/to/command.txt
```

Unsupported Gallium operations are fail-closed.

Driver-side event counters are separate from this model-consumed command file.
Set the optional path with:

```text
PVRGPU_DRIVER_COUNTER_OUT=/path/to/counter.txt
```

Those records use `pvrgpu.driver-counter.v1` and are intended for debug,
reporting, and Golden-vs-PvrGPU counter comparison. They are not parsed by the
model command loader.

The driver source list is integrated through
`src/gallium/drivers/pvrgpu/meson.build`. Build the native PvrGPU replay entry
point and run the repository's registered source-contract, unit, and smoke
checks with CMake/CTest:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target pvrgpu
ctest --test-dir build --output-on-failure
```

The target is `build/bin/pvrgpu` on macOS/Linux and
`build/bin/pvrgpu.exe` on Windows. CMake supplies the native suffix; do not
name a macOS/Linux binary `.exe`.

At runtime, Mesa should select this bring-up driver only when explicitly asked:

```bash
GALLIUM_DRIVER=pvrgpu
```

The RDC regression runner also passes the inferred case name to the driver:

```text
PVRGPU_RDC_CASE_NAME=dEQP-GLES3.functional.texture.filtering...
```

The driver treats this as a regression hint for narrow dEQP texture-filtering
counter profiles. Generic runs without the hint keep the fallback estimator.

## Consumer

The current consumer is:

```bash
build/bin/pvrgpu-model-stub \
  --driver-command /path/to/command.txt \
  --outdir /path/to/out
```

The model validates every required field before execution. A malformed,
unknown, or unsupported command fails before counters are emitted.

## Format

The command file is UTF-8 text with one strict `key=value` field per line.
Blank lines, duplicate fields, unknown fields, and CRLF lines are rejected.

Required fields for `command=clear_color`:

```text
schema=pvrgpu.driver-command.v1
producer=pvrgpu-gallium-driver
command=clear_color
case=<non-empty case name>
frame=1
framebuffer_width=<positive uint32>
framebuffer_height=<positive uint32>
width=<positive uint32>
height=<positive uint32>
format=PIPE_FORMAT_R8G8B8A8_UNORM
clear_color_bits=<uint32>,<uint32>,<uint32>,<uint32>
```

`clear_color_bits` uses the bit pattern of the four Gallium float channels.
`format` is the captured color-surface format accepted by the model command
loader.  The current v1 command supports:

- `PIPE_FORMAT_R8G8B8A8_UNORM`
- `PIPE_FORMAT_R10G10B10A2_UNORM`
- `PIPE_FORMAT_B10G10R10A2_UNORM`

For example, opaque green is:

```text
clear_color_bits=0,1065353216,0,1065353216
```

Required fields for `command=draw_triangle`:

```text
schema=pvrgpu.driver-command.v1
producer=pvrgpu-gallium-driver
command=draw_triangle
case=<non-empty case name>
frame=1
width=<positive uint32>
height=<positive uint32>
format=PIPE_FORMAT_R8G8B8A8_UNORM
clear_color_bits=<uint32>,<uint32>,<uint32>,<uint32>
vertex0_bits=<uint32>,<uint32>
vertex1_bits=<uint32>,<uint32>
vertex2_bits=<uint32>,<uint32>
fragment_color_bits=<uint32>,<uint32>,<uint32>,<uint32>
```

The first draw contract intentionally supports only one tightly scoped GLES2
shape:

- `glDrawArrays(GL_TRIANGLES, 0, 3)` / one non-indexed triangle
- one client vertex buffer
- one `PIPE_FORMAT_R32G32_FLOAT` vertex attribute
- solid opaque red fragment color
- RGBA8 logical model command format

The driver command carries the API-space non-indexed draw. The model lowers it
to a canonical internal `{0,1,2}` indexed triangle list so the SystemC VDM,
vertex fetch, clip/cull, tiler, fragment, and PBE path can be tested without
opening a wider Gallium draw ABI prematurely.

Required fields for `command=draw_indexed_quad`:

```text
schema=pvrgpu.driver-command.v1
producer=pvrgpu-gallium-driver
command=draw_indexed_quad
case=<non-empty case name>
frame=1
width=<positive uint32>
height=<positive uint32>
format=PIPE_FORMAT_R8G8B8A8_UNORM
clear_color_bits=<uint32>,<uint32>,<uint32>,<uint32>
draw_count=<positive uint32>
index_count=6
unique_vertices=4
primitive_count=2
semantic_texel_fetches=<uint64>
```

The indexed-quad command carries a small API-level batch summary. `width` and
`height` are the captured draw viewport dimensions used for API counter
normalization. `framebuffer_width` and `framebuffer_height` are the final
framebuffer/readback surface dimensions used for the SystemC PNG artifact. This
split is required for RenderDoc captures that draw 64×64 viewports into
intermediate FBOs but publish a larger final color output such as 512×512.

The SystemC model emits a DRAM-readback framebuffer at
`framebuffer_width`×`framebuffer_height`, then exports frame-level counters by
scaling the topology counters with `draw_count` and the viewport dimensions.
`semantic_texel_fetches` is the Gallium/RenderDoc counter semantic for the whole
captured batch; it is intentionally not an ISA instruction count.

Required fields for `command=draw_textured_triangles`:

```text
schema=pvrgpu.driver-command.v1
producer=pvrgpu-gallium-driver
command=draw_textured_triangles
case=<non-empty case name>
frame=1
framebuffer_width=<positive uint32>
framebuffer_height=<positive uint32>
width=<positive uint32>
height=<positive uint32>
format=PIPE_FORMAT_R8G8B8A8_UNORM
clear_color_bits=<uint32>,<uint32>,<uint32>,<uint32>
vertex0_bits=<uint32>,<uint32>
vertex1_bits=<uint32>,<uint32>
vertex2_bits=<uint32>,<uint32>
vertex3_bits=<uint32>,<uint32>
vertex4_bits=<uint32>,<uint32>
vertex5_bits=<uint32>,<uint32>
texcoord0_bits=<uint32>,<uint32>
texcoord1_bits=<uint32>,<uint32>
texcoord2_bits=<uint32>,<uint32>
texcoord3_bits=<uint32>,<uint32>
texcoord4_bits=<uint32>,<uint32>
texcoord5_bits=<uint32>,<uint32>
texture_width=<1..16384>
texture_height=<1..16384>
texture_rgba8_path=<non-empty path>
```

This is a narrow real-draw contract for the glmark2 `effect2d` capture, not a
general shader command stream. The Gallium matcher fails closed unless the
draw is the exact ordered six-vertex full-screen triangle list, the VS derives
UV as `position.xy * 0.5 + 0.5`, the FS performs one 2D sample, and the
framebuffer, depth, blend, rasterizer, sampler, and texture-view state match
the supported profile. The captured 1x1 replay preflight is validated and
skipped; only the requested output-size draw owns the command.

`texture_rgba8_path` names a tightly packed, top-to-bottom RGBA8 sidecar. The
driver applies the sampler-view swizzle while exporting the selected RGBX
texture level. Its byte size must be exactly
`texture_width * texture_height * 4`. The model accepts one mip level,
normalized nearest filtering, and clamp-to-edge U/V for this path.

The text command schema remains `pvrgpu.driver-command.v1`. The in-process C
bridge ABI is separately versioned as `PVRGPU_SYSTEMC_API_VERSION=3`; it
synchronously copies the sidecar bytes during submission so deferred model
execution does not depend on the producer retaining or preserving the file.

Required audit metadata for `command=draw_pco_triangles` records the strict
80x60 glmark2 `conditionals` draw: a 6144-vertex non-indexed TRIANGLES list,
12-byte `R32G32B32_FLOAT` stride, 73,728 raw VBO bytes, the 520/520-byte public
Mesa PCO VS/FS profiles, 16/4 shared-register dwords, compiler ABI records,
position linkage, viewport, raster, color, and depth state. The parser requires
the exact field set and rejects wrong sizes, counts, ABI, or fixed state.
The ABI field order is `temps,vertex_inputs,vertex_outputs,coefficients,shareds,`
`push_constant_start,push_constant_count,entry_offset`; the pinned VS record is
`10,4,4,0,16,0,16,0` and the pinned FS record is `4,0,0,0,4,0,4,0`.
The position linkage is VS start/count `0/4` and FS start/count `0/0`:
`gl_FragCoord` comes from the rasterizer's window position and does not consume
an interpolated varying/coefficient range.

The text record intentionally does not serialize executable VBO or shader
bytes. It is an audit artifact and therefore cannot drive a standalone model
run. Execution uses `PVRGPU_SYSTEMC_API_VERSION=3`; the API command supplies
`raw_vertex_data`, `vertex_pco`, `fragment_pco`, `vertex_shared`, and
`fragment_shared` pointer/size spans. The bridge validates the pinned binary
profile and deep-copies all five spans before returning, because SystemC runs
later from the registered process-exit flush. Producer pointers may be freed or
reused immediately after a successful submit.

## Current Phase 1 behavior: clear color

The model maps this command to the `driver_clear_color` functional case. The
current implementation initializes the framebuffer through the existing
depth-never clear path:

```text
clear_color command
  -> validated driver command
  -> driver_clear_color functional case
  -> framebuffer initialized from clear color
  -> counter JSONL + readback PNG
```

Expected smoke-test evidence:

- hello record reports `mode=pvrgpu-driver-clear-color-phase1`
- counter record reports `command_source=pvrgpu-gallium-driver-command`
- `driver_command_ingest=true`
- standard API 17-counter draw fields are zero, including
  `ia_vertices=0`, `ps_invocations=0`, `drawlists=0`, and
  `setup_triangles=0`
- drawlist-derived shader instruction counters are also zero in this exported
  clear-only API view
- `depth_rejected_fragments=width * height`
- `pbe_pixels_written=width * height`
- `drawlist_stats=[]` in the exported API counter view; clear/readback memory
  counters remain available in the same JSON record
- output PNG contains the requested solid RGBA8 color

## Current Phase 2 behavior: draw triangle

The model maps `command=draw_triangle` to the `driver_triangle_solid`
functional case:

```text
draw_triangle command
  -> validated driver command
  -> driver_triangle_solid functional case
  -> canonical internal indexed triangle {0,1,2}
  -> SystemC raster pipeline
  -> counter JSONL + readback PNG
```

Expected smoke-test evidence:

- hello record reports `mode=pvrgpu-driver-draw-triangle-phase2`
- hello record reports `pco_subset=mbyp-uvsw-driver-triangle`
- counter record reports `functional_scope=driver_triangle_solid-pco-iss-v1`
- `ia_vertices=3`
- `ia_primitives=1`
- `vs_invocations=3`
- `ps_invocations>0`
- `pbe_pixels_written>0`
- output PNG contains both opaque red triangle pixels and opaque black clear
  pixels

The registered native smoke suite verifies the checked-in driver/model
contracts and the model boundary. End-to-end Mesa replay still consumes the
separately configured Mesa prefix at runtime:

```bash
cmake --build build --target pvrgpu pvrgpu-model-stub
ctest --test-dir build --output-on-failure
```

The clear smoke creates a surfaceless GLES2 pbuffer, reports
`GL_RENDERER=PvrGPU SystemC Gallium bring-up`, checks `glReadPixels()` returns
`CLEAR_PIXEL=32,64,128,255`, and emits both `driver-command.txt` and
`driver-counter.txt`.

The triangle smoke compiles a minimal GLES2 vertex/fragment shader pair, binds
one client vertex array, calls `glDrawArrays(GL_TRIANGLES, 0, 3)`, verifies
`driver-counter.txt` contains `event=draw_triangles`, and verifies
`driver-command.txt` ends as `command=draw_triangle`.

The native `pvrgpu` process owns the direct RenderDoc → Mesa/Gallium pvrgpu →
SystemC vertical path for one capture:

```bash
build/bin/pvrgpu \
  /path/to/triangle.rdc \
  --case driver_triangle_solid \
  --width 16 --height 16 \
  --outdir /tmp/pvrgpu-triangle
```

The batch worker may supply the equivalent `--rdc /path/to/triangle.rdc`
form.

It verifies the model JSONL, counters, memory-pool leak status, and final PNG.
For the current fixture, a 16×16 run should contain 128 opaque red triangle
pixels and 128 opaque black clear pixels. Directory comparison treats the
decoded RGBA8 PNG as a PASS gate, not as an informational artifact.

The Phase 3 state smoke is also counter-only. It exercises fixed-function
GLES2 state object traffic for blend, depth/stencil/alpha, rasterizer, blend
color, stencil ref, scissor, and one indexed `glDrawElements(GL_TRIANGLES, 3,
GL_UNSIGNED_SHORT, ...)` call. The expected evidence is driver-counter events
such as `create_blend_state`, `bind_depth_stencil_alpha_state`,
`set_blend_color`, `set_stencil_ref`, `set_scissor`, and
`draw_indexed_triangles`. No Phase 3 draw/state command is consumed by the
model yet.

The generic Phase 4 texture smoke still proves state plumbing through counters.
In addition, the strictly matched glmark2 `effect2d` workload is model-consumed:
the driver emits `draw_textured_triangles` plus a tight RGBA8 sidecar, and the
SystemC path runs the six-vertex draw, depth test/write, varying interpolation,
nearest texture sampling, and framebuffer readback. Unsupported texture draws
still emit `unsupported_draw`; they are never replaced by presentation pixels.
Batch acceptance requires both the exact normalized 17 counters and an exact
decoded-RGBA PNG match. The command-specific interpolation setup also preserves
llvmpipe's BACK/CW v0/v1 normalization, binary32 coefficient order, and integer
pixel-offset evaluation; those details are observable at exact nearest-texel
boundaries and are covered by the SystemC API bridge regression test.

The Phase 5 FBO smoke creates a texture-backed FBO, verifies
`glCheckFramebufferStatus(GL_FRAMEBUFFER)` is complete, clears the FBO and
checks `glReadPixels()` returns the expected clear color, copies that image to
another same-format 2D texture through `glCopyTexSubImage2D`, attaches the copy
texture and checks the copied pixel, then issues one triangle draw plus
`glFlush()`/`glFinish()`. The expected evidence is driver-counter events such
as `set_framebuffer_state`, FBO-sized `clear_color`, `resource_copy_region` or
`blit`, `draw_triangles`, and `flush`. No Phase 5 FBO, copy/blit, draw, or sync
command is consumed by the model yet.

The Phase 6 uniform smoke is also counter-only. It binds a GLES2 fragment
shader using a `vec4[8]` uniform array, uploads data with `glUniform4fv`, and
issues one triangle draw. The expected evidence is driver-counter events such
as `set_constant_buffer stage=fragment ... has_buffer=1`, payload
`first_words`, and `draw_uniform_triangles`. No Phase 6 uniform or UBO command
is consumed by the model yet.

The Phase 7 indexed-quad path is model-consumed. It recognizes tightly scoped
RDC/dEQP full-screen indexed quad draws, emits `command=draw_indexed_quad`, and
keeps updating that command until the current supported batch is complete. For
texture-filtering captures the command also carries `semantic_texel_fetches`,
derived from the observed Gallium sampler state and viewport size, so the
exported 17-counter report can compare against RenderDoc's llvmpipe counter
view without treating NIR or PowerVR ISA instruction counts as equivalent.
The current dEQP texture-filtering regression hint covers 2D, 2D array, 3D,
cube, sRGB, ETC1, and wrap/filter combination captures in the checked-in
100-frame sample set.

The depth-never reuse is temporary. It keeps the first path executable while a
dedicated clear engine and later transfer/readback hooks are developed.

## Next contract extensions

Grow the contract only when a dEQP capture needs it:

1. scissor clear rectangle
2. depth/stencil clear
3. transfer/readback command
4. draw command with fixed minimal shader subset
5. texture/sampler descriptors and upload payload references
6. fence/sync/reporting records

Each extension should add a unit test and one end-to-end model smoke before it
is used by the dEQP capture runner.
