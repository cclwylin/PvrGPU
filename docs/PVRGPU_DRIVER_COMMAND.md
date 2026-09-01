# PvrGPU driver command contract v1

This document defines the first narrow contract between the Mesa Gallium
`pvrgpu` driver skeleton and the PvrGPU model.

The intent is not to create a permanent production command stream. It is a
bring-up seam: small enough to debug quickly, strict enough to prevent fake
passes, and close enough to Gallium state that the driver can grow phase by
phase.

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
