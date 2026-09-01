# Mesa PCO lowering boundary

`src/gallium/drivers/pvrgpu/pvrgpu_pco.c` is the runtime compiler boundary
between Gallium NIR and Mesa's public PowerVR PCO backend.  It does not contain
a project-local assembler or instruction encoding.  The compiler owns one PCO
context for the public `gx6250` target, clones caller-owned VS/FS NIR, runs the
public preprocess/link/reverse-link/lower/postprocess/translate/process/encode
pipeline, and copies the resulting stage bytes into driver-owned arrays.

The initial entry point is deliberately fail-closed and supports only the
GLBench/glmark2 conditionals profile:

- one `PIPE_FORMAT_R32G32B32_FLOAT` `VERT_ATTRIB_GENERIC0` input;
- a VS writing only `gl_Position` and using four CB0 vec4 matrix slots;
- an FS reading `gl_FragCoord.xy`, writing one RGBA color, and using one CB0
  vec4 Y-transform slot;
- the observed `ffract`/`fge`/`bcsel` data-flow signature, with no textures,
  buffers, dynamic uniform indexing, or NIR control flow.

The captured GLES fragment shader explicitly keeps `FragCoord` highp but uses
the default mediump precision for its local `d`.  llvmpipe represents the `d`
chain as f16 ALU.  Until PCO advertises native 16-bit ALU, the compiler boundary
models each f16 result with explicit `f2f16_rtne`/`f2f32` pairs around f32 ALU;
Mesa PCO emits public F16 pack/unpack instructions for those boundaries.  This
matches the reference without treating color dither or a post-render pixel
tolerance as shader semantics.

The profile first validates the raw Gallium NIR, then compiles clones using the
PCO NIR options.  The clones are marked internal as a model/compiler canonical
profile: SystemC owns raster/depth behavior, so the shader bytes omit the
Vulkan-specific ISP feedback and default point-size export.  The FS color store
is canonicalized to four PCO float pixel outputs; the model PBE owns RGBA8
packing.  This is deterministic model ABI, not a claim that the bytes are
identical to compiling the untouched application NIR as a Vulkan pipeline.

## Uniform addressing

Gallium `load_uniform` offsets are vec4 slots because this screen does not
advertise packed uniforms.  NIR `load_push_constant` offsets are bytes, while
`pco_nir_lower_io()` later shifts those byte offsets right by two to obtain
DWORD/shared-register indices.  The required mapping is therefore:

```
push_byte_offset = (load_uniform.base + vec4_slot) * 16
```

For the VS matrix, slots 0, 1, 2, and 3 consequently become SH0, SH4, SH8, and
SH12 and span SH0..15.  Multiplying the slot by four at the Gallium-to-NIR seam
is wrong: PCO would divide those values again and overlap the matrix rows.  The
FS preserves the complete four-DWORD CB0 ABI at SH0..3 even though the shader
only consumes its first two words.

The public result struct contains only stable owned bytes and compact POD ABI
metadata.  It intentionally does not expose or serialize `pco_data`, which has
Mesa-version-specific pointers.  Call
`pvrgpu_pco_graphics_binary_finish()` after the command bridge has deep-copied
the byte arrays.

## Mesa build integration

The PvrGPU driver Meson file links `libpowervr_compiler` and
`libpowervr_common`.  Mesa normally enters `src/imagination` only for the
Imagination Vulkan driver or tools, so
`third_party/mesa-gallium-pvrgpu-pco.patch` records the two outer Meson
condition changes needed when `with_gallium_pvrgpu` is enabled.  Apply that
tracked patch to the selected Mesa source tree; do not rely on an unrecorded
edit in a developer checkout.

With Mesa tests enabled, the `pvrgpu_pco_lowering` native test compiles a
Gallium-style conditionals shader pair, checks clone ownership, non-empty owned
binaries, VS SH0..15/FS SH0..3 metadata, and fail-closed vertex-format handling.
It also pins the exact `gx6250` runtime output consumed by the SystemC decoder:
the 520-byte VS has FNV-1a64 `88ef7e84a69a0db7`, and the 520-byte FS has
FNV-1a64 `e33aaff7bc4d515c`.  A mismatch is a compiler/ABI regression to
investigate, not a fixture hash to update casually.
