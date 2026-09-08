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

## Fragment shader texture LOD bias

The color-primitive compiler accepts fragment `txb` for bound, non-shadow,
non-array 2D, 3D and cube textures with a scalar 32-bit floating-point bias.
Other shader stages and shadow, array or sparse bias operations remain
fail-closed. Shader LOD bias is distinct from sampler-state `lod_bias`; this
compiler support does not expand the sampler-state contract.

No additional external Mesa patch is required for this operation. The pinned
`pco_nir_tex.c` already appends the original bias SSA value after the texture
coordinates, sets `PPLOD` and `PCO_LOD_MODE_BIAS`, and preserves those flags
through `pco_trans_nir.c` to public SMP encoding. Cube bias keeps its original
three direction coordinates; it is not the separate cube-gradient lowering.
The native consumer must obtain implicit derivatives from the complete
fragment quad, add the per-lane shader bias before the effective LOD clamps,
and perform real texture fetches. Replacing `txb` with `txl`, selecting a mip
on the host, or dropping helper-lane bias is not equivalent.

Compiler acceptance alone does not establish native support. Regression
coverage must compile genuine NIR texture operations through Mesa PCO, decode
their emitted BIAS operands, and exercise the native request/FIFO consumer
with positive, negative and dynamic biases, LOD clamps and distinct mip data.
Explicit-LOD and unbiased controls, plus the rejected stage/shape cases, keep
the new path separate from existing contracts.

Run the isolated current-driver producer and native consumer verification with:

```bash
bash script/run_mesa_pco_texture_bias_unit.sh
```

It compiles the repository's driver source explicitly, using the selected Mesa
build only for compiler flags and dependencies, and writes private artifacts
without installing a runtime. The native ASan/UBSan test exercises real PCO
through raw/prepared ISS, USC continuations and TextureUnit memory fetches.
`texture-bias-unit` provides a CTest entry using checked-in genuine PCO fixtures;
the script also regenerates instructions with the current compiler. Include the
separate stage-rejection and existing unbiased/explicit-LOD tests when changing
the request ABI. Host FIFO allocation sizes are not GPU memory traffic.

## Mesa build integration

The PvrGPU driver Meson file links `libpowervr_compiler` and
`libpowervr_common`.  Mesa normally enters `src/imagination` only for the
Imagination Vulkan driver or tools, so
`third_party/mesa-gallium-pvrgpu-pco.patch` records the two outer Meson
condition changes needed when `with_gallium_pvrgpu` is enabled.  Apply that
tracked patch to the selected Mesa source tree; do not rely on an unrecorded
edit in a developer checkout.

The configured Mesa 26.2.1 source also requires
[`mesa-26.2.1-pco-register-bank-copies.patch`](../third_party/mesa-26.2.1-pco-register-bank-copies.patch).
PCO may allocate vector components in either TEMP or VTXIN. The patch removes
TEMP-only array indexing from parallel-copy scheduling, keeps register-bank
dependencies distinct, and preserves source modifiers while breaking cycles.
It applies to real compiler IR independently of capture or test names.

The repository CMake build uses an existing Mesa installation; it does not
apply external Mesa patches or rebuild that installation. Before configuring
a fresh Mesa tree, or rebuilding an existing one, run this idempotent step
from the PvrGPU repository root after setting up `config/local.env`:

```bash
set -a
source config/local.env
set +a
pvrgpu_ra_patch="$PWD/third_party/mesa-26.2.1-pco-register-bank-copies.patch"
if git -C "$PVRGPU_MESA_POC_SOURCE_DIR" apply --reverse --check "$pvrgpu_ra_patch" 2>/dev/null; then
    printf '%s\n' 'PCO register-bank copy patch is already applied.'
elif git -C "$PVRGPU_MESA_POC_SOURCE_DIR" apply --check "$pvrgpu_ra_patch"; then
    git -C "$PVRGPU_MESA_POC_SOURCE_DIR" apply "$pvrgpu_ra_patch"
else
    printf '%s\n' 'Mesa source differs from the pinned patch; resolve the mismatch before building.' >&2
    exit 1
fi
```

For an already configured Mesa build, rebuild the compiler, run the isolated
ASan/UBSan parallel-copy test, and then rebuild/install the configured runtime:

```bash
ninja -C "$PVRGPU_MESA_PVRGPU_BUILD_DIR" src/imagination/pco/libpowervr_compiler.a
bash script/run_mesa_pco_register_bank_unit.sh
ninja -C "$PVRGPU_MESA_PVRGPU_BUILD_DIR" install
```

The isolated test includes the selected external `pco_ra.c`, emits real PCO
MBYP/MBYP2/MOVS1 instructions, and compares their sequential execution against
parallel assignment across all registers. Its 20,016 directed/randomized cases
cover VTXIN-only copies, equal indices in different banks, cycles, repeated
sources, constants/immediates/special registers, source modifiers, and execution
conditions. Test outputs stay in a private temporary directory; the test itself
does not apply patches or alter the installed Mesa runtime.

The selected Mesa 26.2.1 source also requires
[`mesa-26.2.1-nir-shrink-load-footprint.patch`](../third_party/mesa-26.2.1-nir-shrink-load-footprint.patch).
NIR may combine adjacent memory loads and then trim unused leading components.
When the remaining component count rounds up to a supported vector width, the
new load must still end within the original byte footprint. For example,
reading components 3 and 7 of a vec8 cannot become a vec8 load starting at
component 3. The patch limits the leading trim before adjusting offsets and
use swizzles; it does not enlarge a resource or relax runtime bounds.

Apply this tracked patch using the same reverse-check / forward-check workflow
above, substituting its path. Then run the isolated real-NIR regression before
rebuilding a private runtime:

```bash
bash script/run_mesa_nir_shrink_load_unit.sh
```

The test exercises the actual selected shrink pass with ASan/UBSan, checks every
fetched byte and selected output, and includes the real Mesa load-vectorizer
followed by shrink. It covers UBO, SSBO and global loads at different offsets,
8/16/32/64-bit elements, and vec2/3/4/5/8/16 component-use masks. The script reads
the configured Mesa build's compile/link commands but never builds shared
objects, applies patches, or installs a runtime. Set
`PVRGPU_NIR_SHRINK_TEST_SOURCE` to an unpatched source copy for a negative
control; the adjacent-vec4 regression must fail its original 32-byte bound.

With Mesa tests enabled, the `pvrgpu_pco_lowering` native test compiles a
Gallium-style conditionals shader pair, checks clone ownership, non-empty owned
binaries, VS SH0..15/FS SH0..3 metadata, and fail-closed vertex-format handling.
It also pins the exact `gx6250` runtime output consumed by the SystemC decoder:
the 520-byte VS has FNV-1a64 `88ef7e84a69a0db7`, and the 520-byte FS has
FNV-1a64 `e33aaff7bc4d515c`.  A mismatch is a compiler/ABI regression to
investigate, not a fixture hash to update casually.
