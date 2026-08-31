# PvrGPU counter adapter protocol v1

## Boundary

The Qt process never loads Mesa or SystemC in-process. It launches a fresh
runner and consumes either:

- `Report.md` from the current patched llvmpipe Mesa; or
- newline-delimited JSON plus framebuffer PNG artifacts from a PvrGPU
  runner/model.

This keeps Qt's event loop independent from `sc_start()` and makes backend
selection deterministic.

The currently executable SystemC functional slice supports four Fill.Solid
state cases, three indexed Triangle.Setup cases, and the one-/two-/four-/eight-attribute
`attribute_fetch_shader` cases, plus `varyings_shader_1/2/4/8`,
`fill_tex_nearest`, and `fill_tex_bilinear`. Inputs are built-in fixtures
matching those exact GLBench workloads; the slice does **not** yet ingest a
Mesa command stream. Modules communicate through bounded FIFOs
carrying generation-checked MemoryPool handles, while vertex
resources/bindings, geometry, interpolation coefficients, fragment tasks and
RGBA framebuffer payloads remain in the MemoryPool. Completed frames are
published as PNG files next to the JSONL run output.

## JSONL envelope

Each line is UTF-8 JSON, at most 1 MiB, and contains either the legacy schema
name or the explicit protocol/version pair:

```json
{
  "protocol": "pvrgpu-jsonl",
  "version": 1,
  "schema": "pvrgpu.counter.v1",
  "type": "counter"
}
```

Supported MVP message types are:

- `hello`: backend, renderer, profile, fidelity, calibration, cache-bypass
  configuration and warnings;
- `counter`: one record/frame with a flat `counters` object;
- `done`: terminal summary and resource-leak counts;
- `error`: producer-side failure.

The SystemC producer reports the cache mode once in `hello`. It is JSON
metadata, not a numeric counter:

```json
{
  "protocol": "pvrgpu-jsonl",
  "version": 1,
  "schema": "pvrgpu.counter.v1",
  "type": "hello",
  "backend": "pvrgpu",
  "source": "pvrgpu-systemc",
  "cache_bypass": false,
  "memory_mode": "cache",
  "cache_simulated": true
}
```

`memory_mode` is `direct`, `bypass`, or `cache`. `direct` uses the unified DRAM
backing directly for fast functional simulation and does not report modeled
cache/DRAM traffic. `bypass` skips cache lookup/allocation but still reports DRAM
transactions. `cache` is the default and simulates the shared SLC. The legacy
`cache_bypass` boolean is still emitted for compatibility and is true only in
`bypass` mode.

Example sample:

```json
{
  "protocol": "pvrgpu-jsonl",
  "version": 1,
  "schema": "pvrgpu.counter.v1",
  "type": "counter",
  "source": "pvrgpu-systemc",
  "provenance": "modeled",
  "functional_scope": "fill_solid-pco-iss-v1",
  "command_source": "builtin-glbench-fixture",
  "timing_provenance": "uncalibrated",
  "frame": 12,
  "marker": "fill_solid",
  "artifact_png": "/external/run/png/fill_solid_sample_000012.png",
  "virtual_time_ns": 327680,
  "counters": {
    "ps_invocations": 262144,
    "tiles_binned": 256,
    "tiles_scheduled": 256,
    "covered_pixels": 262144,
    "fragment_candidates": 262144,
    "hsr_rejected_fragments": 0,
    "pco_instructions": 6,
    "vs_alu_instructions": 0,
    "vs_tex_instructions": 0,
    "vs_memory_instructions": 16,
    "fs_alu_instructions": 1048576,
    "fs_tex_instructions": 0,
    "fs_memory_instructions": 0,
    "pbe_color_reads": 0,
    "pbe_blended_fragments": 0,
    "pbe_fragment_writes": 262144,
    "pbe_pixels_written": 262144,
    "virtual_gpu_cycles": 24580,
    "fifo_stall_events": 3,
    "pool_bytes_in_flight": 8192
  },
  "drawlist_stats": [
    {
      "drawlist": 0,
      "draw_id": 0,
      "vs": {
        "invocations": 4,
        "program": {
          "groups": 2,
          "instructions": 2,
          "alu": 0,
          "tex": 0,
          "memory": 2
        },
        "executed": {"alu": 0, "tex": 0, "memory": 16}
      },
      "fs": {
        "invocations": 262144,
        "program": {
          "groups": 4,
          "instructions": 4,
          "alu": 4,
          "tex": 0,
          "memory": 0
        },
        "executed": {"alu": 1048576, "tex": 0, "memory": 0}
      }
    }
  ]
}
```

## Provenance classes

| Class | UI badge | Meaning |
|---|---|---|
| `reported` | `REP · llvmpipe` | Directly emitted by the software renderer instrumentation |
| `observed` | `OBS · host` | Host process/OS observation, not a GPU counter |
| `derived` | `DRV` | Formula computed from named input counters |
| `modeled` | `MOD · PvrGPU` | Real SystemC model output, with fidelity/calibration metadata |
| `mock` / `mock-derived` | `MOCK · counter-only` | Protocol fixture; forbidden as a benchmark claim |

The SystemC producer must additionally report `profile`, `fidelity`
(`F0`–`F3`), `calibration` (`P0`–`P2`), and `config_hash` in `hello` or run
metadata. For the current slice it must also identify
`functional_scope=<case>-pco-iss-v1`, `command_source=builtin-glbench-fixture`,
`timing_provenance=uncalibrated`.
`<case>` is the exact canonical case token selected for that run, so scope is
never silently inherited from another fixture; examples include
`fill_solid-pco-iss-v1`, `triangle_setup-pco-iss-v1`, and
`attribute_fetch_shader-pco-iss-v1`.

The active tile dimension is 32×32 because the user selected it as a model
specification. It is not source-derived PowerVR evidence. Public BXS 16×16
tile information may be recorded separately as reference metadata, but must
not be used to claim public support for the active 32×32 choice.

## DrawList shader instruction accounting

The current SystemC producer includes `drawlist_stats[]` in each `counter`
message. Each entry identifies one DrawList and splits vertex (`vs`) and
fragment (`fs`) statistics into:

- `program`: static decoded program composition. `groups`, `instructions`,
  `alu`, `tex`, and `memory` count semantic program items once, before PCO
  repeat expansion and independently of shader invocation count.
- `executed`: dynamic ALU/Tex/Memory totals. Non-textured straight-line
  programs use `sum(instruction.repeat_count) × executed lanes`. A texture
  program also counts raster/helper lanes that genuinely execute `SMP` and
  wait for a response; logical `ps_invocations` continues to count covered
  fragments rather than helper work.

The Fill.Solid reference fixture has this exact accounting:

| Stage | Invocations | Static groups / instructions | Static ALU / Tex / Memory | Dynamic ALU / Tex / Memory |
|---|---:|---:|---:|---:|
| VS | 4 | 2 / 2 | 0 / 0 / 2 | 0 / 0 / 16 |
| FS | `width × height` | 4 / 4 | 4 / 0 / 0 | `4 × width × height` / 0 / 0 |

The VS program consists of a `UVSW.write` with `repeat_count=3` and a final
`UVSW.write.emit.endtask` with repeat 1: four Memory/export operations per
vertex invocation and 16 dynamically executed operations for four vertices.
The FS program executes four `MBYP` ALU/move instructions per fragment.
Accordingly, `pco_instructions=6` is a **static** total (two VS plus four FS
semantic instructions), not an executed total. Static `groups` and
`instructions` happen to be equal in this subset but remain distinct fields.

The exact 64×64 `attribute_fetch_shader` case-1 gate reports:

| Stage | Invocations | Static groups / instructions | Static ALU / Tex / Memory | Dynamic ALU / Tex / Memory |
|---|---:|---:|---:|---:|
| VS | 5,317 | 6 / 6 | 4 / 0 / 2 | 21,268 / 0 / 26,585 |
| FS | 0 | 4 / 4 | 4 / 0 / 0 | 0 / 0 / 0 |

Its VS has four `MBYP` moves plus a repeated `UVSW.write` and standalone
`UVSW.emit.endtask`; repeat expansion therefore executes four ALU and five
Memory/export operations per post-transform-cache miss. The gray FS is decoded
and included in the static program total, but BACK/CCW culling rejects all
8,192 clockwise setup candidates before fragment execution. Consequently this
case reports `pco_instructions=10` as its static VS+FS semantic total.

The exact two-attribute case has the same DrawList table: two public-PCO
`FADD` groups replace two moves, while static and repeat-expanded ALU/Memory
classes remain `4/0/2` and `4/0/5` per vertex invocation. The ISS result is
`(2x,2y,0,2)` for every lane and perspective divide preserves the case-1 NDC.
FADD uses a pure-integer binary32 round-to-nearest-even implementation for
finite-normal/signed-zero operands; unsupported NaN, Inf, subnormal and
overflow policies fail closed instead of inheriting host floating-point mode.

The exact four-attribute VS has 10 groups/instructions: six public-PCO `FADD`,
two `MBYP`, one repeated `UVSW.write`, and one `UVSW.emit.endtask`. Its static
ALU/Tex/Memory classes are `8/0/2`; each lane executes `8/0/5`, so 5,317
invocations report VS dynamic `42536/0/26585`. The unchanged gray FS is decoded
but has zero invocations, making the combined static total
`pco_instructions=14`. Four float2 bindings alias one VBO resource, feed
VTXIN0..7, and execute the exact clip result `(4x,4y,0,4)` before perspective
divide and BACK/CCW culling.

The exact eight-attribute VS has 18 groups/instructions: 14 public-PCO `FADD`,
two `MBYP`, one repeated `UVSW.write`, and one `UVSW.emit.endtask`. Its static
ALU/Tex/Memory classes are `16/0/2`; each lane executes `16/0/5`, so 5,317
invocations report VS dynamic `85072/0/26585`. The unchanged gray FS is
decoded but has zero invocations, making the combined static total
`pco_instructions=22`. Eight float2 bindings alias one VBO resource, feed
VTXIN0..15, and execute the exact clip result `(8x,8y,0,8)` before perspective
divide and BACK/CCW culling.

The exact 64×64 `varyings_shader_2` gate has two 12-group programs:

| Stage | Invocations | Static groups / instructions | Static ALU / Tex / Memory | Dynamic ALU / Tex / Memory |
|---|---:|---:|---:|---:|
| VS | 25 | 12 / 12 | 8 / 0 / 4 | 200 / 0 / 325 |
| FS | 4,096 | 12 / 12 | 10 / 0 / 0 | 40,960 / 0 / 0 |

The VS exports position plus two copies of `c/2`; the FS performs two
perspective `FITRP` requests and four component-wise `FADD` operations for
`v1+v2`. `WDF` remains dependency-control work and is not misclassified as
ALU. The exact parameter path produces 288 coefficient sets / 4,608 bytes,
1,152 PDS tasks / 2,304 DOUTI issues, and 165,888 USC coefficient-load bytes.

The exact 64×64 `varyings_shader_4` gate reports:

| Stage | Invocations | Static groups / instructions | Static ALU / Tex / Memory | Dynamic ALU / Tex / Memory |
|---|---:|---:|---:|---:|
| VS | 25 | 14 / 14 | 8 / 0 / 6 | 200 / 0 / 525 |
| FS | 4,096 | 24 / 24 | 20 / 0 / 0 | 81,920 / 0 / 0 |

Its VS exports position plus four copies of `c/4`; its FS performs four
perspective `FITRP`/`WDF` pairs and 12 component-wise `FADD` operations for
the left-associative sum. The parameter path produces 544 sets / 8,704 bytes,
1,152 PDS tasks / 2,304 DOUTI issues, and 313,344 USC coefficient-load bytes.

The exact 64×64 `varyings_shader_8` gate reports:

| Stage | Invocations | Static groups / instructions | Static ALU / Tex / Memory | Dynamic ALU / Tex / Memory |
|---|---:|---:|---:|---:|
| VS | 25 | 18 / 18 | 8 / 0 / 10 | 200 / 0 / 925 |
| FS | 4,096 | 48 / 48 | 40 / 0 / 0 | 163,840 / 0 / 0 |

Its VS exports position plus eight copies of `c/8`; its FS performs eight
perspective `FITRP`/`WDF` pairs and 28 component-wise `FADD` operations for
the encoded left-associative sum. The parameter path produces 1,056 sets /
16,896 bytes, 1,152 PDS tasks / 2,304 DOUTI issues, and 608,256 USC
coefficient-load bytes. The complete frame reports `pco_instructions=66`,
`pco_decode_cycles=23`, and `virtual/tiler/renderer=1392/58/1309`; these
cycle-equivalent values remain assumed and uncalibrated.

The DrawList **Memory** class means PCO memory/export instructions and includes
UVSW output/export. Its unit is `instructions`; it is not bytes transferred,
MemoryPool occupancy, cache traffic, or DRAM traffic. Byte quantities use
separate fields such as `pool_bytes_in_flight`, `pixel_data_master_bytes`,
`dram_read_bytes`, and `dram_write_bytes`.

A DrawList **Texture instruction** is also distinct from both
`texture_requests` and `texel_fetches`: the first is repeat-expanded PCO
texture instruction execution, the second is a modeled logical TPU request,
and the third counts sampler fetch lanes including filtering. Predication,
LOD, filtering, helper-lane execution, and sampling can make these quantities
differ. Non-textured cases report zero for all three; a texture case reports
the work actually executed rather than deriving it from logical
`ps_invocations`. One executed `SMP` creates one logical `texture_requests`
event. Nearest filtering expands that request to one real texel fetch and one
TCU access, whereas bilinear filtering expands the same request to four real
texel fetches and four TCU accesses. The DrawList texture count therefore
remains shader-level work; `texel_fetches` and TCU access counters are
filter-tap-level work.

### Vertex-input counter meanings

| Field | Unit | Meaning |
|---|---|---|
| `vertex_attribute_fetches` | logical binding fetches | One fetch for each active vertex-attribute binding on every newly generated shader lane. On an indexed draw, a new lane is a post-transform-cache miss; repeated indices that hit the cache do not fetch again. |
| `vertex_attribute_bytes` | bytes | Sum of `source_components × component_size` for those logical binding fetches. Destination expansion such as the GLES float2→vec4 defaults z=0 and w=1 does not read VBO bytes and is excluded. |

These fields describe logical VBO reads from MemoryPool-owned resources, not
SLC/DRAM transactions or cache-line traffic. The exact case-1 fixture has one
float32×2 binding, so its 5,317 misses report
`vertex_attribute_fetches=5317` and `vertex_attribute_bytes=42536`.
The two-attribute case aliases both float2 bindings to the same VBO resource,
so the same 5,317 misses produce `vertex_attribute_fetches=10634`,
`vertex_attribute_bytes=85072`, and `vertex_fetch_cycles=46`.
The four-attribute case still owns only one VBO resource; its four bindings
produce `vertex_attribute_fetches=21268`, `vertex_attribute_bytes=170144`, and
`vertex_fetch_cycles=88`.
The eight-attribute case likewise owns one VBO resource; its eight bindings
produce `vertex_attribute_fetches=42536`, `vertex_attribute_bytes=340288`, and
`vertex_fetch_cycles=171`.

### Executed vertex-program identity

Every modeled `counter` message carries `vertex_pco_binary` and
`vertex_pco_opcodes`. The reporter computes the FNV-1a-64 fingerprint from the
actual MemoryPool shader bytes and the opcode histogram from the actual
decoded semantic IR before releasing either payload. These are program
identity/provenance fields, not performance counters. The cryptographic
SHA-256 fixture pins remain in `third_party/mesa-pco.lock` and the PCO unit
gate; the runtime fingerprint lets a differential artifact prove which of
those compiled payloads was executed.

For the four attribute gates the expected evidence is:

| Case | Binary fingerprint / bytes | FADD | MBYP | UVSW write | UVSW emit/end-task |
|---|---|---:|---:|---:|---:|
| `attribute_fetch_shader` | `fnv1a64:48cf8717db7aa8cf` / 56 | 0 | 4 | 1 | 1 |
| `attribute_fetch_shader_2_attr` | `fnv1a64:4fb7f3aba4b44c19` / 56 | 2 | 2 | 1 | 1 |
| `attribute_fetch_shader_4_attr` | `fnv1a64:c54ea51cdaab08a0` / 96 | 6 | 2 | 1 | 1 |
| `attribute_fetch_shader_8_attr` | `fnv1a64:87d4d7b5e46ff241` / 176 | 14 | 2 | 1 | 1 |

`varyings_shader_1` additionally carries both stage identities. Its 72-byte VS
is `fnv1a64:c1a8f8a4f58fc81f` with four `MBYP`, two `UVSW.write`, and one
`UVSW.emit.endtask`; its 48-byte FS is `fnv1a64:76458bbcec6f53bf` with one
`FITRP`, one dependency-control `WDF`, and four `MBYP`. The corresponding
SHA-256 pins are
`09636842506c3a05b4dfae96d232274bb2eeb59876591e9fe29fc27a2e0860df` and
`a9c070ea3feb5dc4f7666b1fc019aaa9e3c522f5a8a73605ea07481104efc71c`.

`varyings_shader_2` carries a 120-byte VS
`fnv1a64:ffd408e5a8ae5f7c` and a 104-byte FS
`fnv1a64:9c2a8c68ef09d5d1`. Their SHA-256 pins are
`11a9256581cec718761818f8907337c86e458d2e44884ffe89a8d20c44647535`
and `8c3c5427a0064009d8799a120f3e34645031f8c73b15a30ca224f0b007e21e99`.
The hello evidence names the executed subset
`fmul-fitrp-wdf-fadd-mbyp-uvsw-varying`; raw-byte and semantic mutations fail
closed in the PCO unit gate.

`varyings_shader_4` carries a 136-byte VS
`fnv1a64:a654c923dfea45ef` and a 216-byte FS
`fnv1a64:1dbf7c0552b6b385`. Their SHA-256 pins are
`d98cefa0385a774d1a7b0ddb0149cc6b5aca3023cccd287e3eeea1ca410f6538`
and `f5c1fbac1b9281ce5093ba9c629c90ff5cd81e1807351f3bee2f1f5700f1a08a`.
The VS opcode evidence is `FMUL×2, MBYP×6, UVSW.write×5, emit×1`; the FS
evidence is `FITRP×4, WDF×4, FADD×12, MBYP×4`.

`varyings_shader_8` carries a 176-byte VS
`fnv1a64:3ea4e650a43484ce` and a 440-byte FS
`fnv1a64:b1f3b2aa7d58d59d`. Their SHA-256 pins are
`f5314dcc5a24dca2c7d716b9d0c3bd1696df0038e826b34ce7f7e208945bb45a`
and `aaebb7b4e027f846eecda4687dbc14fb10dc8b9bb3881ef0134cd0255449c385`.
The VS opcode evidence is `FMUL×2, MBYP×6, UVSW.write×9, emit×1`; the FS
evidence is `FITRP×8, WDF×8, FADD×28, MBYP×4`. Both the binary identity
and decoded histogram are emitted from the MemoryPool-owned programs.

The case-4 96-byte and case-8 176-byte raw USC binaries are pinned by SHA-256
`81b4bf2b412eb2ba35adcd1076d965918336ffb0ffb860e66547695ef4a6ae28` and
`877802fe53fd258bb114aa2cf5713c317405c986b3a43b4612a58b6db9f7eccb`,
respectively. They are generated by `tools/pco-fixtures/generate_attribute_fetch_shader.c`
through Mesa 26.2.1 commit
`da14d65e4499e66468094be52bff9ea0915a695e` for the public `gx6250` target.
The runtime embeds those immutable bytes and has no Mesa compiler dependency.

## Stable counter naming

The MVP preserves the existing Mesa field names so the current report remains
usable. PvrGPU-only fields use semantic names such as:

- `virtual_gpu_cycles`
- `tiler_cycles`
- `renderer_cycles`
- `usc_groups`
- `texture_requests`
- `fifo_stall_events`
- `pool_bytes_in_flight`
- `pool_high_water_bytes`
- `vdm_cycles`, `vertex_fetch_cycles`
- `vertex_attribute_fetches`, `vertex_attribute_bytes`
- `pco_decode_cycles`, `pco_instructions`, `usc_slot_cycles`, `usc_cluster_cycles`
- `vs_alu_instructions`, `vs_tex_instructions`, `vs_memory_instructions`
- `fs_alu_instructions`, `fs_tex_instructions`, `fs_memory_instructions`
- `clip_cull_cycles`, `tiler_bin_cycles`, `parameter_buffer_cycles`
- `parameter_coefficient_sets`, `parameter_write_bytes`
- `pds_coefficient_tasks`, `pds_douti_issues`, `usc_coefficient_load_bytes`
- `tile_scheduler_cycles`, `isp_cycles`, `fragment_frontend_cycles`
- `texture_cycles`, `pbe_cycles`
- `pixel_data_master_transactions`, `pixel_data_master_bytes`,
  `pixel_data_master_cycles`
- `tcu_line_accesses`, `tcu_read_accesses`
- `tcu_hits`, `tcu_misses`, `tcu_evictions`, `tcu_writebacks`,
  `tcu_bypassed`, `tcu_cycles`
- `slc_line_accesses`, `slc_read_accesses`, `slc_write_accesses`
- `slc_hits`, `slc_misses`, `slc_evictions`, `slc_writebacks`,
  `slc_bypassed`, `slc_cycles`
- `dram_read_transactions`, `dram_write_transactions`, `dram_read_bytes`,
  `dram_write_bytes`, `dram_cycles`
- `framebuffer_dram_readback_bytes`
- `tiles_binned`: row-major 32×32 tile records produced by the current tiler
- `tiles_scheduled`: non-empty tile records submitted to ISP
- `covered_pixels`: unique framebuffer pixels covered by ISP
- `fragment_candidates`: primitive/sample candidates evaluated before HSR
- `hsr_rejected_fragments`: candidates eliminated before fragment USC
- `pbe_color_reads`: destination colors read for fixed-function blending
- `pbe_blended_fragments`: fragment outputs evaluated by the blend equation
- `pbe_fragment_writes`: post-blend fragment colors written to the render target
- `pbe_pixels_written`: RGBA8 pixels committed to the framebuffer

### Parameter/PDS coefficient counter meanings

| Field | Unit | Meaning |
|---|---|---|
| `parameter_coefficient_sets` | sets | Four-dword A/B/C/PAD interpolation coefficient sets written per primitive |
| `parameter_write_bytes` | bytes | Coefficient payload bytes written into the on-chip parameter buffer |
| `pds_coefficient_tasks` | tasks | Fragment-quad coefficient-copy tasks dispatched by PDS |
| `pds_douti_issues` | issues | PDS DOUTI coefficient-transfer issues sent to USC |
| `usc_coefficient_load_bytes` | bytes | Bytes copied into independent USC fragment-task coefficient banks |

These fields measure the real `Parameter Buffer → Fragment Frontend → PDS →
USC` payload path. For perspective-correct interpolation, Parameter Buffer
stores one `1/w` plane plus one `varying/w` plane per varying component;
`FITRP` divides the interpolated `varying/w` value by interpolated `1/w`.
They are not inferred from the case name or the final framebuffer.

For `varyings_shader_2`, each of 32 primitives owns nine sets: one shared
`1/w` plane plus eight component planes for `v1/w` and `v2/w`. Therefore the
exact totals are 288 sets and 4,608 bytes; 1,152 fragment-quad tasks copy 36
dwords each, giving 165,888 coefficient-load bytes.

For `varyings_shader_4`, each primitive owns 17 sets: one shared `1/w` plane
plus 16 component planes for four vec4 varyings. The exact totals are 544
sets / 8,704 bytes, and 1,152 tasks copy 68 dwords each for 313,344 USC
coefficient-load bytes.

For `varyings_shader_8`, each primitive owns 33 sets: one shared `1/w` plane
plus 32 component planes for eight vec4 varyings. The exact totals are 1,056
sets / 16,896 bytes, and 1,152 tasks copy 132 dwords each for 608,256 USC
coefficient-load bytes.

The sealed Gate-15 differential evidence is
`$PVRGPU_WORK_ROOT/differential/glbench/varyings_shader_8-64x64-20260829-175749-18505/`.
It reports exact agreement for all 17 Golden fields and decoded RGBA8
(`ps_invocations=4096`, `drawlists=1`, `pool_leaks=0`, zero differing pixels,
and maximum channel delta zero). Cache bypass is off; PixelDM sends one
16,384-byte request, SLC performs 128 write misses/writebacks, DRAM performs
one read plus 128 writes, and the PNG is published only from the separate
16,384-byte DRAM readback.

### PBE fixed-function blend counter meanings

| Field | Unit | Meaning |
|---|---|---|
| `pbe_color_reads` | fragments | Destination render-target colors read for blend read-modify-write |
| `pbe_blended_fragments` | fragments | Post-depth fragment outputs evaluated by the enabled fixed-function blend equation |
| `pbe_fragment_writes` | fragments | Post-blend fragment colors written to the render target |

These counters describe PBE fixed-function work and are intentionally separate
from `fs_alu_instructions`, `fs_tex_instructions`, and
`fs_memory_instructions`. A fragment blend does not add a shader instruction to
the DrawList totals. With blending disabled, `pbe_color_reads` and
`pbe_blended_fragments` are zero; `pbe_fragment_writes` still counts fragment
color writes that survive the preceding tests.

### TCU, PixelDM, SLC and DRAM counter meanings

| Field | Unit | Meaning |
|---|---|---|
| `tcu_line_accesses` | lines | Legacy TCU cache-line access attempts; unified-memory model runs texture traffic through shared SLC and leaves TCU counters zero |
| `tcu_read_accesses` | lines | Legacy TCU cache-line read accesses issued by TextureUnit |
| `tcu_hits` | lines | Legacy texture reads satisfied by resident TCU lines |
| `tcu_misses` | lines | Legacy texture reads that require an SLC fill |
| `tcu_evictions` | lines | Legacy valid TCU lines displaced by replacement |
| `tcu_writebacks` | lines | Legacy dirty TCU lines written to SLC |
| `tcu_bypassed` | lines | Legacy texture lines forwarded without TCU lookup or allocation |
| `tcu_cycles` | cycles | Legacy event-driven TCU service cycles |
| `pixel_data_master_transactions` | transactions | Framebuffer writeback transactions issued by the PBE writeback boundary |
| `pixel_data_master_bytes` | bytes | Framebuffer payload bytes issued by PixelDM |
| `pixel_data_master_cycles` | cycles | Event-driven PixelDM service cycles |
| `slc_line_accesses` | lines | Cache-line lookups actually performed by SLC |
| `slc_read_accesses` | lines | SLC cache-line read accesses |
| `slc_write_accesses` | lines | SLC cache-line write accesses |
| `slc_hits` | lines | Lookups satisfied by resident SLC lines |
| `slc_misses` | lines | Lookups requiring lower-memory service |
| `slc_evictions` | lines | Valid SLC lines displaced by replacement |
| `slc_writebacks` | lines | Dirty SLC lines written back to DRAM |
| `slc_bypassed` | transactions | Transactions forwarded to DRAM without lookup or allocation |
| `slc_cycles` | cycles | Event-driven SLC service cycles |
| `dram_read_transactions` | transactions | Read transactions serviced by modeled DRAM (`bypass`/`cache` modes only) |
| `dram_write_transactions` | transactions | Write transactions serviced by modeled DRAM (`bypass`/`cache` modes only) |
| `dram_read_bytes` | bytes | Payload bytes returned by modeled DRAM |
| `dram_write_bytes` | bytes | Payload bytes committed by modeled DRAM |
| `dram_cycles` | cycles | Fixed-latency DRAM service cycles |
| `memory_direct_read_bytes` | bytes | Fast functional reads from the authoritative DRAM backing (`direct` mode only) |
| `memory_direct_write_bytes` | bytes | Fast functional writes to the authoritative DRAM backing (`direct` mode only) |
| `framebuffer_dram_readback_bytes` | bytes | Final framebuffer bytes fetched from DRAM for PNG publication |

The unified memory route keeps texture, vertex, index, parameter, and
framebuffer persistent data in one DRAM backing. In `cache` mode GPU clients
access that data through the shared SLC and dirty lines are flushed before final
framebuffer readback. In `bypass` mode cache lookup/allocation is skipped while
DRAM transactions are still modeled. In `direct` mode modules access the same
authoritative backing directly for speed.
With cache active, every 128-byte SLC writeback line is one request and final
PNG readback is one exact-size bulk request; bypass uses one bulk write plus
the same independent readback request. The PNG must be published only after
DRAM readback completes; for a successful
RGBA8 frame, `framebuffer_dram_readback_bytes` therefore records the bytes
actually returned for that artifact.

`pco_instructions` is the static VS+FS semantic-program total. The six
stage/class instruction counters are dynamic, repeat-expanded execution totals
aggregated over DrawLists; `drawlist_stats[]` provides both views and their
per-DrawList breakdown.

Unavailable values are omitted and displayed as `—`; they must not be emitted
as zero merely because a backend cannot measure or model them.

Tile/pixel counters and the six VS/FS instruction counters describe executed
functional work; `pco_instructions` and `drawlist_stats[].*.program` describe
static program composition. Cycle fields remain modeled and uncalibrated;
none of these are hardware-reported counters. The runner publishes each PNG
atomically before announcing the completed capture so the Qt process never
observes a partially written image.

## Future native adapter

The native `pvr_model_winsys`/SystemC runner should preserve this boundary:

1. Qt launches a backend-specific GLBench binary in a new process.
2. Mesa selects the backend before EGL initialization.
3. The model emits counters asynchronously as work completes.
4. The GUI uses the same parser/table/chart without backend-specific widgets.
5. Exit success requires the expected renderer, terminal message/fence, and no
   MemoryPool handle leak.
