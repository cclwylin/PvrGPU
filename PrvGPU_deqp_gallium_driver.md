# PrvGPU dEQP Gallium Driver Plan

> Date: 2026-08-30  
> Goal: use captured dEQP RDC files first, then grow PrvGPU from the current RDC trace-capsule POC into a real Mesa Gallium driver.  
> Principle: dEQP captures drive early debug; on-the-fly dEQP starts only after the capture replay path is stable; RDC validates real captured workloads. The paths should stay separate but share counter/report infrastructure where useful.

## Capture-First Policy

The current Phase 0 policy is **not** to run dEQP binaries directly. The prepared capture corpus is the first bring-up input:

```text
$HOME/Downloads/_Codex/Working/deqp
```

Use this first because it gives stable, repeatable `.rdc` reproducers. If coverage is not enough, record more dEQP samples and add them to the same corpus. Only after the capture path is stable should the project add on-the-fly dEQP execution.

Current capture-first order:

1. Catalog captured dEQP `.rdc` files and map each one back to suite/group/case metadata.
2. Replay selected captures through RenderDoc + Mesa + llvmpipe to produce Golden counters.
3. Route selected captures through the PrvGPU hook. Until lowering exists for that phase, it must return `UNSUPPORTED` and must not write fake counters.
4. Compare `counter_golden.txt` and `counter_pvrgpu.txt` only when both real files exist.
5. Bring up the real Gallium driver hooks against this fixed corpus.
6. Add on-the-fly dEQP only after the capture replay flow is boring and repeatable.

## Why dEQP Captures

The current GLBench RDC path is intentionally narrow:

```text
RenderDoc .rdc
  -> Mesa/POC trace capsule
  -> PrvGPU SystemC/C-model
  -> counter_pvrgpu.txt
```

That is good for validating PrvGPU counter behavior on known frames, but it is not a production Mesa Gallium driver. A real Gallium driver must accept arbitrary Mesa state, resources, shaders, draws, clears, flushes, and synchronization through the normal Mesa frontend.

dEQP is still the right source of tests, but early execution should come from the captured `.rdc` corpus because it removes live harness variability while PrvGPU is still immature. The captures cover driver behavior systematically:

- context, screen, framebuffer, surface, and resource lifetime
- clear, draw, viewport, scissor, and rasterization state
- vertex buffers, index buffers, and vertex element layout
- shader compile, constants, varyings, and program linkage
- texture formats, sampler state, filtering, and mipmaps
- depth, stencil, blend, FBO, flush, fence, and sync behavior

## Target Architecture

Long-term target:

```text
dEQP / GLES app / RenderDoc replay
  |
  v
Mesa state tracker
  |
  v
PrvGPU Gallium driver
  |
  +-- pipe_screen
  +-- pipe_context
  +-- resource / surface / transfer
  +-- shader / state objects
  +-- draw / clear / flush
  |
  v
PrvGPU command lowering
  |
  v
PrvGPU C-model / SystemC
  |
  v
counter / image / trace artifacts
```

The important change is that Mesa calls PrvGPU through normal Gallium hooks. The trace-capsule translator becomes a bring-up aid, not the driver boundary.

## Keep Three Test Paths

### dEQP Official Path

Purpose: grow Gallium frontend completeness through the real conformance-style entry point.

```text
dEQP test
  -> Mesa Gallium state tracker
  -> PrvGPU Gallium driver
  -> PrvGPU C-model/SystemC
  -> dEQP PASS/FAIL + artifacts
```

This path is intentionally delayed. Use it after captured dEQP RDC replay is stable enough that live harness failures are likely to mean real driver problems.

### dEQP Capture Replay Path

Purpose: quickly debug already-captured dEQP tests without re-running the full dEQP harness every time.

Current capture root:

```text
$HOME/Downloads/_Codex/Working/deqp
```

Observed layout:

```text
deqp/
  gles3/
    dEQP-GLES3.functional.color_clear./
      recorder/
        trace/*.rdc
        qpa/*.qpa
        manifest.txt
        Report.md
  gles31/
  egl/
```

Use this path for tight debug loops:

```text
dEQP captured .rdc
  -> RenderDoc replay / trace extraction
  -> PrvGPU Gallium-driver lowering or POC lowering
  -> PrvGPU C-model/SystemC
  -> counter/image/log artifact
```

This path is not a substitute for passing dEQP officially. It is a fast reproduction path after a dEQP case has already been captured.

### RDC App Workload Path

Purpose: validate real captured workloads and counter equivalence.

```text
.rdc
  -> RenderDoc + Mesa + llvmpipe -> counter_golden.txt
  -> RenderDoc + Mesa/POC + PrvGPU -> counter_pvrgpu.txt
  -> exact counter compare
```

Use this for GLBench, Manhattan, and app captures. For large captures, split by frame and later by draw when debugging.

## Phase 0: Driver Skeleton

Goal: Mesa can load a `pvrgpu` Gallium driver and create/destroy a context without drawing.

Required Gallium surface:

- `pipe_screen` creation and destroy
- driver name, vendor name, device name
- minimal `get_param`, `get_shader_param`, `get_paramf`
- `pipe_context` creation and destroy
- no-op or fail-closed flush path
- resource object allocation enough for simple window/FBO setup

Initial expected dEQP scope:

```text
dEQP-GLES2.functional.info.*
dEQP-GLES2.functional.context.*
```

Exit criteria:

- Mesa chooses the PrvGPU Gallium driver intentionally.
- Unsupported caps return conservative values.
- Any unsupported operation fails clearly instead of silently producing fake output.

## Phase 1: Clear-Only Rendering

Goal: support framebuffer clear with color target.

Gallium hooks/features:

- resource create/destroy for RGBA8 render targets
- surface create/destroy
- framebuffer state bind
- color clear path
- flush enough to complete the clear
- image readback path for dEQP validation

Suggested first tests:

```text
dEQP-GLES2.functional.clear.*
dEQP-GLES2.functional.scissor.clear.*
```

Exit criteria:

- solid color clear matches dEQP expectation.
- scissor clear is either correct or reported unsupported until implemented.

## Phase 2: First Triangle

Goal: draw one non-textured triangle with simple vertex and fragment shaders.

Gallium hooks/features:

- vertex buffer binding
- vertex element state
- draw_vbo for non-indexed triangles
- viewport and scissor
- rasterizer state
- blend state with blending disabled
- shader object creation for a limited NIR subset
- constant/uniform path for simple values
- VS to FS varying linkage

Suggested tests:

```text
dEQP-GLES2.functional.draw.basic.*
dEQP-GLES2.functional.vertex_arrays.single_attribute.*
dEQP-GLES2.functional.shaders.basic.*
```

Exit criteria:

- single triangle renders correctly.
- counters for `ia_vertices`, `ia_primitives`, `vs_invocations`, `c_primitives`, and `ps_invocations` are explainable.

## Phase 3: State Completeness for GLES2 Core

Goal: build enough fixed-function state handling to make simple applications meaningful.

Feature order:

1. viewport and scissor interactions
2. indexed draw
3. culling and front-face state
4. depth test and depth write
5. stencil test
6. blend factors and blend equations
7. color mask and depth/stencil masks
8. clear interactions with depth/stencil/blend state

Suggested tests:

```text
dEQP-GLES2.functional.rasterization.*
dEQP-GLES2.functional.depth_stencil.*
dEQP-GLES2.functional.fragment_ops.*
```

Exit criteria:

- Each feature has an explicit PASS, PARTIAL, TODO, or UNSUPPORTED status.
- Unsupported state combinations fail before reaching SystemC if they cannot be lowered safely.

## Phase 4: Textures and Samplers

Goal: support the common texture path needed by GLBench and Manhattan-style workloads.

Feature order:

1. 2D RGBA8 texture upload
2. nearest sampling
3. linear sampling
4. mipmap selection
5. clamp/repeat wrap modes
6. basic texture formats beyond RGBA8
7. texture readback/copy/blit where needed

Suggested tests:

```text
dEQP-GLES2.functional.texture.*
dEQP-GLES2.functional.shaders.texture_functions.*
```

Exit criteria:

- Texture payload comes from Mesa resources, not built-in fixtures.
- Sampler and texture descriptors are generated from bound Gallium state.

## Phase 5: FBO, Sync, and Larger Workloads

Goal: support multi-pass rendering and larger captured patterns.

Feature order:

- framebuffer object attach/detach
- multiple render targets, if in scope
- blit/copy/resolve paths
- flush and fence behavior
- resource transfer correctness
- larger shader variants and uniforms

Suggested tests:

```text
dEQP-GLES2.functional.fbo.*
dEQP-GLES2.functional.sync.*
dEQP-GLES3.functional.fbo.*
```

Exit criteria:

- Multi-pass dEQP tests can run without trace-capsule shortcuts.
- Manhattan frames can be probed and reduced to specific unsupported feature groups.

## Phase 6: Advanced and Large Features

Goal: make the driver robust enough for GLES3/GLES31 advanced features and Manhattan-class captured workloads.

Feature order:

1. EGL image import/export paths that feed real resources
2. multisample render/resolve behavior
3. UBO and larger uniform layouts
4. SSBO and image access, if exposed
5. compute dispatch lowering
6. transform feedback
7. geometry/tessellation feature rejection or implementation, depending on advertised caps
8. robustness/debug/stress cases
9. large-frame scheduling, memory pressure, and artifact reduction

Suggested tests/captures:

```text
dEQP-EGL.functional.image.*
dEQP-GLES3.functional.multisample.*
dEQP-GLES31.functional.compute.*
dEQP-GLES31.functional.ssbo.*
dEQP-GLES31.functional.ubo.*
dEQP-GLES3.functional.transform_feedback.*
Manhattan captured RDC frames after Phase 1-5 are stable
```

Exit criteria:

- Unsupported advanced features are rejected by conservative caps or explicit `UNSUPPORTED`, not lowered incorrectly.
- Large captures can be bisected to feature group, frame, and eventually draw.
- Manhattan-class captures become integration tests, not the first place where basic clear/draw/texture bugs are discovered.

## dEQP Runner Contract

The live dEQP runner is delayed until the capture path is stable. When added, it should produce a report per run:

```text
out/deqp/<timestamp>/
  report.md
  run.txt
  caselist.txt
  results.qpa
  cases/
    <safe-test-name>/
      result.txt
      stdout.log
      stderr.log
      pvrgpu/
        counter_pvrgpu.txt
        systemc.log
        command.log
        image.png
```

Recommended command shape:

```bash
./scripts/run-deqp-pvrgpu.sh \
  --caselist config/deqp-gles2-phase1.txt \
  --output "$PVRGPU_WORK_ROOT/out/deqp/phase1"
```

Each test should be classified as:

```text
PASS        dEQP passed and required PrvGPU artifacts are valid
FAIL        dEQP ran and reported an image/state/counter mismatch
UNSUPPORTED driver rejected a not-yet-supported feature intentionally
ERROR       infrastructure, crash, protocol, timeout, or invalid artifact
```

## dEQP Capture Runner Contract

The capture replay runner should recursively scan a capture directory for `.rdc` files and produce a report. It should not require the frozen GLBench manifest.

Implemented files:

```text
config/deqp-capture-phases.tsv
scripts/run-deqp-capture-golden.sh
scripts/run-deqp-capture-pvrgpu-probe.sh
scripts/run-deqp-capture-rdc-report.sh
tools/deqp_capture_report.py
```

Recommended command shape:

```bash
./scripts/run-deqp-capture-rdc-report.sh \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase-max 6 \
  --filter color_clear \
  --limit 10
```

Golden replay mode:

```bash
./scripts/run-deqp-capture-rdc-report.sh \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --filter color_clear \
  --limit 1 \
  --run-golden
```

PvrGPU probe mode:

```bash
./scripts/run-deqp-capture-rdc-report.sh \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase 6 \
  --limit 1 \
  --run-pvrgpu
```

Combined counter mode:

```bash
./scripts/run-deqp-capture-rdc-report.sh \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase 1 \
  --limit 1 \
  --run-golden \
  --run-pvrgpu
```

Expected output:

```text
out/deqp-capture/<name>/
  report.md
  run.txt
  discovered-rdc.txt
  cases/
    <safe-test-name>-<sha12>/
      input.txt
      result.txt
      golden/
        counter_golden.txt
        stdout.log
        stderr.log
      pvrgpu/
        counter_pvrgpu.txt
        stdout.jsonl
        stderr.log
        mesa-poc/
      counter_diff.txt
```

Recommended classification:

```text
PASS        golden and PrvGPU counters match exactly
FAIL        both sides ran, but counters differ
UNSUPPORTED capture uses a feature not yet handled by PrvGPU lowering
ERROR       replay, extraction, protocol, crash, timeout, or invalid artifact
```

Current PrvGPU capture hook behavior:

```text
scripts/run-deqp-capture-pvrgpu-probe.sh
  -> writes pvrgpu/probe.txt
  -> writes pvrgpu/unsupported.txt
  -> exits 3
  -> report classifies the case as UNSUPPORTED
```

This is intentional. It keeps the runner honest while arbitrary dEQP RDC lowering is still being implemented. A phase is allowed to move from `UNSUPPORTED` to `PVRGPU_PASS` only when the hook writes a real `pvrgpu/counter_pvrgpu.txt` produced by PrvGPU execution.

For debug speed, this runner should support:

- `--limit N` for the first N captures in deterministic order
- `--filter TEXT` for one dEQP group or test-name fragment
- `--sample-per-group N` to build a small corpus across groups
- `--phase N` and `--phase-max N` for Phase 0 through Phase 6 bring-up
- default catalog mode with no dEQP binary
- optional `--run-golden` mode for RenderDoc + Mesa + llvmpipe replay
- optional `--run-pvrgpu` mode for the PrvGPU capture hook

The capture runner should preserve dEQP identity from path and `manifest.txt`/`.qpa` when available, because file names alone can collide across groups.

## Capability Table

Keep a checked-in table so the driver grows deliberately:

```text
Feature                          Status       First dEQP gate
pipe_screen/context              TODO         Phase 0
RGBA8 render target resource     TODO         Phase 1
clear color                      TODO         Phase 1
scissor clear                    TODO         Phase 1
draw arrays triangle             TODO         Phase 2
vertex buffer                    TODO         Phase 2
simple VS/FS shader              TODO         Phase 2
viewport/scissor draw            TODO         Phase 2
indexed draw                     TODO         Phase 3
depth test/write                 TODO         Phase 3
stencil                          TODO         Phase 3
blend                            TODO         Phase 3
2D RGBA8 texture upload          TODO         Phase 4
nearest/linear sampler           TODO         Phase 4
mipmap                           TODO         Phase 4
FBO attach/detach                TODO         Phase 5
flush/fence                      TODO         Phase 5
EGL image                        TODO         Phase 6
multisample/resolve              TODO         Phase 6
UBO/SSBO/image                   TODO         Phase 6
compute                          TODO         Phase 6
transform feedback               TODO         Phase 6
large workload bisection         TODO         Phase 6
```

## Debug Policy

When a dEQP test fails:

1. Re-run the same single test and preserve the artifact directory.
2. Classify the failure as frontend state, resource lifetime, shader lowering, command lowering, SystemC behavior, or test infrastructure.
3. Add the smallest missing Gallium behavior needed for that test.
4. Re-run the same test until it passes.
5. Only then move to the next dEQP case group.

When an RDC/Manhattan frame fails:

1. Probe the frame to list commands, states, shaders, resources, and unsupported features.
2. If unsupported, map it to the matching dEQP feature group.
3. If supported but mismatching, reduce to frame/draw/counter and debug PrvGPU or lowering.

When a dEQP capture fails:

1. Use the capture path to identify the original dEQP suite, group, and case name.
2. Replay that single `.rdc` and preserve the artifact directory.
3. If replay extraction fails, fix the capture/replay frontend first.
4. If PrvGPU lowering rejects the trace, map the unsupported feature to the capability table.
5. If counters/images mismatch, debug the PrvGPU command lowering or SystemC behavior.
6. After the capture passes, run the corresponding official dEQP case to confirm the real driver path.

## First Implementation Milestone

The first concrete driver milestone should be:

```text
Mesa loads PrvGPU Gallium driver
  -> dEQP creates GLES2 context
  -> clear color test runs
  -> PrvGPU SystemC receives a real clear command
  -> dEQP PASS artifact is written
```

Suggested files to add next:

```text
src/gallium/drivers/pvrgpu/
```

Do not start with Manhattan or live dEQP as the first driver target. Use captured dEQP RDC first, then move to live dEQP, then use Manhattan as a later integration workload after the relevant feature groups are stable.

Current capture infrastructure milestone:

```text
captured dEQP RDC corpus
  -> Phase 0-6 catalog
  -> optional Golden counter
  -> optional PrvGPU probe/lowering hook
  -> exact counter compare when both counters exist
  -> report.md + run.json artifacts
```
