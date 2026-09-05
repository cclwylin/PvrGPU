# PvrGPU

PvrGPU is an experimental PowerVR-oriented GPU model workspace. The current codebase focuses on an event-driven SystemC/C++ functional model, RenderDoc/RDC counter workflows, and a capture-first path for growing toward a real Mesa Gallium driver.

The short version:

- source and docs live in this repository;
- generated builds, Mesa checkouts, logs, `.rdc` captures, `.qpa` files, and reports stay outside the repository;
- local configuration belongs in `config/local.env`, which is intentionally ignored;
- checked-in defaults use `$HOME/Downloads/_Codex/Working/PvrGPU` unless overridden.

## Ground Rule: No Prepared Answers

**Every result must be produced by the PCO driver and the PvrGPU model doing the
work. Nothing may be keyed to the name of a test, a capture, or a workload.**

This is the project's first rule because it is the one that decides whether any
number here means anything. A GPU model that recognises `fill_tex_bilinear` and
replies with the counters `fill_tex_bilinear` is known to want has learned
nothing about texture filtering; it has learned an answer key. It will report
PASS on the case it memorised and be silently, unboundedly wrong on the next
one, and there is no way to tell the two apart from the outside.

So behaviour comes from one place only:

- the **PCO driver** lowers the real GL/Gallium state it was handed --
  the actual vertex buffers, shader binaries, formats, topology and
  fixed-function state -- into `pvrgpu.driver-command.v1`; and
- the **PvrGPU model** executes that command through VertexFetch, the USC ISS,
  ClipCull, the tiler, ISP, PBE and the memory hierarchy, and reports what those
  stages actually did.

### Forbidden

- Branching on a test, capture, or case name -- `strcmp`, prefix, or substring --
  in a way that changes counters, geometry, pixels, or whether a draw runs.
- Tables of expected counters, draw counts, vertex counts, or pixels, and
  formulas fitted to a golden report.
- Recognising a shader binary and returning a hand-written instruction list
  instead of decoding it.
- Per-case magic constants: a rounding threshold, a coefficient count, or a
  clear-colour channel chosen because one case wanted it.
- Dropping or suppressing a draw so a comparison stops disagreeing.
- Tests that assert the presence of any of the above.

### The test to apply

**Rename the capture and run it again.** If any counter, any pixel, or any
decision changes, the behaviour was a prepared answer, not an implementation.
The same question, put differently: would this still be correct if the identical
geometry arrived from an application nobody had ever seen?

### What is still allowed

Fail-closed narrowness is not cheating, and the difference matters:

- **Declining** work that is not implemented -- returning `unsupported_draw`,
  `NotSupported`, or a named refusal -- is honest. It says "the model cannot do
  this yet" and costs a PASS.
- **Supplying** the answer for work that is not implemented is the violation. It
  says "the model did this" when it did not, and buys a PASS.

Every gate must therefore state *what it checks*, not *who it is for*, and must
name the field that failed when it declines. A gate that reads
`vertex_stride == 32 && vertex_inputs == 12` describes a payload the lowering
path can handle; a gate that reads `case_name == "ideas.ideas.capture.1"` names
a customer.

### Failing honestly beats passing dishonestly

This has already been paid for, deliberately. Removing fifteen shader-recognition
predicates (`3ed6fdf`) cost the `conditionals` workload and took glmark2 from
20/20 to 18/20, because the real decoder stops on a PCK format it does not
implement. That is the correct outcome: a located bug is worth more than
seventeen counters that came from a hand-written list. Likewise `76d1e16`
removed per-filter `texel_fetches` constants, a name-keyed formula, a draw
suppressor and a name-keyed clear alpha.

**A regression that reveals missing work is progress. Do not repair it by
restoring the answer key.**

### Known debt

The tree is not clean yet, and pretending otherwise would be its own version of
the same problem. These name-keyed paths still exist and are tracked for
removal:

| Where | What it keys on | Effect |
| --- | --- | --- |
| [`pvrgpu_cmd.c`](src/gallium/drivers/pvrgpu/pvrgpu_cmd.c) `pvrgpu_case_reserves_native_pco_sequence()` | `PVRGPU_RDC_CASE_NAME` equals `refract`/`shadow`/`terrain` capture literals | changes which lowering path owns the frame |
| [`pvrgpu_resource.c`](src/gallium/drivers/pvrgpu/pvrgpu_resource.c) `pvrgpu_case_suppresses_driver_commands()` | the above, plus one exact dEQP `negative_coverage` case | suppresses driver commands entirely |
| [`pvrgpu_resource.c`](src/gallium/drivers/pvrgpu/pvrgpu_resource.c) `pvrgpu_deqp_fbo_default_framebuffer_blit_to_default_case()` | dEQP case-name prefix + substring | selects a counter path |
| [`pvrgpu_cmd.c`](src/gallium/drivers/pvrgpu/pvrgpu_cmd.c), [`pvrgpu_context.c`](src/gallium/drivers/pvrgpu/pvrgpu_context.c) | `case_name == "ideas.ideas.capture.1"` | selects vertex layouts, and gates a per-ordinal expected-draw table |
| [`functional_types.h`](src/systemc/common/functional_types.h) `FunctionalCase` | 20 GLBench/glmark2 fixture names plus 5 driver-command buckets, matched as exact workload strings | an unrecognised workload becomes `kNone` and the submitter throws; PDS, the fragment frontend and the reporter all branch on the enum |
| [`model_types.h`](model_stub/model_types.h) `kDriverPcoIdeasSequenceCommands` | one workload's exact draw count (180) | a sequence-length contract keyed to a capture |

Adding to this table is not an option. Removing a row, and taking the honest
regression that comes with it, is the work.

Detail per component: [PvrGPU.md §3.5](PvrGPU.md), the
[driver README](src/gallium/drivers/pvrgpu/README.md), and the
[dEQP runner README](script/README.md).

## Current Status

- GLBench/RDC counter infrastructure exists for fixed captured workloads.
- dEQP capture cataloging is ready through Phase 0 to Phase 6.
- `pvrgpu-deqp` now runs EGL/GLES2/GLES3/GLES31 dEQP cases directly through
  the PvrGPU Mesa Gallium driver and the SystemC bridge, without RDC replay.
- The PrvGPU dEQP capture hook is fail-closed: unsupported phases return `UNSUPPORTED` and do not write fake counters.
- A Phase 0/1/2/3/4/5/6 Mesa Gallium driver skeleton now lives under `src/gallium/drivers/pvrgpu/`.
- The driver-to-model text contract is `pvrgpu.driver-command.v1`; it includes the original RGBA8 clear path and narrowly matched draw commands.
- The driver can now create a surfaceless GLES2 context, clear a pbuffer, read back the clear pixel through CPU-backed transfer hooks, and emit a lightweight `pvrgpu.driver-counter.v1` event log.
- Phase 2 driver bring-up can now observe a minimal GLES2 VS/FS + client vertex array + `glDrawArrays(GL_TRIANGLES, 0, 3)` path and emits `event=draw_triangles` in the driver counter log. This is not pixel-correct rasterization yet.
- Phase 3 driver bring-up can now copy/bind fixed-function blend, depth/stencil/alpha, and rasterizer state, track scissor/blend-color/stencil-ref state, observe one indexed `glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, ...)` path, and emit `event=draw_indexed_triangles`. This is still counter-only, not depth/blend/stencil pixel correctness.
- Phase 4 driver bring-up observes RGBA8 upload, sampler/view state, and textured draws. The generic smoke remains counter-oriented, while the strictly matched glmark2 `effect2d` six-vertex full-screen draw now lowers through a tight RGBA8 sidecar into the SystemC depth/varying/nearest-sampling/readback path. Other texture draws remain fail-closed.
- Phase 5 driver bring-up can now observe texture-backed FBO attachment/framebuffer-state traffic, FBO clear/readback, same-format 2D copy/blit traffic, one triangle draw into the FBO, and `glFlush`/`glFinish` visibility via driver counters. This is still counter-only for draw/sync correctness; scaled blits and resolves are not implemented.
- Phase 6 driver bring-up can now retain GLES2 uniform uploads as Gallium constant-buffer state, expose first payload words in the driver counter log, and observe one uniform-driven triangle via `event=draw_uniform_triangles`. This is still counter-only, not uniform math or UBO/model correctness.
- `src/gallium/drivers/pvrgpu/meson.build` is the Mesa integration seam for
  `-Dgallium-drivers=llvmpipe,zink,pvrgpu`.
- The native single-RDC entry points are built as `llvmpipe` and `pvrgpu`.
  They keep RenderDoc/Mesa/SystemC outside the UI process.

## Important Documents

- [PvrGPU.md](PvrGPU.md): main architecture and implementation plan.
- [PrvGPU_rdc.md](PrvGPU_rdc.md): RDC counter comparison workflow.
- [script/README.md](script/README.md): running dEQP against the built PCO driver and PvrGPU model.
- [docs/COUNTER_PROTOCOL.md](docs/COUNTER_PROTOCOL.md): counter protocol contract.
- [docs/PVRGPU_DRIVER_COMMAND.md](docs/PVRGPU_DRIVER_COMMAND.md): first Gallium-driver command contract consumed by the model.
- [docs/RDC_COUNTER_UI.md](docs/RDC_COUNTER_UI.md): RDC counter UI notes.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): third-party provenance and notices.

## Local Setup

Copy the example environment file and adjust paths for your machine:

```bash
cp config/local.env.example config/local.env
```

The default working root is:

```bash
$HOME/Downloads/_Codex/Working/PvrGPU
```

Override it if desired:

```bash
export PVRGPU_WORK_ROOT=/path/to/working/PvrGPU
```

## Build and Smoke Checks

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
python3 tests/check_source_tree.py --root .
PYTHONPATH=tools PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

CMake uses native executable suffixes. The paths shown in this document are
for macOS/Linux, where the files are `build/bin/llvmpipe`,
`build/bin/pvrgpu`, and `build/bin/pvrgpu-model-stub`. Windows produces the
same targets as `llvmpipe.exe`, `pvrgpu.exe`, and
`pvrgpu-model-stub.exe`; append `.exe` to the command paths below.

The two RDC targets are compiled entry points, not monolithic Mesa/RenderDoc
links. They directly launch the pinned `renderdoc-mesa-player` with the
backend-specific Mesa prefix; `pvrgpu` then loads the built SystemC bridge (or
falls back to `pvrgpu-model-stub`). Keep those machine-specific sidecar paths
in `config/local.env` so every run records and reuses the same runtime.

## RDC Comparison UI

The current UI is the retained PySide6 thin process front end. It launches the
native backend executables; it does not load Mesa or SystemC into the UI
process:

```bash
python3 -m venv "$HOME/Downloads/_Codex/Working/PvrGPU/venv"
"$HOME/Downloads/_Codex/Working/PvrGPU/venv/bin/python" \
  -m pip install -r requirements-ui.txt
PVRGPU_BUILD_DIR="$PWD/build" \
  "$HOME/Downloads/_Codex/Working/PvrGPU/venv/bin/python" \
  tools/rdc_counter_ui.py
```

A native Qt/C++ UI is follow-up work and requires a Qt 6 C++ SDK. The batch
PASS gate already requires both the normalized 17-counter comparison and the
decoded-RGBA PNG comparison to pass, except for captures with explicit
no-color evidence. See [docs/RDC_COUNTER_UI.md](docs/RDC_COUNTER_UI.md).

## Driver Command and Counter Smoke

Build the two native replay entry points and run the registered native smoke
tests:

```bash
cmake --build build --target llvmpipe pvrgpu
ctest --test-dir build --output-on-failure
```

Each player accepts the same single-capture contract. It writes all raw and
normalized artifacts below the requested output directory:

```bash
build/bin/llvmpipe \
  /path/to/file.rdc \
  --case fill_solid --width 512 --height 512 \
  --outdir /tmp/pvrgpu-golden

build/bin/pvrgpu \
  /path/to/file.rdc \
  --case fill_solid --width 512 --height 512 \
  --outdir /tmp/pvrgpu-model
```

For worker compatibility, both players also accept the input as
`--rdc /path/to/file.rdc` instead of the positional argument.

The checked-in model can consume one clear command emitted by the Gallium skeleton:

```bash
build/bin/pvrgpu-model-stub \
  --driver-command /path/to/command.txt \
  --outdir /tmp/pvrgpu-driver-smoke
```

That command path is deliberately small:

```text
Mesa Gallium clear()
  -> src/gallium/drivers/pvrgpu writes pvrgpu.driver-command.v1
  -> model_stub validates --driver-command
  -> SystemC functional path produces counters and framebuffer PNG
```

This is the first real driver/model seam. The strictly matched glmark2
`effect2d` draw additionally crosses the seam with six positions/UVs and a
tight RGBA8 texture sidecar, then runs SystemC depth, interpolation, nearest
sampling, and readback. Phase 2/3/5/6 still have counter-oriented or narrowly
modeled portions; arbitrary draws and shaders, general texture sampling,
UBO/model layout correctness, scaled blits, resolves, real fences/sync,
pixel-correct FBO draws and general depth/blend/stencil behavior remain future
work. Live dEQP is now an executable integration gate for the supported slice.

When testing through Mesa, select the driver explicitly:

```bash
GALLIUM_DRIVER=pvrgpu \
PVRGPU_DRIVER_COMMAND_OUT=/tmp/pvrgpu-clear-command.txt \
PVRGPU_DRIVER_COUNTER_OUT=/tmp/pvrgpu-driver-counter.txt \
<mesa-test>
```

## Live dEQP Runner

`pvrgpu-deqp` is one directly linked runner process. It compiles the upstream
EGL, GLES2, GLES3, and GLES31 dEQP packages into one executable, links the
PvrGPU Mesa EGL/GLES libraries, and directly imports the SystemC bridge. The
PvrGPU driver itself is folded into Mesa's Gallium dylib. This is therefore a
single executable entry point, not a fully static single-file distribution;
the Mesa/SystemC dylibs and dEQP data directories remain runtime sidecars.

Configure it with a VK-GL-CTS checkout and a Mesa prefix that contains the
PvrGPU Gallium driver:

```bash
set -a
source config/local.env
set +a

cmake -S . -B "$PVRGPU_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPVRGPU_ENABLE_LIVE_DEQP=ON \
  -DPVRGPU_DEQP_SOURCE_DIR="$PVRGPU_DEQP_SOURCE_DIR" \
  -DPVRGPU_DEQP_MESA_PREFIX="$PVRGPU_MESA_PVRGPU_PREFIX"
cmake --build "$PVRGPU_BUILD_DIR" --target pvrgpu-deqp
```

Launch the desktop front end after the runner is built:

```bash
python3 tools/deqp_live_ui.py
```

The launcher automatically re-executes through `PVRGPU_UI_PYTHON` when the
current Python does not have PySide6. The UI provides both exact-case mode and
a 24-group batch catalog (16,608 leaf cases in the locked CTS), plus GL config,
surface, watchdog, image logging, output, and optional bridge controls. Group
mode discovers the current caselist, filters it through canonical selectors,
and launches one fresh `pvrgpu-deqp` child per exact case. It aggregates QPA,
progress, and a JSONL journal without violating the SystemC one-command-per-
process boundary. Wildcards and comma lists remain rejected in exact-case mode.
The catalog follows the 24 capture categories but uses the locked CTS names:
`instanced` replaces stale `instancing`, scissor maps to
`fragment_ops.scissor`, and shader stress maps to `long_shaders` plus
`long_running_shaders`. Because this CTS has no GLES31 robustness leaf at all,
that zero-case slot is explicitly named and mapped as draw-indirect stress
instead of silently running unrelated tests.

The driver advertises OpenGL ES 3.1 by default, so every group in the catalog
can start. `PVRGPU_DISABLE_ES3=1` restores the earlier ES2-only surface, in
which only the three EGL groups start and the GLES3/GLES31 groups show their
blocked reason again. Two caveats remain for ES3: compute is advertised through
no-op stubs (compute cases fail rather than crash), and because the SystemC
bridge defers simulation to process exit, the model's framebuffer is never
written back for `glReadPixels`, so image-comparison cases report Fail even
when the model's own PNG matches dEQP's reference pixel for pixel. Judge those
from the per-case `systemc/*.png` until readback is wired.

Run an exact case on the fly; the executable configures the surfaceless PvrGPU
runtime and artifact paths itself:

```bash
pvrgpu_deqp_out="$(mktemp -d /tmp/pvrgpu-deqp.XXXXXX)"
"$PVRGPU_BUILD_DIR/bin/pvrgpu-deqp" \
  --pvrgpu-output-dir="$pvrgpu_deqp_out" \
  --deqp-case=dEQP-GLES2.functional.prerequisite.clear_color \
  --deqp-gl-config-name=rgba8888d24s8ms0 \
  --deqp-log-images=disable
```

The run writes `results.qpa`, `run.txt`, and a safe-name directory below
`cases/` containing the driver command/counter and SystemC JSONL/PNG artifacts.
The dEQP per-case callback switches driver output paths for expanded case
lists. The current SystemC bridge deliberately defers simulation until process
exit and retains only its latest submitted command. The UI group worker handles
this by expanding first and invoking one exact case per process; do not pass a
multi-case wildcard directly when complete per-case SystemC evidence is
required.

## dEQP Capture-First Flow

Prepared dEQP `.rdc` captures are expected outside the repository, for example:

```bash
$HOME/Downloads/_Codex/Working/deqp
```

List the Phase 0-6 contract:

```bash
python3 tools/deqp_capture_report.py --list-phases
```

Catalog captures without running dEQP binaries:

```bash
python3 tools/deqp_capture_report.py \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase-max 6
```

Run one golden replay:

```bash
python3 tools/deqp_capture_report.py \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase 1 \
  --limit 1 \
  --run-golden \
  --golden-runner build/bin/llvmpipe
```

Probe the current PrvGPU hook:

```bash
python3 tools/deqp_capture_report.py \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase 6 \
  --limit 1 \
  --run-pvrgpu \
  --pvrgpu-runner build/bin/pvrgpu
```

## GitHub Notes

Do not commit local runtime artifacts:

- `config/local.env`
- `.rdc` captures
- `.qpa` files
- Mesa/GLBench source checkouts
- build directories
- report/output/log directories

License has not been selected in this repository yet. Until a license is added, treat the project as all-rights-reserved by default.
