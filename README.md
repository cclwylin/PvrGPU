# PvrGPU

PvrGPU is an experimental PowerVR-oriented GPU model workspace. The current codebase focuses on an event-driven SystemC/C++ functional model, RenderDoc/RDC counter workflows, and a capture-first path for growing toward a real Mesa Gallium driver.

The short version:

- source and docs live in this repository;
- generated builds, Mesa checkouts, logs, `.rdc` captures, `.qpa` files, and reports stay outside the repository;
- local configuration belongs in `config/local.env`, which is intentionally ignored;
- checked-in defaults use `$HOME/Downloads/_Codex/Working/PvrGPU` unless overridden.

## Current Status

- GLBench/RDC counter infrastructure exists for fixed captured workloads.
- dEQP capture cataloging is ready through Phase 0 to Phase 6.
- dEQP binary execution is intentionally not part of the current flow.
- The PrvGPU dEQP capture hook is fail-closed: unsupported phases return `UNSUPPORTED` and do not write fake counters.
- A Phase 0/1/2/3/4/5/6 Mesa Gallium driver skeleton now lives under `src/gallium/drivers/pvrgpu/`.
- The first driver-to-model contract is `pvrgpu.driver-command.v1` for RGBA8 color clear.
- The driver can now create a surfaceless GLES2 context, clear a pbuffer, read back the clear pixel through CPU-backed transfer hooks, and emit a lightweight `pvrgpu.driver-counter.v1` event log.
- Phase 2 driver bring-up can now observe a minimal GLES2 VS/FS + client vertex array + `glDrawArrays(GL_TRIANGLES, 0, 3)` path and emits `event=draw_triangles` in the driver counter log. This is not pixel-correct rasterization yet.
- Phase 3 driver bring-up can now copy/bind fixed-function blend, depth/stencil/alpha, and rasterizer state, track scissor/blend-color/stencil-ref state, observe one indexed `glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, ...)` path, and emit `event=draw_indexed_triangles`. This is still counter-only, not depth/blend/stencil pixel correctness.
- Phase 4 driver bring-up can now observe `glTexImage2D` RGBA8 upload, fragment sampler view binding, nearest/clamp sampler state, and one textured `glDrawArrays(GL_TRIANGLES, 0, 3)` path via `event=draw_textured_triangles`. This is still counter-only, not texture sampling pixel correctness.
- Phase 5 driver bring-up can now observe texture-backed FBO attachment/framebuffer-state traffic, FBO clear/readback, same-format 2D copy/blit traffic, one triangle draw into the FBO, and `glFlush`/`glFinish` visibility via driver counters. This is still counter-only for draw/sync correctness; scaled blits and resolves are not implemented.
- Phase 6 driver bring-up can now retain GLES2 uniform uploads as Gallium constant-buffer state, expose first payload words in the driver counter log, and observe one uniform-driven triangle via `event=draw_uniform_triangles`. This is still counter-only, not uniform math or UBO/model correctness.
- `src/gallium/drivers/pvrgpu/meson.build` is the Mesa integration seam for
  `-Dgallium-drivers=llvmpipe,zink,pvrgpu`.
- The native single-RDC entry points are built as `llvmpipe` and `pvrgpu`.
  They keep RenderDoc/Mesa/SystemC outside the UI process.

## Important Documents

- [PvrGPU.md](PvrGPU.md): main architecture and implementation plan.
- [PrvGPU_rdc.md](PrvGPU_rdc.md): RDC counter comparison workflow.
- [PrvGPU_deqp_gallium_driver.md](PrvGPU_deqp_gallium_driver.md): capture-first dEQP/Gallium bring-up plan through Phase 6.
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

This is the first real driver/model seam. Phase 2/3/4/5/6 currently observe one
simple triangle draw, one fixed-function indexed draw path, one textured
triangle frontend path, one texture-backed FBO path, one same-format 2D
copy/blit path, and one uniform/constant-buffer path through driver counters
only; arbitrary draws, shader lowering, model texture sampling, UBO/model
layout correctness, scaled blits, resolves, real fences/sync, pixel-correct FBO
draw, depth/blend/stencil behavior, and live dEQP are still future phases.

When testing through Mesa, select the driver explicitly:

```bash
GALLIUM_DRIVER=pvrgpu \
PVRGPU_DRIVER_COMMAND_OUT=/tmp/pvrgpu-clear-command.txt \
PVRGPU_DRIVER_COUNTER_OUT=/tmp/pvrgpu-driver-counter.txt \
<mesa-test>
```

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
