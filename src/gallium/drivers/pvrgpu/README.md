# PvrGPU Gallium Bring-up Driver

This directory is the source-of-truth for the future Mesa Gallium driver.

Current status:

- Phase 0/1/2/3/4/5/6 bring-up driver.
- `pipe_screen` / `pipe_context` names and file boundaries are established.
- Mesa software-loader selection is explicit through `GALLIUM_DRIVER=pvrgpu`.
- GLES2 context creation works through the Mesa software DRI loader.
- full-frame RGBA8/BGRA8 color clear lowers to `pvrgpu.driver-command.v1`.
- CPU-backed resource storage and `buffer_map` / `texture_map` / subdata hooks are present for bring-up readback/upload.
- fence callbacks are no-op-complete because this driver currently flushes synchronously and does not create real fences.
- `PVRGPU_DRIVER_COUNTER_OUT=/path/to/counter.txt` emits lightweight driver-side event records for clear/resource/transfer/shader/vertex/state/draw visibility.
- simple GLES2 vertex/fragment shader creation, vertex element state, client vertex buffers, viewport/scissor state, and one non-indexed `GL_TRIANGLES` draw are observable through `draw_triangles` counters.
- Phase 3 fixed-function state objects for blend, depth/stencil/alpha, and rasterizer are copied, bound, and reported through counters.
- one indexed `GL_TRIANGLES` draw with user indices is observable through `draw_indexed_triangles` counters.
- Phase 4 sampler state, sampler views, 2D RGBA8 texture upload, and one textured triangle are observable through `draw_textured_triangles` counters.
- Phase 5 texture-backed FBO color attachment traffic, FBO clear/readback, same-format 2D copy/blit traffic, and flush/finish visibility are observable through driver counters.
- Phase 6 GLES2 uniform uploads are retained as Gallium constant-buffer state and one uniform-driven triangle is observable through `draw_uniform_triangles` counters.
- Mesa's upload manager is initialized for state-tracker internal uploads, and resource release hooks are wired so teardown is clean.
- real draw lowering/rasterization is not implemented yet; unsupported draw shapes still record `unsupported_draw`.
- real texture sampling, shader lowering, scaled/format-converting blits, resolves, FBO draw pixels, UBO layout correctness, EGL image import/export, depth/stencil/blend pixel behavior, compute, and real synchronization paths are still intentionally fail-closed, tracked-only, or no-op.
- these files are not wired into the repository CMake build; use `scripts/install-pvrgpu-mesa-driver.sh` to copy and patch them into a Mesa source tree.

The first real target is:

```text
Mesa Gallium clear()
  -> GALLIUM_DRIVER=pvrgpu
  -> PVRGPU_DRIVER_COMMAND_OUT=/path/to/command.txt
  -> PVRGPU_DRIVER_COUNTER_OUT=/path/to/counter.txt
  -> pvrgpu.driver-command.v1 clear_color
  -> pvrgpu-model-stub --driver-command
  -> counter JSONL + DRAM-readback PNG
```

Do not extend this driver by special-casing captured test names. The driver boundary must be normal Gallium state.

The command format consumed by the model is documented in
`docs/PVRGPU_DRIVER_COMMAND.md`.

Direct Mesa smoke:

```bash
./scripts/check-pvrgpu-mesa-driver-build.sh --platforms macos --full-dri --install
./scripts/run-pvrgpu-mesa-driver-smoke.sh --size 16x16
./scripts/run-pvrgpu-mesa-driver-triangle-smoke.sh --size 16x16
./scripts/run-pvrgpu-mesa-driver-phase3-state-smoke.sh --size 16x16
./scripts/run-pvrgpu-mesa-driver-phase4-texture-smoke.sh --size 16x16
./scripts/run-pvrgpu-mesa-driver-phase5-fbo-smoke.sh --size 16x16 --fbo-size 8x8
./scripts/run-pvrgpu-mesa-driver-phase6-uniform-smoke.sh --size 16x16
```

The clear smoke creates a surfaceless GLES2 pbuffer, clears it, verifies
`glReadPixels()` returns `32,64,128,255`, writes `driver-command.txt`, and
writes `driver-counter.txt`.

The triangle smoke creates the same kind of pbuffer, compiles a minimal GLES2
VS/FS pair, binds one client vertex array, issues `glDrawArrays(GL_TRIANGLES,
0, 3)`, and verifies `driver-counter.txt` contains `event=draw_triangles`.
It does not compare pixels because this phase only proves that Mesa reaches the
driver draw hook with deterministic state.

The Phase 3 state smoke is also counter-only. It enables scissor, cull/front
face, depth, stencil, blend, blend color, and color mask state, then issues
`glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, ...)`. The expected proof is
that the driver records the fixed-function state create/bind calls and
`event=draw_indexed_triangles`. It does not prove depth, stencil, blend, or
triangle pixels are correct yet.

The Phase 4 texture smoke uploads a 2x2 RGBA8 texture with `glTexImage2D`,
configures nearest/clamp sampling, binds a `sampler2D` fragment shader, and
issues `glDrawArrays(GL_TRIANGLES, 0, 3)`. The expected proof is
`texture_subdata`, sampler state/view events, and `event=draw_textured_triangles`.
It does not prove texture sampling pixels are correct yet.

The Phase 5 FBO smoke creates an 8x8 texture-backed FBO, verifies
`glCheckFramebufferStatus(GL_FRAMEBUFFER)` returns complete, clears the FBO,
checks `glReadPixels()` returns `64,128,191,255`, copies that FBO into another
RGBA8 texture through `glCopyTexSubImage2D`, attaches the copy texture and
checks the copied pixel, draws one triangle into that FBO, and calls
`glFlush()`/`glFinish()`. The expected proof is `set_framebuffer_state`,
FBO-sized `clear_color`, `resource_copy_region` or `blit`, `draw_triangles`,
and `flush` events. It does not prove scaled blits, resolves, FBO draw pixels,
or real async synchronization yet.

The Phase 6 uniform smoke binds a GLES2 fragment shader with a `vec4[8]`
uniform array, uploads it with `glUniform4fv`, and draws one triangle. The
expected proof is `set_constant_buffer stage=fragment ... has_buffer=1`,
payload `first_words`, and `event=draw_uniform_triangles`. It does not prove
uniform math or full UBO layout correctness in the model yet.
