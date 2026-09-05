# Verifying the readback change

`docs/PVRGPU_READBACK_PLAN.md` ends with a warning: every glmark2 and GLBench
case that passes today goes through the same loop, and they pass with the
single deferred run.  Mid-frame flushes change when the model sees work, so
both suites have to be checked *before* committing, not after.

This is that checklist.  What has already been run is listed first, so the only
thing left is the part that needs the macOS toolchain.

## Already run

| Check | Result |
| --- | --- |
| `cmake --build build` (whole project) | clean |
| `ctest --test-dir build` | 48/48 |
| `python3 -m unittest discover -s tests` | 78/78 |
| `python3 tests/check_source_tree.py --root .` | pass |
| `python3 tests/check_systemc_module_layout.py --root .` | pass (36 modules) |
| `python3 tests/check_model_pipeline.py build/bin/pvrgpu-model-stub` | pass |

Those were run on Linux against SystemC 2.3.4 rather than macOS, so re-run at
least the first two locally.  They will not have exercised the Mesa Gallium
driver at all: `pvrgpu_resource.c`, `pvrgpu_context.c` and `pvrgpu_cmd.c` are
built by Mesa's meson, not by this project's CMake, so **the driver half of
this change has never been compiled**.  That is the first step below.

The new `systemc-readback-flush-unit` test is the one that covers the change
itself: it flushes three times on one elaborated model -- two clears of
different colours and a real triangle draw -- and checks that each flush
returns its own pixels.

## 1. Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```

Then rebuild Mesa with the pvrgpu Gallium driver, the way you normally do.
This is the first compile of the driver changes; expect to fix something.

## 2. The suites the plan asks for

Run both before committing.  Each case that passes today must still pass, and
its counters must still match its golden report.

```bash
./script/run_regression.sh --suite glmark2
./script/run_regression.sh --suite GLBench
```

What to look at when one fails:

- **`PvrGPU protocol requires one complete hello/counter/done set per flush`** --
  the run left an incomplete set.  A missing `counter` or `done` means the model
  did not finish, not that the record count changed; `ValidatePvrgpuCompletion`
  now accepts N complete sets instead of exactly one.
- **A counter mismatch against the golden report** -- the workload was read back
  more than once, so its counters split across flushes and
  `ParsePvrgpuCounters` took the last set rather than the whole frame.  Check
  `model.stdout.jsonl` for the number of `"type":"counter"` lines.  One line
  means the flush moved from `atexit` to the readback without splitting
  anything, which is the intended outcome for a replay.
- **A PNG comparison failure** -- check whether the case now renders at readback
  and then renders *again* at exit.  The `readback_sequence_flush` and
  `systemc_readback_flush` events in `driver-counter.txt` say how many times the
  model ran.

An RDC replay is deliberately left on the old once-per-process footing: the
trace declares its own draw-action count, RenderDoc replays the same frame
afterwards, and reopening the frame gates there would let a repeat describe
itself as a second frame.  That split is in
`pvrgpu_context_end_frame_at_readback()` and turns on
`PVRGPU_RDC_TRACE_DRAW_ACTIONS`, which only the RDC path sets.

## 3. The suite this was for

```bash
./script/run_regression.sh --suite dEQP --limit 50
```

or, for one case with the artifacts to look at:

```bash
./script/run_deqp_dynamic.sh -c dEQP-GLES2.functional.prerequisite.clear_color
```

The thing to check in `results.qpa` is the shape of the failure, not just the
count.  Before, image comparisons read

```
539 missing pixels.   0 incorrectly filled pixels.
```

Missing pixels and never a wrong one is the signature of a readback that never
saw the draw.  If cases still fail but now report *incorrectly filled* pixels,
the readback is working and what is left is a rasterization or state
difference -- a different problem, and a better one.

## 4. Two things this change does not do

Worth knowing before reading a dEQP failure as a regression.

- **A second frame starts from a fresh clear.**  A readback ends the frame, and
  the draws after it start a new sequence whose first draw carries
  `PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR`.  A case that draws, reads back, then
  draws *more into the same surface without clearing* will lose the first
  half.  Carrying an attachment across sequences needs the model to keep the
  previous colour address, which is not in this change.
- ~~**Only RGBA8 and RGBX8 colour attachments are copied back.**~~  Closed --
  and it was the whole of the remaining failure.  dEQP's pbuffer comes up as
  `PIPE_FORMAT_B8G8R8A8_UNORM`, so every readback was declined at this gate and
  `driver-counter.txt` carried zero `framebuffer_readback` events.  The four
  8-bit orderings are the same bytes rearranged, so the readback now reorders
  them on the way in, matching `pvrgpu_store_clear_color_pixel()` channel for
  channel including an X8 format's alpha lane reading back as one.  Anything
  wider or packed is still declined.

## 5. Protocol note for anything else reading `systemc.jsonl`

The reporter reports per flush, so a case read back three times leaves three
complete `hello`/`counter`/`done` sets in one file, appended rather than
truncated.  `src/rdc_runner/main.cpp` has been updated.  Anything else that
assumes a single record set -- ad-hoc scripts, notebooks -- has not.

## 6. What the run found

Both remaining steps have been run on macOS against the meson-built driver.

| Check | Result |
| --- | --- |
| `es3_bringup.sh build` (Mesa PCO driver) | clean |
| `cmake --build build` | clean |
| `ctest` | 50/50 |
| `run_regression.sh --suite glmark2` | 20/20 |
| `run_regression.sh --suite GLBench` | 20/20 |
| `rdc_counter_report.py --rdc-dir .../4.glmark2_800x600` | 5/10, unchanged |

The counter report's five failures are the ones already on record and their
deltas are identical: ideas differs only in `ps_invocations`, by 6; terrain and
refract differ in `c_primitives`/`ps_invocations`/`drawlists`/`texel_fetches`;
shadow differs in `ps_invocations`.  `ia_vertices` and `vs_invocations` appear
in no diff, so the flush split does not disturb vertex accounting.

`dEQP-GLES3.functional.rasterization.primitives` went from 0 passing to 3, with
4 legitimately NotSupported (wide lines and points are not advertised).  Two
defects had to be fixed on top of the change this document verifies:

- **The format gate above.**  Without it nothing was read back at all.
- **A host write left stale lines in the SLC.**  The model now outlives a
  flush, so `GpuMemorySystem::HostWrite()` -- which establishes DRAM contents
  directly -- was leaving the previous flush's vertex data resident.  Flushes 2
  and 3 of a case hit 2076 times with 0 misses and re-rendered flush 1's
  geometry down to an identical `ps_invocations`.  Host writes now clean and
  drop the lines they cover, which is what the §3 signature was really saying:
  the readback worked from the first flush, and the later ones read a cache.

## 7. What is left

The three line topologies still fail, and now for a reason of their own.  The
readback is correct -- dEQP reports "No invalid deviations found", so the
fragments are in the right places -- but a width-1 line comes out with 323
fragments against the diamond-exit rule's 301, and some x-major spans are two
pixels wide instead of one.

`BuildLineQuadCorners()` widens a segment into a quad offset perpendicular to
it and hands that to triangle setup.  A rotated quad straddles two rows at many
steps along a diagonal, which is the rectangle rule rather than the diamond-exit
rule GLES wants for aliased width-1 lines.  Getting these to pass means a line
setup path that emits one fragment per major-axis step, not a wider quad.
