# Making `glReadPixels` see what the model drew

## The problem, stated exactly

dEQP draws, reads back, checks, and draws again.  Today the readback is always
black, so every image-comparison case fails.  `results.qpa` says it plainly:

```
Diamond-exit rule: 301 fragments.
Result image:        0 fragments.
```

and for triangles, across all three iterations:

```
539 missing pixels.   0 incorrectly filled pixels.
6045 missing pixels.  0 incorrectly filled pixels.
4815 missing pixels.  0 incorrectly filled pixels.
```

Never a wrong pixel, only absent ones -- and the PNG the model writes at exit
shows correct geometry.  The model rasterises fine; nothing carries the result
back to the driver.

## Why it is black

`pvrgpu_transfer_map()` returns `pvrgpu->data`, the driver's own CPU backing
store.  Clears write there, draws do not: a draw goes to the model, and the
model's output never returns.

It never returns because the bridge defers the whole simulation to `atexit`.
That deferral is not laziness -- SystemC elaborates once per process:

```
--- run 1 ---   ok
--- run 2 ---   Error: (E529) insert module failed: elaboration done
```

`RunConfiguredModel` builds its ~60 modules and FIFOs as locals and calls
`sc_start()` once.  A second call re-elaborates, which SystemC refuses.  One
process, one simulation, and it has to be last -- hence `atexit`, hence a black
readback.

## What actually works

`sc_start()` may be called as often as you like, as long as you elaborate once
and do not `sc_stop()`.  Control returns to the caller each time and the
simulation resumes with its state intact:

```
readback 1: sim=10 ns jobs=1 pixels=100
readback 2: sim=20 ns jobs=2 pixels=300
readback 3: sim=30 ns jobs=3 pixels=600
```

The idiom, with the ordering trap that cost the first job:

1. after elaboration, `sc_start(SC_ZERO_TIME)` once, so every process reaches
   its first `wait` before any work is queued;
2. per flush: push the command, `notify(SC_ZERO_TIME)`, then
   `while (!done && sc_pending_activity()) sc_start(sc_time_to_pending_activity());`
3. `sc_stop()` only at teardown -- after it, no `sc_start()` will run again.

This is also the more faithful arrangement: the model stays alive between draws
rather than being rebuilt, which is what hardware does.

## The change

- `model_stub/pvrgpu_model_stub.cpp`, `RunConfiguredModel`: the module and FIFO
  set must outlive one call.  The wiring itself (currently around lines
  700-723) does not change, only where the objects live and that elaboration
  happens once.
- `model_stub/submitter.cpp`, `Submitter::Run()`: today it walks
  `options_.driver_commands` and returns, ending the thread.  It needs an outer
  loop -- wait for queued work, drain it, signal completion -- so the pipeline
  is idle rather than finished between flushes.
- The reporter reports per flush instead of once at the end.  Note this means a
  case that reads back three times emits three counter records; whatever
  consumes `systemc.jsonl` has to accept that.
- `model_stub/pvrgpu_systemc_bridge.cpp`: a flush entry point that runs the
  accumulated commands and hands back RGBA8 pixels, replacing
  `FlushPendingSubmitAtExit` as the normal path.
- `src/gallium/drivers/pvrgpu/pvrgpu_resource.c`: in `pvrgpu_transfer_map()`,
  when a colour attachment with pending draws is mapped for read, flush and
  copy the pixels into `pvrgpu->data`.

## The risk to watch

Every glmark2 and GLBench case that passes today goes through this same loop,
and they pass with the single deferred run.  Introducing mid-frame flushes
changes when the model sees work.  Verify both suites (20 + 20) before
committing, not after.
