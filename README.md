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
- A full Mesa Gallium driver under `src/gallium/drivers/pvrgpu/` is planned but not implemented yet.

## Important Documents

- [PvrGPU.md](PvrGPU.md): main architecture and implementation plan.
- [PrvGPU_rdc.md](PrvGPU_rdc.md): RDC counter comparison workflow.
- [PrvGPU_deqp_gallium_driver.md](PrvGPU_deqp_gallium_driver.md): capture-first dEQP/Gallium bring-up plan through Phase 6.
- [docs/COUNTER_PROTOCOL.md](docs/COUNTER_PROTOCOL.md): counter protocol contract.
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
./scripts/build.sh
./scripts/check-source-tree.sh
PYTHONPATH=tools PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

## dEQP Capture-First Flow

Prepared dEQP `.rdc` captures are expected outside the repository, for example:

```bash
$HOME/Downloads/_Codex/Working/deqp
```

List the Phase 0-6 contract:

```bash
./scripts/run-deqp-capture-rdc-report.sh --list-phases
```

Catalog captures without running dEQP binaries:

```bash
./scripts/run-deqp-capture-rdc-report.sh \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase-max 6
```

Run one golden replay:

```bash
./scripts/run-deqp-capture-rdc-report.sh \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase 1 \
  --limit 1 \
  --run-golden
```

Probe the current PrvGPU hook:

```bash
./scripts/run-deqp-capture-rdc-report.sh \
  --rdc-dir "$HOME/Downloads/_Codex/Working/deqp" \
  --phase 6 \
  --limit 1 \
  --run-pvrgpu
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
