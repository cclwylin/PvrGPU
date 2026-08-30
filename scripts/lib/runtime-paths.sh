#!/usr/bin/env bash

# Generated files stay outside iCloud Drive. Callers may override the root or
# any individual path before sourcing this file (or in config/local.env).
pvrgpu_default_work_root="${PVRGPU_DEFAULT_WORK_ROOT:-${HOME}/Downloads/_Codex/Working/PvrGPU}"
PVRGPU_WORK_ROOT="${PVRGPU_WORK_ROOT:-${pvrgpu_default_work_root}}"
PVRGPU_BUILD_DIR="${PVRGPU_BUILD_DIR:-${PVRGPU_WORK_ROOT}/build}"
PVRGPU_UI_VENV="${PVRGPU_UI_VENV:-${PVRGPU_WORK_ROOT}/venv}"
PVRGPU_UI_PYTHON="${PVRGPU_UI_PYTHON:-${PVRGPU_UI_VENV}/bin/python}"
PVRGPU_OUTPUT_ROOT="${PVRGPU_OUTPUT_ROOT:-${PVRGPU_WORK_ROOT}/out/runs}"
PVRGPU_TMP_ROOT="${PVRGPU_TMP_ROOT:-${PVRGPU_WORK_ROOT}/tmp}"
PVRGPU_GLBENCH_SOURCE_DIR="${PVRGPU_GLBENCH_SOURCE_DIR:-${PVRGPU_WORK_ROOT}/third_party/glbench-src}"
PVRGPU_GLBENCH_RUNNER="${PVRGPU_GLBENCH_RUNNER:-${PVRGPU_BUILD_DIR}/bin/glbench-mesa}"
PVRGPU_MODEL_STUB="${PVRGPU_MODEL_STUB:-${PVRGPU_BUILD_DIR}/bin/pvrgpu-model-stub}"
PVRGPU_SMOKE_OUTPUT="${PVRGPU_SMOKE_OUTPUT:-${PVRGPU_WORK_ROOT}/out/smoke/llvmpipe-fill-solid}"
PVRGPU_LOG_DIR="${PVRGPU_LOG_DIR:-${PVRGPU_WORK_ROOT}/logs}"
