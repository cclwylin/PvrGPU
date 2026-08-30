# SPDX-License-Identifier: MIT

files_pvrgpu = files(
  'pvrgpu_clear.c',
  'pvrgpu_cmd.c',
  'pvrgpu_context.c',
  'pvrgpu_counter.c',
  'pvrgpu_resource.c',
  'pvrgpu_screen.c',
  'pvrgpu_state.c',
)

libpvrgpu = static_library(
  'pvrgpu',
  files_pvrgpu,
  include_directories : [inc_gallium_aux, inc_gallium, inc_include, inc_src],
  c_args : [c_msvc_compat_args],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil],
)

driver_pvrgpu = declare_dependency(
  compile_args : '-DGALLIUM_PVRGPU',
  link_with : libpvrgpu,
)
