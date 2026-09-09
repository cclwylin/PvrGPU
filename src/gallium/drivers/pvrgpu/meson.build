# SPDX-License-Identifier: MIT

files_pvrgpu = files(
  'pvrgpu_clear.c',
  'pvrgpu_cmd.c',
  'pvrgpu_context.c',
  'pvrgpu_counter.c',
  'pvrgpu_pco.c',
  'pvrgpu_resource.c',
  'pvrgpu_screen.c',
  'pvrgpu_state.c',
)

libpvrgpu = static_library(
  'pvrgpu',
  files_pvrgpu,
  include_directories : [
    inc_gallium_aux,
    inc_gallium,
    inc_imagination,
    inc_include,
    inc_src,
  ],
  c_args : [c_msvc_compat_args],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil],
)

driver_pvrgpu = declare_dependency(
  compile_args : '-DGALLIUM_PVRGPU',
  link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
)

if with_tests
  test(
    'pvrgpu_pco_packed_uniforms',
    executable(
      'pvrgpu_pco_packed_uniforms_test',
      files('tests/pvrgpu_pco_packed_uniforms_test.c'),
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_pco_many_textures',
    executable(
      'pvrgpu_pco_many_textures_test',
      files('tests/pvrgpu_pco_many_textures_test.c'),
      c_args : ['-DPVRGPU_MANY_TEXTURES_DIRECT_HELPER'],
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_pco_shadow_gather',
    executable(
      'pvrgpu_pco_shadow_gather_test',
      files('tests/pvrgpu_pco_shadow_gather_test.c'),
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_pco_front_face',
    executable(
      'pvrgpu_pco_front_face_test',
      files('tests/pvrgpu_pco_front_face_test.c'),
      c_args : ['-DPVRGPU_FRONT_FACE_DIRECT_HELPER'],
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_pco_wide_ubo',
    executable(
      'pvrgpu_pco_wide_ubo_test',
      files('tests/pvrgpu_pco_wide_ubo_test.c'),
      c_args : ['-DPVRGPU_WIDE_UBO_DIRECT_HELPER'],
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_pco_shared_budget',
    executable(
      'pvrgpu_pco_shared_budget_test',
      files('tests/pvrgpu_pco_shared_budget_test.c'),
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_stream_output_query',
    executable(
      'pvrgpu_stream_output_query_test',
      files('tests/pvrgpu_stream_output_query_test.c'),
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common, libgallium],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['pvrgpu'],
  )
  test(
    'pvrgpu_pco_geometry',
    executable(
      'pvrgpu_pco_geometry_test',
      files('tests/pvrgpu_pco_geometry_test.c'),
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_pco_tessellation',
    executable(
      'pvrgpu_pco_tessellation_test',
      files('tests/pvrgpu_pco_tessellation_test.c'),
      include_directories : [inc_gallium_aux, inc_gallium, inc_imagination, inc_include, inc_src],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
  test(
    'pvrgpu_pco_lowering',
    executable(
      'pvrgpu_pco_lowering_test',
      files('tests/pvrgpu_pco_lowering_test.c'),
      include_directories : [
        inc_gallium_aux,
        inc_gallium,
        inc_imagination,
        inc_include,
        inc_src,
      ],
      link_with : [libpvrgpu, libpowervr_compiler, libpowervr_common],
      dependencies : [idep_nir, idep_mesautil],
    ),
    suite : ['compiler', 'pvrgpu'],
  )
endif
