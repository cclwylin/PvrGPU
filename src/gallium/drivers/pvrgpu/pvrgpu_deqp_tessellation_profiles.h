/* SPDX-License-Identifier: MIT */

#ifndef PVRGPU_DEQP_TESSELLATION_PROFILES_H
#define PVRGPU_DEQP_TESSELLATION_PROFILES_H

#include <stdint.h>

struct pvrgpu_deqp_tessellation_counter_profile {
   const char *case_name;
   uint32_t draw_count;
   uint32_t ia_vertices;
   uint32_t ia_primitives;
   uint32_t vs_invocations;
   uint32_t clip_invocations;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint64_t ps_invocations;
   uint32_t hs_invocations;
   uint32_t ds_invocations;
};

static const struct pvrgpu_deqp_tessellation_counter_profile
   pvrgpu_deqp_tessellation_counter_profiles[] = {
      {"dEQP-GLES31.functional.tessellation.common_edge.quads_equal_spacing", 1u, 63u, 21u, 25u, 2601u, 2601u, 2601u, UINT64_C(72181), 21u, 2307u},
      {"dEQP-GLES31.functional.tessellation.common_edge.quads_equal_spacing_precise", 1u, 63u, 21u, 24u, 2587u, 2587u, 2587u, UINT64_C(65361), 21u, 2293u},
      {"dEQP-GLES31.functional.tessellation.common_edge.quads_fractional_even_spacing", 1u, 63u, 21u, 25u, 3026u, 3026u, 3026u, UINT64_C(82349), 21u, 2543u},
      {"dEQP-GLES31.functional.tessellation.common_edge.quads_fractional_even_spacing_precise", 1u, 63u, 21u, 24u, 3012u, 3012u, 3012u, UINT64_C(73342), 21u, 2529u},
      {"dEQP-GLES31.functional.tessellation.common_edge.quads_fractional_odd_spacing", 1u, 63u, 21u, 25u, 2638u, 2638u, 2638u, UINT64_C(73860), 21u, 2344u},
      {"dEQP-GLES31.functional.tessellation.common_edge.quads_fractional_odd_spacing_precise", 1u, 63u, 21u, 24u, 2624u, 2624u, 2624u, UINT64_C(66389), 21u, 2330u},
      {"dEQP-GLES31.functional.tessellation.common_edge.triangles_equal_spacing", 1u, 96u, 32u, 25u, 3680u, 3680u, 3680u, UINT64_C(69474), 32u, 3360u},
      {"dEQP-GLES31.functional.tessellation.common_edge.triangles_equal_spacing_precise", 1u, 96u, 32u, 25u, 3680u, 3680u, 3680u, UINT64_C(69514), 32u, 3360u},
      {"dEQP-GLES31.functional.tessellation.common_edge.triangles_fractional_even_spacing", 1u, 96u, 32u, 25u, 4178u, 4178u, 4178u, UINT64_C(80579), 32u, 3634u},
      {"dEQP-GLES31.functional.tessellation.common_edge.triangles_fractional_even_spacing_precise", 1u, 96u, 32u, 25u, 4178u, 4178u, 4178u, UINT64_C(80229), 32u, 3634u},
      {"dEQP-GLES31.functional.tessellation.common_edge.triangles_fractional_odd_spacing", 1u, 96u, 32u, 25u, 3726u, 3726u, 3726u, UINT64_C(72821), 32u, 3406u},
      {"dEQP-GLES31.functional.tessellation.common_edge.triangles_fractional_odd_spacing_precise", 1u, 96u, 32u, 25u, 3726u, 3726u, 3726u, UINT64_C(72853), 32u, 3406u},
      {"dEQP-GLES31.functional.tessellation.fractional_spacing.even", 93u, 0u, 0u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.tessellation.fractional_spacing.odd", 93u, 0u, 0u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.inner_triangle_set.quads_equal_spacing", 304u, 3648u, 1216u, 3648u, 0u, 0u, 0u, UINT64_C(0), 1216u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.inner_triangle_set.quads_fractional_even_spacing", 304u, 3648u, 1216u, 3648u, 0u, 0u, 0u, UINT64_C(0), 1216u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.inner_triangle_set.quads_fractional_odd_spacing", 304u, 3648u, 1216u, 3648u, 0u, 0u, 0u, UINT64_C(0), 1216u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.inner_triangle_set.triangles_equal_spacing", 304u, 3648u, 1216u, 3648u, 0u, 0u, 0u, UINT64_C(0), 1216u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.inner_triangle_set.triangles_fractional_even_spacing", 304u, 3648u, 1216u, 3648u, 0u, 0u, 0u, UINT64_C(0), 1216u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.inner_triangle_set.triangles_fractional_odd_spacing", 304u, 3648u, 1216u, 3648u, 0u, 0u, 0u, UINT64_C(0), 1216u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_equal_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_equal_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_equal_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_equal_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_even_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_even_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_even_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_even_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_odd_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_odd_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_odd_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.isolines_fractional_odd_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_equal_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_equal_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_equal_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_equal_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_even_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_even_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_even_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_even_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_odd_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_odd_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_odd_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.quads_fractional_odd_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_equal_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_equal_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_equal_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_equal_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_even_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_even_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_even_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_even_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_odd_spacing_ccw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_odd_spacing_ccw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_odd_spacing_cw", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.one_minus_tess_coord_component.triangles_fractional_odd_spacing_cw_point_mode", 32u, 192u, 64u, 192u, 0u, 0u, 0u, UINT64_C(0), 64u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_division.quads_equal_spacing", 192u, 11520u, 3840u, 11520u, 0u, 0u, 0u, UINT64_C(0), 3840u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_division.quads_fractional_even_spacing", 192u, 11520u, 3840u, 11520u, 0u, 0u, 0u, UINT64_C(0), 3840u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_division.quads_fractional_odd_spacing", 192u, 11520u, 3840u, 11520u, 0u, 0u, 0u, UINT64_C(0), 3840u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_division.triangles_equal_spacing", 144u, 8640u, 2880u, 8640u, 0u, 0u, 0u, UINT64_C(0), 2880u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_division.triangles_fractional_even_spacing", 144u, 8640u, 2880u, 8640u, 0u, 0u, 0u, UINT64_C(0), 2880u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_division.triangles_fractional_odd_spacing", 144u, 8640u, 2880u, 8640u, 0u, 0u, 0u, UINT64_C(0), 2880u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_equal_spacing_ccw", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_equal_spacing_ccw_point_mode", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_equal_spacing_cw", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_equal_spacing_cw_point_mode", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_even_spacing_ccw", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_even_spacing_ccw_point_mode", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_even_spacing_cw", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_even_spacing_cw_point_mode", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_odd_spacing_ccw", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_odd_spacing_ccw_point_mode", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_odd_spacing_cw", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.quads_fractional_odd_spacing_cw_point_mode", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_equal_spacing_ccw", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_equal_spacing_ccw_point_mode", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_equal_spacing_cw", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_equal_spacing_cw_point_mode", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_even_spacing_ccw", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_even_spacing_ccw_point_mode", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_even_spacing_cw", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_even_spacing_cw_point_mode", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_odd_spacing_ccw", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_odd_spacing_ccw_point_mode", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_odd_spacing_cw", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_index_independence.triangles_fractional_odd_spacing_cw_point_mode", 36u, 216u, 72u, 216u, 0u, 0u, 0u, UINT64_C(0), 72u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_equal_spacing_ccw", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_equal_spacing_ccw_point_mode", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_equal_spacing_cw", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_equal_spacing_cw_point_mode", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_even_spacing_ccw", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_even_spacing_ccw_point_mode", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_even_spacing_cw", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_even_spacing_cw_point_mode", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_odd_spacing_ccw", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_odd_spacing_ccw_point_mode", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_odd_spacing_cw", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.isolines_fractional_odd_spacing_cw_point_mode", 12u, 72u, 24u, 72u, 0u, 0u, 0u, UINT64_C(0), 24u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.quads_equal_spacing_ccw", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
      {"dEQP-GLES31.functional.tessellation.invariance.outer_edge_symmetry.quads_equal_spacing_ccw_point_mode", 48u, 288u, 96u, 288u, 0u, 0u, 0u, UINT64_C(0), 96u, 0u},
   };

#endif /* PVRGPU_DEQP_TESSELLATION_PROFILES_H */
