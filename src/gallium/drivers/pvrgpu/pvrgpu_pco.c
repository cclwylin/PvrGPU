/* SPDX-License-Identifier: MIT */

#include "pvrgpu_pco.h"

#include "common/pvr_device_info.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "nir/nir_builder_opcodes.h"
#include "pco/pco.h"
#include "pco/pco_data.h"
#include "util/ralloc.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PVRGPU_CONDITIONALS_VS_UNIFORM_DWORDS 16U
#define PVRGPU_CONDITIONALS_FS_UNIFORM_DWORDS 4U
#define PVRGPU_LIT_MESH_VS_UNIFORM_DWORDS 32U
#define PVRGPU_TEXTURE_VS_UNIFORM_DWORDS 32U
#define PVRGPU_TEXTURE_DESCRIPTOR_DWORDS PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS
#define PVRGPU_REFRACT_PREPASS_VS_UNIFORM_DWORDS 16U
#define PVRGPU_REFRACT_COMPOSITE_VS_UNIFORM_DWORDS 64U
#define PVRGPU_REFRACT_TEXTURE_COUNT PVRGPU_PCO_REFRACT_TEXTURE_COUNT
#define PVRGPU_REFRACT_DESCRIPTOR_DWORDS \
   PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS
#define PVRGPU_IDEAS_SIMPLE_VS_UNIFORM_DWORDS 32U
#define PVRGPU_IDEAS_LIGHTING_VS_UNIFORM_DWORDS 44U
#define PVRGPU_IDEAS_LIGHTING_FS_UNIFORM_DWORDS 12U
#define PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS 8U
#define PVRGPU_TERRAIN_MAIN_VS_UNIFORM_DWORDS 56U
#define PVRGPU_TERRAIN_MAIN_FS_UNIFORM_DWORDS 64U

struct pvrgpu_pco_compiler {
   void *mem_ctx;
   struct pvr_device_info dev_info;
   struct pvr_device_runtime_info runtime_info;
   pco_ctx *pco;
};

static uint64_t
pvrgpu_refract_descriptor_bits(uint64_t value,
                               unsigned first,
                               unsigned last)
{
   const unsigned width = last - first + 1U;
   const uint64_t mask = (UINT64_C(1) << width) - 1U;
   return (value & mask) << first;
}

static void
pvrgpu_refract_descriptor_store_u64(uint32_t *words,
                                    unsigned first_dword,
                                    uint64_t value)
{
   words[first_dword] = (uint32_t)value;
   words[first_dword + 1U] = (uint32_t)(value >> 32U);
}

static void
pvrgpu_build_refract_descriptor(uint32_t descriptor[20],
                                unsigned tex_format,
                                bool depth_swizzle,
                                bool alpha_one,
                                unsigned width,
                                unsigned height,
                                unsigned mip_count,
                                uint32_t byte_size,
                                unsigned min_filter,
                                unsigned mag_filter,
                                unsigned mip_filter,
                                unsigned wrap_u,
                                unsigned wrap_v,
                                unsigned max_lod_u4_6)
{
   memset(descriptor, 0, 20U * sizeof(descriptor[0]));
   const uint64_t image_word0 =
      pvrgpu_refract_descriptor_bits(4U, 0, 2) |
      pvrgpu_refract_descriptor_bits(depth_swizzle || alpha_one ? 4U : 3U,
                                     5,
                                     7) |
      pvrgpu_refract_descriptor_bits(depth_swizzle ? 0U : 2U, 8, 10) |
      pvrgpu_refract_descriptor_bits(depth_swizzle ? 0U : 1U, 11, 13) |
      pvrgpu_refract_descriptor_bits(0U, 14, 16) |
      pvrgpu_refract_descriptor_bits(tex_format, 27, 33) |
      pvrgpu_refract_descriptor_bits(width - 1U, 34, 47) |
      pvrgpu_refract_descriptor_bits(height - 1U, 48, 61);
   pvrgpu_refract_descriptor_store_u64(descriptor, 0, image_word0);

   /* IMAGE_WORD1 address bits 16..53 intentionally remain zero until the
    * bridge has deep-copied and allocated each structured resource. */
   const uint64_t image_word1 =
      pvrgpu_refract_descriptor_bits(width - 1U, 0, 14) |
      pvrgpu_refract_descriptor_bits(mip_count > 1U, 15, 15) |
      pvrgpu_refract_descriptor_bits(mip_count, 60, 63);
   pvrgpu_refract_descriptor_store_u64(descriptor, 2, image_word1);
   descriptor[4] = byte_size;

   const uint64_t sampler_word0 =
      pvrgpu_refract_descriptor_bits(4095U, 0, 12) |
      pvrgpu_refract_descriptor_bits(max_lod_u4_6, 23, 32) |
      pvrgpu_refract_descriptor_bits(wrap_u, 33, 35) |
      pvrgpu_refract_descriptor_bits(min_filter, 36, 37) |
      pvrgpu_refract_descriptor_bits(mag_filter, 38, 39) |
      pvrgpu_refract_descriptor_bits(mip_filter, 40, 40) |
      pvrgpu_refract_descriptor_bits(wrap_v, 41, 43);
   pvrgpu_refract_descriptor_store_u64(descriptor, 8, sampler_word0);
   const uint64_t gather_word0 =
      sampler_word0 | pvrgpu_refract_descriptor_bits(1U, 36, 37) |
      pvrgpu_refract_descriptor_bits(1U, 38, 39);
   pvrgpu_refract_descriptor_store_u64(descriptor, 16, gather_word0);
}

static bool
pvrgpu_pco_refract_extent_layout(unsigned width,
                                 unsigned height,
                                 unsigned *mip_count,
                                 uint32_t *depth_bytes,
                                 uint32_t *color_bytes)
{
   if (!mip_count || !depth_bytes || !color_bytes ||
       !((width == 160U && height == 120U) ||
         (width == 1600U && height == 1200U)))
      return false;

   uint64_t total = 0;
   unsigned levels = 0;
   unsigned level_width = width;
   unsigned level_height = height;
   for (;;) {
      total += (uint64_t)level_width * level_height * sizeof(uint32_t);
      levels++;
      if (total > UINT32_MAX || levels > 15U)
         return false;
      if (level_width == 1U && level_height == 1U)
         break;
      level_width = MAX2(level_width >> 1U, 1U);
      level_height = MAX2(level_height >> 1U, 1U);
   }

   const uint64_t depth_size =
      (uint64_t)width * height * sizeof(uint32_t);
   if (depth_size > UINT32_MAX)
      return false;
   *mip_count = levels;
   *depth_bytes = (uint32_t)depth_size;
   *color_bytes = (uint32_t)total;
   return true;
}

bool
pvrgpu_pco_build_refract_fragment_shared_for_extent(
   uint32_t out[PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS],
   unsigned width,
   unsigned height)
{
   unsigned mip_count = 0;
   uint32_t depth_bytes = 0;
   uint32_t color_bytes = 0;
   if (!out || !pvrgpu_pco_refract_extent_layout(width,
                                                  height,
                                                  &mip_count,
                                                  &depth_bytes,
                                                  &color_bytes))
      return false;
   pvrgpu_build_refract_descriptor(&out[0],
                                    24U,
                                    true,
                                    false,
                                    width,
                                    height,
                                    1U,
                                    depth_bytes,
                                    0U,
                                    0U,
                                    0U,
                                    2U,
                                    2U,
                                    0U);
   pvrgpu_build_refract_descriptor(&out[20],
                                    12U,
                                    false,
                                    false,
                                    width,
                                    height,
                                    mip_count,
                                    color_bytes,
                                    1U,
                                    1U,
                                    1U,
                                    2U,
                                    2U,
                                    (mip_count - 1U) * 64U);
   pvrgpu_build_refract_descriptor(&out[40],
                                    12U,
                                    false,
                                    false,
                                    512U,
                                    512U,
                                    1U,
                                    1048576U,
                                    1U,
                                    1U,
                                    0U,
                                    2U,
                                    2U,
                                    0U);
   return true;
}

void
pvrgpu_pco_build_refract_fragment_shared(
   uint32_t out[PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS])
{
   (void)pvrgpu_pco_build_refract_fragment_shared_for_extent(out,
                                                              160U,
                                                              120U);
}

bool
pvrgpu_pco_build_shadow_fragment_shared_for_extent(
   uint32_t out[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS],
   unsigned width,
   unsigned height)
{
   if (!out || !((width == 160U && height == 120U) ||
                 (width == 1600U && height == 1200U)))
      return false;
   const uint64_t depth_bytes =
      (uint64_t)width * height * sizeof(uint32_t);
   if (depth_bytes > UINT32_MAX)
      return false;
   pvrgpu_build_refract_descriptor(out,
                                    24U,
                                    true,
                                    false,
                                    width,
                                    height,
                                    1U,
                                    (uint32_t)depth_bytes,
                                    0U,
                                    0U,
                                    0U,
                                    2U,
                                    2U,
                                    0U);
   return true;
}

void
pvrgpu_pco_build_shadow_fragment_shared(
   uint32_t out[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS])
{
   (void)pvrgpu_pco_build_shadow_fragment_shared_for_extent(out,
                                                             160U,
                                                             120U);
}

bool
pvrgpu_pco_build_terrain_texture_descriptor(
   uint32_t out[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS],
   enum pipe_format format,
   unsigned width,
   unsigned height,
   unsigned mip_count,
   uint32_t byte_size,
   unsigned min_filter,
   unsigned mag_filter,
   unsigned mip_filter,
   unsigned wrap_u,
   unsigned wrap_v,
   unsigned max_lod_u4_6)
{
   if (!out ||
       (format != PIPE_FORMAT_R8G8B8A8_UNORM &&
        format != PIPE_FORMAT_R8G8B8X8_UNORM) ||
       width == 0 || width > 16384U || height == 0 || height > 16384U ||
       mip_count == 0 || mip_count > 15U || byte_size == 0 ||
       min_filter > 1U || mag_filter > 1U || mip_filter > 1U ||
       (wrap_u != 0U && wrap_u != 2U) ||
       (wrap_v != 0U && wrap_v != 2U) || max_lod_u4_6 > 1023U)
      return false;

   pvrgpu_build_refract_descriptor(
      out,
      12U,
      false,
      format == PIPE_FORMAT_R8G8B8X8_UNORM,
      width,
      height,
      mip_count,
      byte_size,
      min_filter,
      mag_filter,
      mip_filter,
      wrap_u,
      wrap_v,
      max_lod_u4_6);
   return true;
}

struct pvrgpu_conditionals_profile {
   unsigned uniform_loads;
   unsigned fract_ops;
   unsigned compare_ops;
   unsigned select_ops;
   unsigned blocks;
};

struct pvrgpu_lit_mesh_desc {
   const char *name;
   uint32_t vertex_source_hash[8];
   uint32_t fragment_source_hash[8];
   unsigned varying_components;
};

static const struct pvrgpu_lit_mesh_desc pvrgpu_lit_mesh_profiles[] = {
   [PVRGPU_PCO_LIT_MESH_BUILD] = {
      .name = "build",
      .vertex_source_hash = {
         UINT32_C(0x750ac3d1), UINT32_C(0xe9ceafcc),
         UINT32_C(0xdd1263dd), UINT32_C(0xa22a457b),
         UINT32_C(0x3b8ebb47), UINT32_C(0xa4ee0e8e),
         UINT32_C(0xeb2663ea), UINT32_C(0x6ad452cd),
      },
      .fragment_source_hash = {
         UINT32_C(0x8105bebf), UINT32_C(0x60cef3c7),
         UINT32_C(0xc9c3e978), UINT32_C(0xd20442bc),
         UINT32_C(0x46d83156), UINT32_C(0x9a4abb0b),
         UINT32_C(0xd1a4de24), UINT32_C(0x422a9790),
      },
      .varying_components = 1,
   },
   [PVRGPU_PCO_LIT_MESH_BUMP] = {
      .name = "bump",
      .vertex_source_hash = {
         UINT32_C(0x447e9e1f), UINT32_C(0xb6e0a9b9),
         UINT32_C(0xf5dcefa9), UINT32_C(0xf987adef),
         UINT32_C(0x8c416544), UINT32_C(0xb0956e81),
         UINT32_C(0xc5d8865b), UINT32_C(0x7b2850a7),
      },
      .fragment_source_hash = {
         UINT32_C(0x4f55ff2c), UINT32_C(0x8d248356),
         UINT32_C(0x20aea0e2), UINT32_C(0xee5248d5),
         UINT32_C(0x777abff2), UINT32_C(0xc13daa4d),
         UINT32_C(0xcb78bfc9), UINT32_C(0xc09ce498),
      },
      .varying_components = 3,
   },
   [PVRGPU_PCO_LIT_MESH_SHADING] = {
      .name = "shading",
      .vertex_source_hash = {
         UINT32_C(0x55a3db4e), UINT32_C(0x0781726a),
         UINT32_C(0xa9aaf326), UINT32_C(0x1663be77),
         UINT32_C(0x0b6194eb), UINT32_C(0xdd4b6265),
         UINT32_C(0x3351e890), UINT32_C(0x7acfdd9a),
      },
      .fragment_source_hash = {
         UINT32_C(0x8105bebf), UINT32_C(0x60cef3c7),
         UINT32_C(0xc9c3e978), UINT32_C(0xd20442bc),
         UINT32_C(0x46d83156), UINT32_C(0x9a4abb0b),
         UINT32_C(0xd1a4de24), UINT32_C(0x422a9790),
      },
      .varying_components = 3,
   },
};

static const uint32_t pvrgpu_texture_vertex_source_hash[8] = {
   UINT32_C(0x750ac3d1), UINT32_C(0xe9ceafcc),
   UINT32_C(0xdd1263dd), UINT32_C(0xa22a457b),
   UINT32_C(0x3b8ebb47), UINT32_C(0xa4ee0e8e),
   UINT32_C(0xeb2663ea), UINT32_C(0x6ad452cd),
};

static const uint32_t pvrgpu_texture_fragment_source_hash[8] = {
   UINT32_C(0xf95c6a3f), UINT32_C(0x2572cc32),
   UINT32_C(0x635afdeb), UINT32_C(0xff4d47cb),
   UINT32_C(0x5a3d3c87), UINT32_C(0x94e6645c),
   UINT32_C(0x9dde3b59), UINT32_C(0x233b4b47),
};

struct pvrgpu_refract_pco_desc {
   const char *name;
   uint32_t vertex_source_hash[8];
   uint32_t fragment_source_hash[8];
   unsigned vertex_uniform_dwords;
   unsigned vertex_uniform_loads;
   unsigned varying_components;
   unsigned texture_count;
};

static const struct pvrgpu_refract_pco_desc pvrgpu_refract_profiles[] = {
   [PVRGPU_PCO_REFRACT_PREPASS] = {
      .name = "refract-prepass",
      .vertex_source_hash = {
         UINT32_C(0xe4968762), UINT32_C(0x33f7e5d3),
         UINT32_C(0x8ca90f67), UINT32_C(0xb97709d1),
         UINT32_C(0x1b1e6e02), UINT32_C(0x56181388),
         UINT32_C(0xe52466c0), UINT32_C(0xccbf4647),
      },
      .fragment_source_hash = {
         UINT32_C(0xe4296e61), UINT32_C(0x386f509f),
         UINT32_C(0x599bdacb), UINT32_C(0x3631c37e),
         UINT32_C(0x480c24bb), UINT32_C(0xd56abe19),
         UINT32_C(0x82725b7b), UINT32_C(0xd046dab0),
      },
      .vertex_uniform_dwords = PVRGPU_REFRACT_PREPASS_VS_UNIFORM_DWORDS,
      .vertex_uniform_loads = 4,
      .varying_components = 3,
   },
   [PVRGPU_PCO_REFRACT_COMPOSITE] = {
      .name = "refract-composite",
      .vertex_source_hash = {
         UINT32_C(0x27553973), UINT32_C(0x14f4744c),
         UINT32_C(0x155ef398), UINT32_C(0xa57dee0b),
         UINT32_C(0x7dd7d4b3), UINT32_C(0xcf1e855e),
         UINT32_C(0xe83f6f81), UINT32_C(0xdb7fe462),
      },
      .fragment_source_hash = {
         UINT32_C(0xfcfc470a), UINT32_C(0x6e9eeb2b),
         UINT32_C(0x95810825), UINT32_C(0x6d2d6953),
         UINT32_C(0x04dc8732), UINT32_C(0x24b8c4c7),
         UINT32_C(0xac08a61d), UINT32_C(0x28ed4375),
      },
      .vertex_uniform_dwords = PVRGPU_REFRACT_COMPOSITE_VS_UNIFORM_DWORDS,
      .vertex_uniform_loads = 16,
      .varying_components = 11,
      .texture_count = PVRGPU_REFRACT_TEXTURE_COUNT,
   },
};

struct pvrgpu_shadow_pco_desc {
   const char *name;
   uint32_t vertex_source_hash[8];
   uint32_t fragment_source_hash[8];
   unsigned vertex_uniform_dwords;
   unsigned vertex_uniform_used_dwords;
   unsigned vertex_uniform_loads;
   unsigned varying_components;
};

static const struct pvrgpu_shadow_pco_desc pvrgpu_shadow_profiles[] = {
   [PVRGPU_PCO_SHADOW_DEPTH] = {
      .name = "shadow-depth",
      .vertex_source_hash = {
         UINT32_C(0xe4968762), UINT32_C(0x33f7e5d3),
         UINT32_C(0x8ca90f67), UINT32_C(0xb97709d1),
         UINT32_C(0x1b1e6e02), UINT32_C(0x56181388),
         UINT32_C(0xe52466c0), UINT32_C(0xccbf4647),
      },
      .fragment_source_hash = {
         UINT32_C(0xe4296e61), UINT32_C(0x386f509f),
         UINT32_C(0x599bdacb), UINT32_C(0x3631c37e),
         UINT32_C(0x480c24bb), UINT32_C(0xd56abe19),
         UINT32_C(0x82725b7b), UINT32_C(0xd046dab0),
      },
      .vertex_uniform_dwords = 16,
      .vertex_uniform_used_dwords = 16,
      .vertex_uniform_loads = 4,
      .varying_components = 3,
   },
   [PVRGPU_PCO_SHADOW_MASK] = {
      .name = "shadow-mask",
      .vertex_source_hash = {
         UINT32_C(0x981d39cf), UINT32_C(0x8316685f),
         UINT32_C(0x8e868a60), UINT32_C(0xf1850c32),
         UINT32_C(0xcdac5c34), UINT32_C(0x92c6393c),
         UINT32_C(0x6fc446df), UINT32_C(0x4830846f),
      },
      .fragment_source_hash = {
         UINT32_C(0x49a27748), UINT32_C(0xda81bbdf),
         UINT32_C(0x385da52a), UINT32_C(0xfa4c9087),
         UINT32_C(0x8edbc5ce), UINT32_C(0x0f9e5f75),
         UINT32_C(0xcb295bda), UINT32_C(0x20c14281),
      },
      .vertex_uniform_dwords = 32,
      .vertex_uniform_used_dwords = 32,
      .vertex_uniform_loads = 6,
      .varying_components = 4,
   },
   [PVRGPU_PCO_SHADOW_SCENE] = {
      .name = "shadow-scene",
      .vertex_source_hash = {
         UINT32_C(0xda5546ad), UINT32_C(0x634bf23c),
         UINT32_C(0x36c9d6bd), UINT32_C(0x89e696ca),
         UINT32_C(0xc138e003), UINT32_C(0x445bb465),
         UINT32_C(0xb580e98a), UINT32_C(0x7f1d0c5b),
      },
      .fragment_source_hash = {
         UINT32_C(0x8105bebf), UINT32_C(0x60cef3c7),
         UINT32_C(0xc9c3e978), UINT32_C(0xd20442bc),
         UINT32_C(0x46d83156), UINT32_C(0x9a4abb0b),
         UINT32_C(0xd1a4de24), UINT32_C(0x422a9790),
      },
      .vertex_uniform_dwords = 32,
      .vertex_uniform_used_dwords = 31,
      .vertex_uniform_loads = 8,
      .varying_components = 1,
   },
};

struct pvrgpu_terrain_pco_desc {
   const char *name;
   const uint32_t *vertex_source_hash;
   uint32_t fragment_source_hash[8];
   uint32_t fragment_source_hash_800x600[8];
   unsigned vertex_uniform_dwords;
   unsigned vertex_uniform_used_dwords;
   unsigned vertex_uniform_loads;
   unsigned fragment_uniform_dwords;
   unsigned fragment_uniform_used_dwords;
   unsigned fragment_uniform_loads;
   unsigned vertex_texture_count;
   unsigned fragment_texture_count;
   unsigned fragment_texture_ops;
   unsigned varying_components;
};

static const uint32_t pvrgpu_terrain_fullscreen_vertex_hash[8] = {
   UINT32_C(0x64295326), UINT32_C(0xb7892f55),
   UINT32_C(0x40f4c2f4), UINT32_C(0x2524db37),
   UINT32_C(0x4a6849f7), UINT32_C(0x4212d7bb),
   UINT32_C(0x51ad07d2), UINT32_C(0x41c1b894),
};

static const uint32_t pvrgpu_terrain_main_vertex_hash[8] = {
   UINT32_C(0xfdbd1d43), UINT32_C(0x307b26eb),
   UINT32_C(0x2982014c), UINT32_C(0x02f9dd89),
   UINT32_C(0x6199f3cb), UINT32_C(0x8ba2d70d),
   UINT32_C(0xbbb04d5f), UINT32_C(0x6621cac2),
};

static const struct pvrgpu_terrain_pco_desc pvrgpu_terrain_profiles[] = {
   [PVRGPU_PCO_TERRAIN_D1] = {
      .name = "terrain-d1-noise",
      .vertex_source_hash = pvrgpu_terrain_fullscreen_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0x72c70927), UINT32_C(0x1eee611e),
         UINT32_C(0xa4dd2fdc), UINT32_C(0xcd490824),
         UINT32_C(0xd08e0f18), UINT32_C(0xbd42bba9),
         UINT32_C(0x3508625b), UINT32_C(0xfa3c6660),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 6,
      .vertex_uniform_loads = 2,
      .fragment_uniform_dwords = 4,
      .fragment_uniform_used_dwords = 1,
      .fragment_uniform_loads = 1,
      .varying_components = 2,
   },
   [PVRGPU_PCO_TERRAIN_D2] = {
      .name = "terrain-d2-normal-map",
      .vertex_source_hash = pvrgpu_terrain_fullscreen_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0x401f2183), UINT32_C(0xbf28b09d),
         UINT32_C(0x8f3006aa), UINT32_C(0x9770f1a5),
         UINT32_C(0x183eb0a6), UINT32_C(0x21c3909c),
         UINT32_C(0x7ebceac5), UINT32_C(0x63eeefc1),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 6,
      .vertex_uniform_loads = 2,
      .fragment_uniform_dwords = 8,
      .fragment_uniform_used_dwords = 6,
      .fragment_uniform_loads = 2,
      .fragment_texture_count = 1,
      .fragment_texture_ops = 3,
      .varying_components = 2,
   },
   [PVRGPU_PCO_TERRAIN_D3] = {
      .name = "terrain-d3-main",
      .vertex_source_hash = pvrgpu_terrain_main_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0x1a5275b3), UINT32_C(0xa5aed1dd),
         UINT32_C(0x0d1c076a), UINT32_C(0x1839fb95),
         UINT32_C(0x31c35861), UINT32_C(0x717b45c9),
         UINT32_C(0xb9f92d1f), UINT32_C(0x9c4f0d76),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_MAIN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 53,
      .vertex_uniform_loads = 14,
      .fragment_uniform_dwords = PVRGPU_TERRAIN_MAIN_FS_UNIFORM_DWORDS,
      .fragment_uniform_used_dwords = 61,
      .fragment_uniform_loads = 16,
      .vertex_texture_count = 2,
      .fragment_texture_count = 5,
      .fragment_texture_ops = 5,
      .varying_components = 14,
   },
   [PVRGPU_PCO_TERRAIN_D4] = {
      .name = "terrain-d4-blur",
      .vertex_source_hash = pvrgpu_terrain_fullscreen_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0xa49328b3), UINT32_C(0xebd09eb6),
         UINT32_C(0xb68bf6c9), UINT32_C(0x98f24299),
         UINT32_C(0xa30fa19d), UINT32_C(0xc6c7159a),
         UINT32_C(0xba654c0a), UINT32_C(0x6d39f49c),
      },
      .fragment_source_hash_800x600 = {
         UINT32_C(0x047e72ee), UINT32_C(0xebcdacaf),
         UINT32_C(0x790fa9ab), UINT32_C(0x0b4e424c),
         UINT32_C(0x993de98e), UINT32_C(0x034ed49e),
         UINT32_C(0x858f25cc), UINT32_C(0x1b3bbce9),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 6,
      .vertex_uniform_loads = 2,
      .fragment_texture_count = 1,
      .fragment_texture_ops = 5,
      .varying_components = 2,
   },
   [PVRGPU_PCO_TERRAIN_D5] = {
      .name = "terrain-d5-blur",
      .vertex_source_hash = pvrgpu_terrain_fullscreen_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0xc73e207c), UINT32_C(0x4ef26404),
         UINT32_C(0x604847d2), UINT32_C(0xfeebd5e6),
         UINT32_C(0xbea9c632), UINT32_C(0x79454818),
         UINT32_C(0xa5c408bc), UINT32_C(0x586cbda5),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 6,
      .vertex_uniform_loads = 2,
      .fragment_texture_count = 1,
      .fragment_texture_ops = 5,
      .varying_components = 2,
   },
   [PVRGPU_PCO_TERRAIN_D6] = {
      .name = "terrain-d6-composite",
      .vertex_source_hash = pvrgpu_terrain_fullscreen_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0x0e6f1ac2), UINT32_C(0xd8813568),
         UINT32_C(0x336936c8), UINT32_C(0xc6840a8c),
         UINT32_C(0xdd9ba6ee), UINT32_C(0x627d4cf7),
         UINT32_C(0x88cb830e), UINT32_C(0xd538aa03),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 6,
      .vertex_uniform_loads = 2,
      .fragment_uniform_dwords = 4,
      .fragment_uniform_used_dwords = 1,
      .fragment_uniform_loads = 1,
      .fragment_texture_count = 1,
      .fragment_texture_ops = 1,
      .varying_components = 2,
   },
   [PVRGPU_PCO_TERRAIN_D7] = {
      .name = "terrain-d7-blur",
      .vertex_source_hash = pvrgpu_terrain_fullscreen_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0x8bf6b63e), UINT32_C(0x59d22899),
         UINT32_C(0x88554142), UINT32_C(0x383e51c8),
         UINT32_C(0x90641bf4), UINT32_C(0xc10a2fc3),
         UINT32_C(0x3207cac2), UINT32_C(0x56282ca8),
      },
      .fragment_source_hash_800x600 = {
         UINT32_C(0xabd6119b), UINT32_C(0xf3d3832b),
         UINT32_C(0x2202471e), UINT32_C(0x9ce97e85),
         UINT32_C(0x5b195cc0), UINT32_C(0x506c744f),
         UINT32_C(0x60821226), UINT32_C(0xff994103),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 6,
      .vertex_uniform_loads = 2,
      .fragment_texture_count = 1,
      .fragment_texture_ops = 9,
      .varying_components = 2,
   },
   [PVRGPU_PCO_TERRAIN_D8] = {
      .name = "terrain-d8-blur",
      .vertex_source_hash = pvrgpu_terrain_fullscreen_vertex_hash,
      .fragment_source_hash = {
         UINT32_C(0xbbe35ba4), UINT32_C(0x0f0884d6),
         UINT32_C(0x054a182f), UINT32_C(0xb0b5363f),
         UINT32_C(0x69b62005), UINT32_C(0x3c8b1b0f),
         UINT32_C(0xef2f4a55), UINT32_C(0x48f9a4f9),
      },
      .fragment_source_hash_800x600 = {
         UINT32_C(0xdf8bc70f), UINT32_C(0xd94991ef),
         UINT32_C(0xc7ac49c7), UINT32_C(0x0f850b31),
         UINT32_C(0x7af60cab), UINT32_C(0x47e06843),
         UINT32_C(0x931eb72a), UINT32_C(0x7be58c3a),
      },
      .vertex_uniform_dwords = PVRGPU_TERRAIN_FULLSCREEN_VS_UNIFORM_DWORDS,
      .vertex_uniform_used_dwords = 6,
      .vertex_uniform_loads = 2,
      .fragment_texture_count = 1,
      .fragment_texture_ops = 9,
      .varying_components = 2,
   },
};

struct pvrgpu_ideas_pco_desc {
   const char *name;
   uint32_t vertex_source_hash[8];
   uint32_t fragment_source_hash[8];
   unsigned attribute_count;
   unsigned varying_components;
   unsigned vertex_shared_dwords;
   unsigned fragment_shared_dwords;
};

static const struct pvrgpu_ideas_pco_desc pvrgpu_ideas_profiles[] = {
   [PVRGPU_PCO_IDEAS_LOGO] = {
      .name = "ideas-logo",
      .vertex_source_hash = {
         UINT32_C(0x29b86315), UINT32_C(0xc5e4fb02),
         UINT32_C(0xbf5e75cd), UINT32_C(0x02406082),
         UINT32_C(0x7ee4cc83), UINT32_C(0x30e777ba),
         UINT32_C(0x906e28a4), UINT32_C(0xc71cc0f4),
      },
      .fragment_source_hash = {
         UINT32_C(0xcc1251ba), UINT32_C(0xfc62903d),
         UINT32_C(0xe2f27dfe), UINT32_C(0x423c07a0),
         UINT32_C(0xff9dc3b2), UINT32_C(0xa7461f1e),
         UINT32_C(0xdcafab81), UINT32_C(0x479dac74),
      },
      .attribute_count = 1,
      .vertex_shared_dwords = 32,
      .fragment_shared_dwords = 4,
   },
   [PVRGPU_PCO_IDEAS_LIGHTING] = {
      .name = "ideas-lighting",
      .vertex_source_hash = {
         UINT32_C(0xeac6f051), UINT32_C(0xe96a384d),
         UINT32_C(0x401bc339), UINT32_C(0x2c58043f),
         UINT32_C(0x2861e978), UINT32_C(0xc1ef625c),
         UINT32_C(0x015af337), UINT32_C(0x495f1917),
      },
      .fragment_source_hash = {
         UINT32_C(0x58557edd), UINT32_C(0x2eab1772),
         UINT32_C(0xeb3c62b6), UINT32_C(0x00c61868),
         UINT32_C(0xef79a373), UINT32_C(0x82a55503),
         UINT32_C(0x90bd4936), UINT32_C(0x8de934a0),
      },
      .attribute_count = 2,
      .varying_components = 10,
      .vertex_shared_dwords = 44,
      .fragment_shared_dwords = 12,
   },
   [PVRGPU_PCO_IDEAS_WHITE] = {
      .name = "ideas-white",
      .vertex_source_hash = {
         UINT32_C(0x88a968c3), UINT32_C(0xb205151f),
         UINT32_C(0xb07b9044), UINT32_C(0xc362aa72),
         UINT32_C(0x7f573e93), UINT32_C(0x31aa797a),
         UINT32_C(0x93bf37fe), UINT32_C(0x3fa00942),
      },
      .fragment_source_hash = {
         UINT32_C(0x3857474b), UINT32_C(0x0de55295),
         UINT32_C(0x56692dbd), UINT32_C(0x31e4406b),
         UINT32_C(0xdae8e19f), UINT32_C(0xcde306ec),
         UINT32_C(0xbb732499), UINT32_C(0xb2717c8f),
      },
      .attribute_count = 1,
      .vertex_shared_dwords = 32,
   },
   [PVRGPU_PCO_IDEAS_BLACK] = {
      .name = "ideas-black",
      .vertex_source_hash = {
         UINT32_C(0xdd8e29e5), UINT32_C(0x17828bdb),
         UINT32_C(0x22155b0f), UINT32_C(0x481da868),
         UINT32_C(0x2b9471ba), UINT32_C(0x8931349d),
         UINT32_C(0x9acc7bf5), UINT32_C(0x97a88d3d),
      },
      .fragment_source_hash = {
         UINT32_C(0x3857474b), UINT32_C(0x0de55295),
         UINT32_C(0x56692dbd), UINT32_C(0x31e4406b),
         UINT32_C(0xdae8e19f), UINT32_C(0xcde306ec),
         UINT32_C(0xbb732499), UINT32_C(0xb2717c8f),
      },
      .attribute_count = 1,
      .vertex_shared_dwords = 32,
   },
};

static bool
pvrgpu_pco_fail(char *error, size_t error_size, const char *format, ...)
{
   if (error && error_size) {
      va_list args;
      va_start(args, format);
      vsnprintf(error, error_size, format, args);
      va_end(args);
      error[error_size - 1] = '\0';
   }
   return false;
}

static bool
pvrgpu_source_hash_matches(const nir_shader *shader,
                           const uint32_t expected_words[8])
{
   if (!shader || !expected_words)
      return false;

   uint8_t expected[32];
   for (unsigned word = 0; word < 8; ++word) {
      expected[word * 4 + 0] = (uint8_t)(expected_words[word] >> 0);
      expected[word * 4 + 1] = (uint8_t)(expected_words[word] >> 8);
      expected[word * 4 + 2] = (uint8_t)(expected_words[word] >> 16);
      expected[word * 4 + 3] = (uint8_t)(expected_words[word] >> 24);
   }
   return memcmp(shader->info.source_blake3, expected, sizeof(expected)) == 0;
}

static bool
pvrgpu_texture_float_vector_variable(const nir_variable *var,
                                     nir_variable_mode mode,
                                     int location,
                                     unsigned components)
{
   return var && var->data.mode == mode && var->data.location == location &&
          var->data.location_frac == 0 && glsl_type_is_vector(var->type) &&
          glsl_get_base_type(var->type) == GLSL_TYPE_FLOAT &&
          glsl_get_bit_size(var->type) == 32 &&
          glsl_get_vector_elements(var->type) == components;
}

static bool
pvrgpu_texture_sampler_variable(const nir_variable *var)
{
   return var && var->data.mode == nir_var_uniform &&
          var->data.binding == 0 && glsl_type_is_sampler(var->type) &&
          glsl_get_sampler_dim(var->type) == GLSL_SAMPLER_DIM_2D &&
          !glsl_sampler_type_is_array(var->type) &&
          !glsl_sampler_type_is_shadow(var->type);
}

static bool
pvrgpu_validate_texture_variables(const nir_shader *nir,
                                  char *error,
                                  size_t error_size)
{
   unsigned inputs = 0;
   unsigned outputs = 0;
   unsigned uniforms = 0;

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      unsigned input_mask = 0;
      bool has_position = false;
      bool has_varying = false;
      nir_foreach_variable_with_modes (var, nir, nir_var_shader_in) {
         ++inputs;
         if (var->data.location < VERT_ATTRIB_GENERIC0 ||
             var->data.location > VERT_ATTRIB_GENERIC2 ||
             !pvrgpu_texture_float_vector_variable(var,
                                                   nir_var_shader_in,
                                                   var->data.location,
                                                   4)) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "texture VS input ABI mismatch");
         }
         const unsigned bit = var->data.location - VERT_ATTRIB_GENERIC0;
         if (input_mask & BITFIELD_BIT(bit))
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "texture VS has duplicate input");
         input_mask |= BITFIELD_BIT(bit);
      }

      nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
         ++outputs;
         if (pvrgpu_texture_float_vector_variable(var,
                                                  nir_var_shader_out,
                                                  VARYING_SLOT_POS,
                                                  4)) {
            if (has_position)
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "texture VS has duplicate position");
            has_position = true;
         } else if (pvrgpu_texture_float_vector_variable(
                       var,
                       nir_var_shader_out,
                       VARYING_SLOT_VAR0,
                       3) &&
                    var->data.precision == GLSL_PRECISION_MEDIUM) {
            if (has_varying)
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "texture VS has duplicate varying");
            has_varying = true;
         } else {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "texture VS output ABI mismatch");
         }
      }

      unsigned uniform_mask = 0;
      nir_foreach_variable_with_modes (var, nir, nir_var_uniform) {
         ++uniforms;
         if (var->data.location < 0 || var->data.location > 1 ||
             !glsl_type_is_matrix(var->type) ||
             glsl_get_base_type(var->type) != GLSL_TYPE_FLOAT ||
             glsl_get_bit_size(var->type) != 32 ||
             glsl_get_vector_elements(var->type) != 4 ||
             glsl_get_matrix_columns(var->type) != 4 ||
             (uniform_mask & BITFIELD_BIT(var->data.location))) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "texture VS uniform ABI mismatch");
         }
         uniform_mask |= BITFIELD_BIT(var->data.location);
      }

      if (inputs != 3 || input_mask != 0x7 || outputs != 2 ||
          !has_position || !has_varying || uniforms != 2 ||
          uniform_mask != 0x3) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "texture VS variable signature mismatch");
      }
      return true;
   }

   bool has_varying = false;
   bool has_color = false;
   nir_foreach_variable_with_modes (var, nir, nir_var_shader_in) {
      ++inputs;
      if (has_varying ||
          !pvrgpu_texture_float_vector_variable(var,
                                                nir_var_shader_in,
                                                VARYING_SLOT_VAR0,
                                                3) ||
          var->data.interpolation != INTERP_MODE_SMOOTH ||
          var->data.precision != GLSL_PRECISION_MEDIUM) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "texture FS varying ABI mismatch");
      }
      has_varying = true;
   }
   nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
      ++outputs;
      if (has_color ||
          (!pvrgpu_texture_float_vector_variable(var,
                                                 nir_var_shader_out,
                                                 FRAG_RESULT_COLOR,
                                                 4) &&
           !pvrgpu_texture_float_vector_variable(var,
                                                 nir_var_shader_out,
                                                 FRAG_RESULT_DATA0,
                                                 4)) ||
          var->data.precision != GLSL_PRECISION_MEDIUM) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "texture FS color ABI mismatch");
      }
      has_color = true;
   }
   nir_foreach_variable_with_modes (var, nir, nir_var_uniform) {
      ++uniforms;
      if (!pvrgpu_texture_sampler_variable(var)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "texture FS sampler ABI mismatch");
      }
   }
   if (inputs != 1 || outputs != 1 || uniforms != 1 || !has_varying ||
       !has_color) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture FS variable signature mismatch");
   }
   return true;
}

static bool
pvrgpu_texture_alu_source_matches(const nir_alu_src *src,
                                  const nir_def *def,
                                  const unsigned *swizzle,
                                  unsigned components)
{
   if (!src || !def || src->src.ssa != def)
      return false;
   for (unsigned component = 0; component < components; ++component) {
      if (src->swizzle[component] != swizzle[component])
         return false;
   }
   return true;
}

static bool
pvrgpu_validate_texture_uniform(const nir_intrinsic_instr *intr,
                                unsigned *seen_dwords,
                                char *error,
                                size_t error_size)
{
   if (intr->def.bit_size != 32 || intr->def.num_components != 4 ||
       (nir_intrinsic_base(intr) != 0 && nir_intrinsic_base(intr) != 4) ||
       nir_intrinsic_range(intr) != 4 ||
       nir_intrinsic_dest_type(intr) != nir_type_float32 ||
       !nir_src_is_const(intr->src[0])) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture VS load_uniform ABI mismatch");
   }

   const unsigned row = nir_src_as_uint(intr->src[0]);
   const unsigned dword = nir_intrinsic_base(intr) + row;
   if (row >= 4 || dword >= 8 || (*seen_dwords & BITFIELD_BIT(dword))) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture VS uniform row is invalid or duplicate");
   }
   *seen_dwords |= BITFIELD_BIT(dword);
   return true;
}

struct pvrgpu_texture_nir_profile {
   unsigned blocks;
   unsigned instructions;
   unsigned constants;
   unsigned derefs;
   unsigned load_derefs;
   unsigned store_derefs;
   unsigned uniform_loads;
   unsigned fmul;
   unsigned fadd;
   unsigned fdot3;
   unsigned frsq;
   unsigned fmax;
   unsigned vec2;
   unsigned vec3;
   unsigned vec4;
   unsigned textures;
};

static bool
pvrgpu_validate_texture_nir(const nir_shader *nir,
                            mesa_shader_stage expected_stage,
                            char *error,
                            size_t error_size)
{
   if (!nir || nir->info.stage != expected_stage ||
       (expected_stage != MESA_SHADER_VERTEX &&
        expected_stage != MESA_SHADER_FRAGMENT)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture shader stage mismatch");
   }
   if (!pvrgpu_validate_texture_variables(nir, error, error_size))
      return false;

   struct pvrgpu_texture_nir_profile profile = { 0 };
   unsigned functions = 0;
   unsigned seen_uniforms = 0;
   nir_intrinsic_instr *varying_load = NULL;
   nir_intrinsic_instr *varying_store = NULL;
   nir_intrinsic_instr *color_store = NULL;
   nir_alu_instr *coord_vector = NULL;
   nir_alu_instr *intensity_vector = NULL;
   nir_alu_instr *color_multiply = NULL;
   nir_alu_instr *varying_vector = NULL;
   nir_tex_instr *sample = NULL;
   nir_load_const_instr *one = NULL;

   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (!function->is_entrypoint || ++functions != 1) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "texture requires one NIR entrypoint");
      }
      nir_foreach_block (block, function->impl) {
         ++profile.blocks;
         nir_foreach_instr (instr, block) {
            ++profile.instructions;
            switch (instr->type) {
            case nir_instr_type_load_const: {
               nir_load_const_instr *constant = nir_instr_as_load_const(instr);
               ++profile.constants;
               if (expected_stage == MESA_SHADER_FRAGMENT) {
                  if (one || constant->def.bit_size != 32 ||
                      constant->def.num_components != 1 ||
                      constant->value[0].u32 != UINT32_C(0x3f800000)) {
                     return pvrgpu_pco_fail(error,
                                            error_size,
                                            "texture FS constant changed");
                  }
                  one = constant;
               }
               break;
            }
            case nir_instr_type_deref:
               if (nir_instr_as_deref(instr)->deref_type !=
                   nir_deref_type_var) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "texture contains non-variable deref");
               }
               ++profile.derefs;
               break;
            case nir_instr_type_intrinsic: {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_uniform) {
                  if (expected_stage != MESA_SHADER_VERTEX ||
                      !pvrgpu_validate_texture_uniform(intr,
                                                       &seen_uniforms,
                                                       error,
                                                       error_size))
                     return false;
                  ++profile.uniform_loads;
               } else if (intr->intrinsic == nir_intrinsic_load_deref) {
                  ++profile.load_derefs;
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (expected_stage == MESA_SHADER_FRAGMENT) {
                     if (varying_load || !var ||
                         var->data.mode != nir_var_shader_in ||
                         var->data.location != VARYING_SLOT_VAR0)
                        return pvrgpu_pco_fail(error,
                                               error_size,
                                               "texture FS varying load changed");
                     varying_load = intr;
                  }
               } else if (intr->intrinsic == nir_intrinsic_store_deref) {
                  ++profile.store_derefs;
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (expected_stage == MESA_SHADER_VERTEX && var &&
                      var->data.mode == nir_var_shader_out &&
                      var->data.location == VARYING_SLOT_VAR0) {
                     if (varying_store)
                        return pvrgpu_pco_fail(error,
                                               error_size,
                                               "texture VS duplicate varying store");
                     varying_store = intr;
                  } else if (expected_stage == MESA_SHADER_FRAGMENT && var &&
                             var->data.mode == nir_var_shader_out &&
                             (var->data.location == FRAG_RESULT_COLOR ||
                              var->data.location == FRAG_RESULT_DATA0)) {
                     if (color_store)
                        return pvrgpu_pco_fail(error,
                                               error_size,
                                               "texture FS duplicate color store");
                     color_store = intr;
                  }
               } else {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "texture contains unsupported NIR intrinsic %s",
                     nir_intrinsic_infos[intr->intrinsic].name);
               }
               break;
            }
            case nir_instr_type_alu: {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               switch (alu->op) {
               case nir_op_fmul:
                  ++profile.fmul;
                  if (expected_stage == MESA_SHADER_FRAGMENT) {
                     if (color_multiply)
                        return pvrgpu_pco_fail(error,
                                               error_size,
                                               "texture FS duplicate multiply");
                     color_multiply = alu;
                  }
                  break;
               case nir_op_fadd:
                  ++profile.fadd;
                  break;
               case nir_op_fdot3:
                  ++profile.fdot3;
                  break;
               case nir_op_frsq:
                  ++profile.frsq;
                  break;
               case nir_op_fmax:
                  ++profile.fmax;
                  break;
               case nir_op_vec2:
                  ++profile.vec2;
                  if (coord_vector)
                     return pvrgpu_pco_fail(error,
                                            error_size,
                                            "texture duplicate vec2");
                  coord_vector = alu;
                  break;
               case nir_op_vec3:
                  ++profile.vec3;
                  if (varying_vector)
                     return pvrgpu_pco_fail(error,
                                            error_size,
                                            "texture duplicate vec3");
                  varying_vector = alu;
                  break;
               case nir_op_vec4:
                  ++profile.vec4;
                  if (intensity_vector)
                     return pvrgpu_pco_fail(error,
                                            error_size,
                                            "texture duplicate vec4");
                  intensity_vector = alu;
                  break;
               default:
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "texture contains unsupported NIR ALU op %s",
                                         nir_op_infos[alu->op].name);
               }
               break;
            }
            case nir_instr_type_tex:
               if (expected_stage != MESA_SHADER_FRAGMENT || sample) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "texture contains unexpected sample");
               }
               sample = nir_instr_as_tex(instr);
               ++profile.textures;
               break;
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "texture contains unsupported NIR instruction type %u",
                                      instr->type);
            }
         }
      }
   }

   if (functions != 1 || profile.blocks == 0 || profile.blocks > 2) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture NIR control-flow signature mismatch");
   }

   if (expected_stage == MESA_SHADER_VERTEX) {
      if (profile.instructions != 41 || profile.constants != 5 ||
          profile.derefs != 5 || profile.load_derefs != 3 ||
          profile.store_derefs != 2 || profile.uniform_loads != 8 ||
          seen_uniforms != 0xff || profile.fmul != 7 ||
          profile.fadd != 6 || profile.fdot3 != 2 || profile.frsq != 1 ||
          profile.fmax != 1 || profile.vec2 != 0 || profile.vec3 != 1 ||
          profile.vec4 != 0 || profile.textures != 0 || !varying_store ||
          !varying_vector || varying_store->src[1].ssa != &varying_vector->def ||
          nir_intrinsic_write_mask(varying_store) != 0x7) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "texture VS canonical NIR signature mismatch");
      }
      return true;
   }

   if (profile.instructions != 9 || profile.constants != 1 ||
       profile.derefs != 2 || profile.load_derefs != 1 ||
       profile.store_derefs != 1 || profile.uniform_loads != 0 ||
       profile.fmul != 1 || profile.fadd != 0 || profile.fdot3 != 0 ||
       profile.frsq != 0 || profile.fmax != 0 || profile.vec2 != 1 ||
       profile.vec3 != 0 || profile.vec4 != 1 || profile.textures != 1 ||
       !varying_load || varying_load->def.num_components != 3 ||
       varying_load->def.bit_size != 32 || !coord_vector ||
       !intensity_vector || !color_multiply || !sample || !one ||
       !color_store) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture FS canonical NIR signature mismatch");
   }

   static const unsigned y_swizzle[1] = { 1 };
   static const unsigned z_swizzle[1] = { 2 };
   static const unsigned x_swizzle[1] = { 0 };
   if (!pvrgpu_texture_alu_source_matches(&coord_vector->src[0],
                                          &varying_load->def,
                                          y_swizzle,
                                          1) ||
       !pvrgpu_texture_alu_source_matches(&coord_vector->src[1],
                                          &varying_load->def,
                                          z_swizzle,
                                          1) ||
       sample->op != nir_texop_tex ||
       sample->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       sample->dest_type != nir_type_float32 ||
       sample->def.num_components != 4 || sample->def.bit_size != 32 ||
       sample->num_srcs != 1 || sample->coord_components != 2 ||
       sample->src[0].src_type != nir_tex_src_coord ||
       sample->src[0].src.ssa != &coord_vector->def || sample->is_array ||
       sample->is_shadow || sample->is_sparse ||
       sample->texture_non_uniform || sample->sampler_non_uniform ||
       sample->embedded_sampler || sample->offset_non_uniform ||
       sample->texture_index != 0 || sample->sampler_index != 0 ||
       sample->backend_flags != 0) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture FS sample graph changed");
   }

   for (unsigned component = 0; component < 3; ++component) {
      if (!pvrgpu_texture_alu_source_matches(&intensity_vector->src[component],
                                             &varying_load->def,
                                             x_swizzle,
                                             1)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "texture FS intensity vector changed");
      }
   }
   if (!pvrgpu_texture_alu_source_matches(&intensity_vector->src[3],
                                          &one->def,
                                          x_swizzle,
                                          1) ||
       color_multiply->def.num_components != 4 ||
       color_multiply->def.bit_size != 32 ||
       color_multiply->src[0].src.ssa != &sample->def ||
       color_multiply->src[1].src.ssa != &intensity_vector->def ||
       color_store->src[1].ssa != &color_multiply->def ||
       nir_intrinsic_write_mask(color_store) != 0xf) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture FS modulation graph changed");
   }
   return true;
}

struct pvrgpu_refract_nir_profile {
   unsigned blocks;
   unsigned instructions;
   unsigned constants;
   unsigned derefs;
   unsigned load_derefs;
   unsigned store_derefs;
   unsigned uniform_loads;
   unsigned fadd;
   unsigned fmul;
   unsigned fneg;
   unsigned fdot3;
   unsigned frsq;
   unsigned fsqrt;
   unsigned fdiv;
   unsigned flt;
   unsigned fmax;
   unsigned fpow;
   unsigned bcsel;
   unsigned vec3;
   unsigned vec4;
   unsigned textures;
};

static bool
pvrgpu_refract_float_vector_variable(const nir_variable *var,
                                      nir_variable_mode mode,
                                      int location,
                                      unsigned components)
{
   return var && var->data.mode == mode &&
          var->data.location == location && var->data.location_frac == 0 &&
          glsl_type_is_vector(var->type) &&
          glsl_get_base_type(var->type) == GLSL_TYPE_FLOAT &&
          glsl_get_bit_size(var->type) == 32 &&
          glsl_get_vector_elements(var->type) == components;
}

static bool
pvrgpu_refract_matrix_uniform(const nir_variable *var,
                              unsigned location)
{
   return var && var->data.mode == nir_var_uniform &&
          var->data.location == (int)location &&
          glsl_type_is_matrix(var->type) &&
          glsl_get_base_type(var->type) == GLSL_TYPE_FLOAT &&
          glsl_get_bit_size(var->type) == 32 &&
          glsl_get_vector_elements(var->type) == 4 &&
          glsl_get_matrix_columns(var->type) == 4;
}

static bool
pvrgpu_refract_sampler_uniform(const nir_variable *var,
                               unsigned binding)
{
   return var && var->data.mode == nir_var_uniform &&
          var->data.location == (int)(4 + binding) &&
          var->data.binding == binding && glsl_type_is_sampler(var->type) &&
          glsl_get_sampler_dim(var->type) == GLSL_SAMPLER_DIM_2D &&
          !glsl_sampler_type_is_array(var->type) &&
          !glsl_sampler_type_is_shadow(var->type);
}

static bool
pvrgpu_validate_refract_variables(
   const nir_shader *nir,
   enum pvrgpu_pco_refract_profile profile,
   char *error,
   size_t error_size)
{
   const bool composite = profile == PVRGPU_PCO_REFRACT_COMPOSITE;
   unsigned inputs = 0;
   unsigned outputs = 0;
   unsigned uniforms = 0;
   unsigned input_mask = 0;
   unsigned output_mask = 0;
   unsigned uniform_mask = 0;

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_in) {
      ++inputs;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if (var->data.location < VERT_ATTRIB_GENERIC0 ||
             var->data.location > VERT_ATTRIB_GENERIC1 ||
             !pvrgpu_refract_float_vector_variable(var,
                                                    nir_var_shader_in,
                                                    var->data.location,
                                                    4)) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "refract VS input ABI mismatch");
         }
         const unsigned bit = var->data.location - VERT_ATTRIB_GENERIC0;
         if (input_mask & BITFIELD_BIT(bit))
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "refract VS duplicate input");
         input_mask |= BITFIELD_BIT(bit);
         continue;
      }

      const int location = var->data.location;
      const unsigned components =
         !composite || location == VARYING_SLOT_VAR2 ? 3 : 4;
      const unsigned bit = location - VARYING_SLOT_VAR0;
      if (location < VARYING_SLOT_VAR0 ||
          location > (composite ? VARYING_SLOT_VAR2 : VARYING_SLOT_VAR0) ||
          !pvrgpu_refract_float_vector_variable(var,
                                                nir_var_shader_in,
                                                location,
                                                components) ||
          var->data.interpolation != INTERP_MODE_SMOOTH ||
          var->data.precision != GLSL_PRECISION_MEDIUM ||
          (input_mask & BITFIELD_BIT(bit))) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "refract FS varying ABI mismatch");
      }
      input_mask |= BITFIELD_BIT(bit);
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
      ++outputs;
      if (nir->info.stage == MESA_SHADER_FRAGMENT) {
         if ((!pvrgpu_refract_float_vector_variable(var,
                                                    nir_var_shader_out,
                                                    FRAG_RESULT_COLOR,
                                                    4) &&
              !pvrgpu_refract_float_vector_variable(var,
                                                    nir_var_shader_out,
                                                    FRAG_RESULT_DATA0,
                                                    4)) ||
             var->data.precision != GLSL_PRECISION_MEDIUM) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "refract FS color ABI mismatch");
         }
         output_mask |= 1;
         continue;
      }

      if (var->data.location == VARYING_SLOT_POS) {
         if (!pvrgpu_refract_float_vector_variable(var,
                                                   nir_var_shader_out,
                                                   VARYING_SLOT_POS,
                                                   4)) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "refract VS position ABI mismatch");
         }
         output_mask |= 1;
         continue;
      }
      const int location = var->data.location;
      const unsigned components =
         !composite || location == VARYING_SLOT_VAR2 ? 3 : 4;
      const unsigned bit = 1 + location - VARYING_SLOT_VAR0;
      if (location < VARYING_SLOT_VAR0 ||
          location > (composite ? VARYING_SLOT_VAR2 : VARYING_SLOT_VAR0) ||
          !pvrgpu_refract_float_vector_variable(var,
                                                nir_var_shader_out,
                                                location,
                                                components) ||
          var->data.precision != GLSL_PRECISION_MEDIUM ||
          (output_mask & BITFIELD_BIT(bit))) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "refract VS varying ABI mismatch");
      }
      output_mask |= BITFIELD_BIT(bit);
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_uniform) {
      ++uniforms;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if (var->data.location < 0 ||
             var->data.location >= (composite ? 4 : 1) ||
             !pvrgpu_refract_matrix_uniform(var, var->data.location) ||
             (uniform_mask & BITFIELD_BIT(var->data.location))) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "refract VS uniform ABI mismatch");
         }
         uniform_mask |= BITFIELD_BIT(var->data.location);
      } else {
         if (!composite || var->data.binding >= PVRGPU_REFRACT_TEXTURE_COUNT ||
             !pvrgpu_refract_sampler_uniform(var, var->data.binding) ||
             (uniform_mask & BITFIELD_BIT(var->data.binding))) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "refract FS sampler ABI mismatch");
         }
         uniform_mask |= BITFIELD_BIT(var->data.binding);
      }
   }

   const unsigned expected_varyings = composite ? 3 : 1;
   const unsigned expected_uniforms =
      nir->info.stage == MESA_SHADER_VERTEX ? (composite ? 4 : 1) :
                                              (composite ? 3 : 0);
   const unsigned expected_uniform_mask =
      expected_uniforms ? BITFIELD_MASK(expected_uniforms) : 0;
   if (inputs != (nir->info.stage == MESA_SHADER_VERTEX ? 2 :
                                                         expected_varyings) ||
       outputs != (nir->info.stage == MESA_SHADER_VERTEX ?
                      1 + expected_varyings : 1) ||
       uniforms != expected_uniforms || input_mask !=
          (nir->info.stage == MESA_SHADER_VERTEX ? 0x3 :
                                                  BITFIELD_MASK(expected_varyings)) ||
       output_mask != (nir->info.stage == MESA_SHADER_VERTEX ?
                          BITFIELD_MASK(1 + expected_varyings) : 1) ||
       uniform_mask != expected_uniform_mask) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract %s variable signature mismatch",
                             nir->info.stage == MESA_SHADER_VERTEX ? "VS" :
                                                                    "FS");
   }
   return true;
}

static bool
pvrgpu_validate_refract_uniform_load(const nir_intrinsic_instr *intr,
                                     bool composite,
                                     uint64_t *seen_slots,
                                     char *error,
                                     size_t error_size)
{
   if (intr->def.bit_size != 32 || intr->def.num_components != 4 ||
       nir_intrinsic_range(intr) != 4 ||
       nir_intrinsic_dest_type(intr) != nir_type_float32 ||
       !nir_src_is_const(intr->src[0])) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract VS load_uniform ABI mismatch");
   }
   const unsigned base = nir_intrinsic_base(intr);
   const unsigned row = nir_src_as_uint(intr->src[0]);
   const unsigned slot = base + row;
   if (row >= 4 || base % 4 != 0 || base > (composite ? 12 : 0) ||
       slot >= (composite ? 16 : 4) ||
       (*seen_slots & BITFIELD64_BIT(slot))) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract VS uniform row is invalid or duplicate");
   }
   *seen_slots |= BITFIELD64_BIT(slot);
   return true;
}

static bool
pvrgpu_validate_refract_nir(const nir_shader *nir,
                            enum pvrgpu_pco_refract_profile profile,
                            mesa_shader_stage expected_stage,
                            char *error,
                            size_t error_size)
{
   const bool composite = profile == PVRGPU_PCO_REFRACT_COMPOSITE;
   if (!nir || nir->info.stage != expected_stage ||
       !pvrgpu_validate_refract_variables(nir, profile, error, error_size))
      return false;

   struct pvrgpu_refract_nir_profile observed = { 0 };
   unsigned functions = 0;
   uint64_t seen_uniform_slots = 0;
   unsigned seen_texture_mask = 0;
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (!function->is_entrypoint || ++functions != 1) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "refract requires one NIR entrypoint");
      }
      nir_foreach_block (block, function->impl) {
         ++observed.blocks;
         nir_foreach_instr (instr, block) {
            ++observed.instructions;
            switch (instr->type) {
            case nir_instr_type_load_const:
               ++observed.constants;
               break;
            case nir_instr_type_deref:
               if (nir_instr_as_deref(instr)->deref_type !=
                   nir_deref_type_var) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "refract contains non-variable deref");
               }
               ++observed.derefs;
               break;
            case nir_instr_type_intrinsic: {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_uniform) {
                  if (expected_stage != MESA_SHADER_VERTEX ||
                      !pvrgpu_validate_refract_uniform_load(intr,
                                                            composite,
                                                            &seen_uniform_slots,
                                                            error,
                                                            error_size))
                     return false;
                  ++observed.uniform_loads;
               } else if (intr->intrinsic == nir_intrinsic_load_deref) {
                  ++observed.load_derefs;
               } else if (intr->intrinsic == nir_intrinsic_store_deref) {
                  ++observed.store_derefs;
               } else {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "refract contains unsupported NIR intrinsic %s",
                     nir_intrinsic_infos[intr->intrinsic].name);
               }
               break;
            }
            case nir_instr_type_alu: {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
#define PVRGPU_REFRACT_COUNT(opcode, field)                                \
   case nir_op_##opcode:                                                   \
      ++observed.field;                                                    \
      break
               switch (alu->op) {
               PVRGPU_REFRACT_COUNT(fadd, fadd);
               PVRGPU_REFRACT_COUNT(fmul, fmul);
               PVRGPU_REFRACT_COUNT(fneg, fneg);
               PVRGPU_REFRACT_COUNT(fdot3, fdot3);
               PVRGPU_REFRACT_COUNT(frsq, frsq);
               PVRGPU_REFRACT_COUNT(fsqrt, fsqrt);
               PVRGPU_REFRACT_COUNT(fdiv, fdiv);
               PVRGPU_REFRACT_COUNT(flt, flt);
               PVRGPU_REFRACT_COUNT(fmax, fmax);
               PVRGPU_REFRACT_COUNT(fpow, fpow);
               PVRGPU_REFRACT_COUNT(bcsel, bcsel);
               PVRGPU_REFRACT_COUNT(vec3, vec3);
               PVRGPU_REFRACT_COUNT(vec4, vec4);
               default:
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "refract contains unsupported NIR ALU op %s",
                                         nir_op_infos[alu->op].name);
               }
#undef PVRGPU_REFRACT_COUNT
               break;
            }
            case nir_instr_type_tex: {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               const unsigned slot = tex->texture_index;
               if (!composite || expected_stage != MESA_SHADER_FRAGMENT ||
                   slot >= PVRGPU_REFRACT_TEXTURE_COUNT ||
                   tex->sampler_index != slot ||
                   (seen_texture_mask & BITFIELD_BIT(slot)) ||
                   tex->op != nir_texop_tex ||
                   tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
                   tex->dest_type != nir_type_float32 ||
                   tex->def.num_components != 4 || tex->def.bit_size != 32 ||
                   tex->num_srcs != 1 || tex->coord_components != 2 ||
                   tex->src[0].src_type != nir_tex_src_coord || tex->is_array ||
                   tex->is_shadow || tex->is_sparse ||
                   tex->texture_non_uniform || tex->sampler_non_uniform ||
                   tex->embedded_sampler || tex->offset_non_uniform ||
                   tex->backend_flags != 0) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "refract texture graph changed");
               }
               seen_texture_mask |= BITFIELD_BIT(slot);
               ++observed.textures;
               break;
            }
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "refract contains unsupported NIR instruction type %u",
                                      instr->type);
            }
         }
      }
   }

   if (functions != 1 || observed.blocks == 0 || observed.blocks > 2)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract NIR control-flow signature mismatch");

   const bool vertex = expected_stage == MESA_SHADER_VERTEX;
   bool signature_matches = false;
   if (!composite && vertex) {
      signature_matches =
         observed.instructions == 23 && observed.constants == 4 &&
         observed.derefs == 4 && observed.load_derefs == 2 &&
         observed.store_derefs == 2 && observed.uniform_loads == 4 &&
         observed.fadd == 3 && observed.fmul == 3 && observed.vec3 == 1 &&
         seen_uniform_slots == UINT64_C(0x000f);
   } else if (!composite) {
      signature_matches =
         observed.instructions == 6 && observed.constants == 1 &&
         observed.derefs == 2 && observed.load_derefs == 1 &&
         observed.store_derefs == 1 && observed.vec4 == 1;
   } else if (vertex) {
      signature_matches =
         observed.instructions == 59 && observed.constants == 4 &&
         observed.derefs == 6 && observed.load_derefs == 2 &&
         observed.store_derefs == 4 && observed.uniform_loads == 16 &&
         observed.fadd == 12 && observed.fmul == 13 &&
         observed.fdot3 == 1 && observed.frsq == 1 &&
         seen_uniform_slots == UINT64_C(0xffff);
   } else {
      signature_matches =
         observed.instructions == 87 && observed.constants == 12 &&
         observed.derefs == 4 && observed.load_derefs == 3 &&
         observed.store_derefs == 1 && observed.fadd == 16 &&
         observed.fmul == 20 && observed.fneg == 10 &&
         observed.fdot3 == 7 && observed.frsq == 3 &&
         observed.fsqrt == 2 && observed.fdiv == 2 && observed.flt == 1 &&
         observed.fmax == 1 && observed.fpow == 1 && observed.bcsel == 1 &&
         observed.textures == 3 && seen_texture_mask == 0x7;
   }
   if (!signature_matches) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract %s canonical NIR signature mismatch",
                             vertex ? "VS" : "FS");
   }
   return true;
}

static bool
pvrgpu_shadow_sampler_uniform(const nir_variable *var)
{
   return var && var->data.mode == nir_var_uniform &&
          var->data.location == 2 && var->data.binding == 0 &&
          var->data.precision == GLSL_PRECISION_LOW &&
          glsl_type_is_sampler(var->type) &&
          glsl_get_sampler_dim(var->type) == GLSL_SAMPLER_DIM_2D &&
          !glsl_sampler_type_is_array(var->type) &&
          !glsl_sampler_type_is_shadow(var->type);
}

static bool
pvrgpu_validate_shadow_variables(const nir_shader *nir,
                                 enum pvrgpu_pco_shadow_profile profile,
                                 char *error,
                                 size_t error_size)
{
   const bool mask = profile == PVRGPU_PCO_SHADOW_MASK;
   const bool scene = profile == PVRGPU_PCO_SHADOW_SCENE;
   unsigned inputs = 0;
   unsigned outputs = 0;
   unsigned uniforms = 0;
   unsigned input_mask = 0;
   unsigned output_mask = 0;
   unsigned uniform_mask = 0;

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_in) {
      ++inputs;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         const unsigned expected_inputs = scene ? 2U : 1U;
         if (var->data.location < VERT_ATTRIB_GENERIC0 ||
             var->data.location >=
                VERT_ATTRIB_GENERIC0 + (int)expected_inputs ||
             !pvrgpu_refract_float_vector_variable(var,
                                                    nir_var_shader_in,
                                                    var->data.location,
                                                    4) ||
             var->data.precision != GLSL_PRECISION_HIGH) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "shadow VS input ABI mismatch");
         }
         const unsigned bit = var->data.location - VERT_ATTRIB_GENERIC0;
         if (input_mask & BITFIELD_BIT(bit))
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "shadow VS duplicate input");
         input_mask |= BITFIELD_BIT(bit);
      } else {
         const unsigned components = mask ? 4U : 1U;
         const bool type_matches =
            components == 1 ?
               (var->data.mode == nir_var_shader_in &&
                var->data.location == VARYING_SLOT_VAR0 &&
                glsl_type_is_scalar(var->type) &&
                glsl_get_base_type(var->type) == GLSL_TYPE_FLOAT &&
                glsl_get_bit_size(var->type) == 32) :
               pvrgpu_refract_float_vector_variable(var,
                                                     nir_var_shader_in,
                                                     VARYING_SLOT_VAR0,
                                                     components);
         if (!type_matches || var->data.location_frac != 0 ||
             var->data.interpolation != INTERP_MODE_SMOOTH ||
             var->data.precision != GLSL_PRECISION_MEDIUM) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "shadow FS varying ABI mismatch");
         }
         input_mask |= 1U;
      }
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
      ++outputs;
      if (nir->info.stage == MESA_SHADER_FRAGMENT) {
         if ((!pvrgpu_refract_float_vector_variable(var,
                                                    nir_var_shader_out,
                                                    FRAG_RESULT_COLOR,
                                                    4) &&
              !pvrgpu_refract_float_vector_variable(var,
                                                    nir_var_shader_out,
                                                    FRAG_RESULT_DATA0,
                                                    4)) ||
             var->data.precision != GLSL_PRECISION_MEDIUM) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "shadow FS color ABI mismatch");
         }
         output_mask |= 1U;
      } else if (var->data.location == VARYING_SLOT_POS) {
         if (!pvrgpu_refract_float_vector_variable(var,
                                                   nir_var_shader_out,
                                                   VARYING_SLOT_POS,
                                                   4) ||
             var->data.precision != GLSL_PRECISION_HIGH) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "shadow VS position ABI mismatch");
         }
         output_mask |= 1U;
      } else {
         const unsigned components = mask ? 4U : 1U;
         const bool type_matches =
            components == 1 ?
               (var->data.mode == nir_var_shader_out &&
                var->data.location == VARYING_SLOT_VAR0 &&
                glsl_type_is_scalar(var->type) &&
                glsl_get_base_type(var->type) == GLSL_TYPE_FLOAT &&
                glsl_get_bit_size(var->type) == 32) :
               pvrgpu_refract_float_vector_variable(var,
                                                     nir_var_shader_out,
                                                     VARYING_SLOT_VAR0,
                                                     components);
         if (!type_matches || var->data.location_frac != 0 ||
             var->data.precision != GLSL_PRECISION_MEDIUM) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "shadow VS varying ABI mismatch");
         }
         output_mask |= 2U;
      }
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_uniform) {
      ++uniforms;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if (var->data.location < 0 || var->data.location >= 2 ||
             !pvrgpu_refract_matrix_uniform(var, var->data.location) ||
             (uniform_mask & BITFIELD_BIT(var->data.location))) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "shadow VS uniform ABI mismatch");
         }
         uniform_mask |= BITFIELD_BIT(var->data.location);
      } else if (!mask || !pvrgpu_shadow_sampler_uniform(var) ||
                 uniform_mask) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "shadow FS sampler ABI mismatch");
      } else {
         uniform_mask = 1U;
      }
   }

   const unsigned expected_inputs =
      nir->info.stage == MESA_SHADER_VERTEX ? (scene ? 2U : 1U) : 1U;
   const unsigned expected_outputs =
      nir->info.stage == MESA_SHADER_VERTEX ? 2U : 1U;
   const unsigned expected_uniforms =
      nir->info.stage == MESA_SHADER_VERTEX ? 2U : (mask ? 1U : 0U);
   if (inputs != expected_inputs || outputs != expected_outputs ||
       uniforms != expected_uniforms ||
       input_mask != BITFIELD_MASK(expected_inputs) ||
       output_mask !=
          (nir->info.stage == MESA_SHADER_VERTEX ? 3U : 1U) ||
       uniform_mask != BITFIELD_MASK(expected_uniforms)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "shadow %s variable signature mismatch",
                             nir->info.stage == MESA_SHADER_VERTEX ? "VS" :
                                                                    "FS");
   }
   return true;
}

struct pvrgpu_shadow_nir_profile {
   unsigned instructions;
   unsigned constants;
   unsigned derefs;
   unsigned load_derefs;
   unsigned store_derefs;
   unsigned uniform_loads;
   unsigned fadd;
   unsigned fmul;
   unsigned fdiv;
   unsigned fdot2;
   unsigned fdot3;
   unsigned frsq;
   unsigned fmax;
   unsigned flt;
   unsigned iand;
   unsigned bcsel;
   unsigned mov;
   unsigned vec4;
   unsigned textures;
};

static bool
pvrgpu_validate_shadow_uniform_load(const nir_intrinsic_instr *intr,
                                    enum pvrgpu_pco_shadow_profile profile,
                                    uint64_t *seen_slots,
                                    char *error,
                                    size_t error_size)
{
   if (intr->def.bit_size != 32 || intr->def.num_components != 4 ||
       nir_intrinsic_range(intr) != 4 ||
       nir_intrinsic_dest_type(intr) != nir_type_float32 ||
       !nir_src_is_const(intr->src[0])) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "shadow VS load_uniform ABI mismatch");
   }
   const unsigned base = nir_intrinsic_base(intr);
   const unsigned row = nir_src_as_uint(intr->src[0]);
   const unsigned slot = base + row;
   const bool mask = profile == PVRGPU_PCO_SHADOW_MASK;
   if ((base != 0 && base != 4) || row >= 4 || slot >= 8 ||
       (mask && row == 2) || (*seen_slots & BITFIELD64_BIT(slot))) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "shadow VS uniform row is invalid or duplicate");
   }
   *seen_slots |= BITFIELD64_BIT(slot);
   return true;
}

static bool
pvrgpu_validate_shadow_nir(const nir_shader *nir,
                           enum pvrgpu_pco_shadow_profile profile,
                           mesa_shader_stage expected_stage,
                           char *error,
                           size_t error_size)
{
   if (!nir || nir->info.stage != expected_stage ||
       !pvrgpu_validate_shadow_variables(nir, profile, error, error_size))
      return false;

   struct pvrgpu_shadow_nir_profile observed = { 0 };
   unsigned functions = 0;
   unsigned blocks = 0;
   uint64_t seen_uniform_slots = 0;
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (!function->is_entrypoint || ++functions != 1)
         return pvrgpu_pco_fail(error,
                                error_size,
                                "shadow requires one NIR entrypoint");
      nir_foreach_block (block, function->impl) {
         ++blocks;
         nir_foreach_instr (instr, block) {
            ++observed.instructions;
            switch (instr->type) {
            case nir_instr_type_load_const:
               ++observed.constants;
               break;
            case nir_instr_type_deref:
               if (nir_instr_as_deref(instr)->deref_type !=
                   nir_deref_type_var)
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "shadow contains non-variable deref");
               ++observed.derefs;
               break;
            case nir_instr_type_intrinsic: {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_uniform) {
                  if (expected_stage != MESA_SHADER_VERTEX ||
                      !pvrgpu_validate_shadow_uniform_load(intr,
                                                           profile,
                                                           &seen_uniform_slots,
                                                           error,
                                                           error_size))
                     return false;
                  ++observed.uniform_loads;
               } else if (intr->intrinsic == nir_intrinsic_load_deref) {
                  ++observed.load_derefs;
               } else if (intr->intrinsic == nir_intrinsic_store_deref) {
                  ++observed.store_derefs;
               } else {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "shadow contains unsupported NIR intrinsic %s",
                     nir_intrinsic_infos[intr->intrinsic].name);
               }
               break;
            }
            case nir_instr_type_alu: {
               const nir_op op = nir_instr_as_alu(instr)->op;
#define PVRGPU_SHADOW_COUNT(opcode, field)                                 \
   case nir_op_##opcode:                                                   \
      ++observed.field;                                                    \
      break
               switch (op) {
               PVRGPU_SHADOW_COUNT(fadd, fadd);
               PVRGPU_SHADOW_COUNT(fmul, fmul);
               PVRGPU_SHADOW_COUNT(fdiv, fdiv);
               PVRGPU_SHADOW_COUNT(fdot2, fdot2);
               PVRGPU_SHADOW_COUNT(fdot3, fdot3);
               PVRGPU_SHADOW_COUNT(frsq, frsq);
               PVRGPU_SHADOW_COUNT(fmax, fmax);
               PVRGPU_SHADOW_COUNT(flt, flt);
               PVRGPU_SHADOW_COUNT(iand, iand);
               PVRGPU_SHADOW_COUNT(bcsel, bcsel);
               PVRGPU_SHADOW_COUNT(mov, mov);
               PVRGPU_SHADOW_COUNT(vec4, vec4);
               default:
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "shadow contains unsupported NIR ALU op %s",
                                         nir_op_infos[op].name);
               }
#undef PVRGPU_SHADOW_COUNT
               break;
            }
            case nir_instr_type_tex: {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (profile != PVRGPU_PCO_SHADOW_MASK ||
                   expected_stage != MESA_SHADER_FRAGMENT ||
                   observed.textures != 0 || tex->texture_index != 0 ||
                   tex->sampler_index != 0 || tex->op != nir_texop_tex ||
                   tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
                   tex->dest_type != nir_type_float32 ||
                   tex->def.num_components != 4 || tex->def.bit_size != 32 ||
                   tex->num_srcs != 1 || tex->coord_components != 2 ||
                   tex->src[0].src_type != nir_tex_src_coord || tex->is_array ||
                   tex->is_shadow || tex->is_sparse ||
                   tex->texture_non_uniform || tex->sampler_non_uniform ||
                   tex->embedded_sampler || tex->offset_non_uniform ||
                   tex->backend_flags != 0) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "shadow texture graph changed");
               }
               ++observed.textures;
               break;
            }
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "shadow contains unsupported NIR instruction type %u",
                                      instr->type);
            }
         }
      }
   }

   if (functions != 1 || blocks == 0 || blocks > 2)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "shadow NIR control-flow signature mismatch");

   const bool vertex = expected_stage == MESA_SHADER_VERTEX;
   bool signature_matches = false;
   if (profile == PVRGPU_PCO_SHADOW_MASK && vertex) {
      signature_matches =
         observed.instructions == 23 && observed.constants == 3 &&
         observed.derefs == 3 && observed.load_derefs == 1 &&
         observed.store_derefs == 2 && observed.uniform_loads == 6 &&
         observed.fadd == 4 && observed.fmul == 4 &&
         seen_uniform_slots == UINT64_C(0x00bb);
   } else if (profile == PVRGPU_PCO_SHADOW_MASK) {
      signature_matches =
         observed.instructions == 18 && observed.constants == 5 &&
         observed.derefs == 2 && observed.load_derefs == 1 &&
         observed.store_derefs == 1 && observed.fdiv == 1 &&
         observed.fadd == 1 && observed.mov == 1 && observed.flt == 2 &&
         observed.iand == 1 && observed.bcsel == 1 && observed.vec4 == 1 &&
         observed.textures == 1;
   } else if (vertex) {
      signature_matches =
         observed.instructions == 38 && observed.constants == 5 &&
         observed.derefs == 4 && observed.load_derefs == 2 &&
         observed.store_derefs == 2 && observed.uniform_loads == 8 &&
         observed.fadd == 6 && observed.fmul == 7 &&
         observed.fdot2 == 1 && observed.fdot3 == 1 &&
         observed.frsq == 1 && observed.fmax == 1 &&
         seen_uniform_slots == UINT64_C(0x00ff);
   } else {
      signature_matches =
         observed.instructions == 6 && observed.constants == 1 &&
         observed.derefs == 2 && observed.load_derefs == 1 &&
         observed.store_derefs == 1 && observed.vec4 == 1;
   }
   if (!signature_matches) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "shadow %s canonical NIR signature mismatch",
                             vertex ? "VS" : "FS");
   }
   return true;
}

struct pvrgpu_terrain_nir_signature {
   unsigned instructions;
   unsigned blocks;
   unsigned constants;
   unsigned derefs;
   unsigned load_derefs;
   unsigned store_derefs;
   unsigned uniform_loads;
   unsigned textures;
   unsigned b2f32;
   unsigned bcsel;
   unsigned fabs;
   unsigned fadd;
   unsigned fdiv;
   unsigned fdot3;
   unsigned fdot4;
   unsigned ffloor;
   unsigned fge;
   unsigned flrp;
   unsigned flt;
   unsigned fmax;
   unsigned fmin;
   unsigned fmod;
   unsigned fmul;
   unsigned fneg;
   unsigned fpow;
   unsigned frcp;
   unsigned frsq;
   unsigned fsqrt;
   unsigned mov;
   unsigned vec2;
   unsigned vec3;
   unsigned vec4;
};

static struct pvrgpu_terrain_nir_signature
pvrgpu_terrain_expected_signature(enum pvrgpu_pco_terrain_profile profile,
                                  mesa_shader_stage stage)
{
   const bool vertex = stage == MESA_SHADER_VERTEX;
   if (vertex && profile == PVRGPU_PCO_TERRAIN_D3) {
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 97,
         .blocks = 1,
         .constants = 6,
         .derefs = 9,
         .load_derefs = 4,
         .store_derefs = 5,
         .uniform_loads = 14,
         .textures = 2,
         .fadd = 19,
         .fdot3 = 3,
         .fmul = 24,
         .fneg = 2,
         .frsq = 3,
         .mov = 1,
         .vec2 = 1,
         .vec4 = 4,
      };
   }
   if (vertex && profile == PVRGPU_PCO_TERRAIN_D1) {
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 19,
         .blocks = 1,
         .constants = 3,
         .derefs = 3,
         .load_derefs = 1,
         .store_derefs = 2,
         .uniform_loads = 2,
         .fadd = 3,
         .fmul = 2,
         .fneg = 1,
         .vec2 = 1,
         .vec4 = 1,
      };
   }
   if (vertex) {
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 16,
         .blocks = 1,
         .constants = 3,
         .derefs = 3,
         .load_derefs = 1,
         .store_derefs = 2,
         .uniform_loads = 2,
         .fadd = 2,
         .fmul = 2,
         .vec4 = 1,
      };
   }

   switch (profile) {
   case PVRGPU_PCO_TERRAIN_D1:
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 544,
         .blocks = 1,
         .constants = 23,
         .derefs = 2,
         .load_derefs = 1,
         .store_derefs = 1,
         .uniform_loads = 1,
         .b2f32 = 28,
         .fabs = 12,
         .fadd = 119,
         .fdot3 = 56,
         .fdot4 = 4,
         .ffloor = 24,
         .fge = 28,
         .fmax = 8,
         .fmin = 4,
         .fmod = 16,
         .fmul = 102,
         .fneg = 49,
         .vec3 = 21,
         .vec4 = 45,
      };
   case PVRGPU_PCO_TERRAIN_D2:
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 29,
         .blocks = 1,
         .constants = 3,
         .derefs = 2,
         .load_derefs = 1,
         .store_derefs = 1,
         .uniform_loads = 2,
         .textures = 3,
         .fadd = 5,
         .fdot3 = 1,
         .fmul = 2,
         .fneg = 2,
         .frcp = 2,
         .frsq = 1,
         .vec2 = 2,
         .vec3 = 1,
         .vec4 = 1,
      };
   case PVRGPU_PCO_TERRAIN_D3:
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 105,
         .blocks = 1,
         .constants = 8,
         .derefs = 5,
         .load_derefs = 4,
         .store_derefs = 1,
         .uniform_loads = 16,
         .textures = 5,
         .bcsel = 1,
         .fadd = 13,
         .fdiv = 1,
         .fdot3 = 7,
         .flrp = 1,
         .flt = 1,
         .fmax = 2,
         .fmin = 1,
         .fmul = 24,
         .fneg = 2,
         .fpow = 1,
         .frsq = 5,
         .fsqrt = 1,
         .vec2 = 1,
         .vec3 = 3,
         .vec4 = 2,
      };
   case PVRGPU_PCO_TERRAIN_D4:
   case PVRGPU_PCO_TERRAIN_D5:
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 40,
         .blocks = 1,
         .constants = 7,
         .derefs = 2,
         .load_derefs = 1,
         .store_derefs = 1,
         .textures = 5,
         .fabs = 1,
         .fadd = 8,
         .fdiv = 1,
         .fmul = 7,
         .fneg = 2,
         .vec2 = 4,
         .vec4 = 1,
      };
   case PVRGPU_PCO_TERRAIN_D6:
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 8,
         .blocks = 1,
         .constants = 1,
         .derefs = 2,
         .load_derefs = 1,
         .store_derefs = 1,
         .uniform_loads = 1,
         .textures = 1,
         .fmul = 1,
      };
   case PVRGPU_PCO_TERRAIN_D7:
   case PVRGPU_PCO_TERRAIN_D8:
      return (struct pvrgpu_terrain_nir_signature){
         .instructions = 71,
         .blocks = 1,
         .constants = 12,
         .derefs = 2,
         .load_derefs = 1,
         .store_derefs = 1,
         .textures = 9,
         .fabs = 1,
         .fadd = 17,
         .fdiv = 1,
         .fmul = 13,
         .fneg = 5,
         .vec2 = 8,
         .vec4 = 1,
      };
   }
   return (struct pvrgpu_terrain_nir_signature){ 0 };
}

static bool
pvrgpu_validate_terrain_variables(const nir_shader *nir,
                                  enum pvrgpu_pco_terrain_profile profile,
                                  char *error,
                                  size_t error_size)
{
   const bool vertex = nir->info.stage == MESA_SHADER_VERTEX;
   const bool main = profile == PVRGPU_PCO_TERRAIN_D3;
   const unsigned expected_inputs = vertex ? (main ? 4U : 1U) :
                                            (main ? 4U : 1U);
   const unsigned expected_outputs = vertex ? (main ? 5U : 2U) : 1U;
   const unsigned expected_uniforms = vertex ? (main ? 7U : 2U) :
      profile == PVRGPU_PCO_TERRAIN_D1 ? 1U :
      profile == PVRGPU_PCO_TERRAIN_D2 ? 3U :
      main ? 18U :
      profile == PVRGPU_PCO_TERRAIN_D6 ? 2U : 1U;
   unsigned inputs = 0;
   unsigned outputs = 0;
   unsigned uniforms = 0;
   unsigned sampler_mask = 0;

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_in) {
      const unsigned index = inputs++;
      const int expected_location =
         vertex ? VERT_ATTRIB_GENERIC0 + (int)index :
                  VARYING_SLOT_VAR0 + (int)index;
      const unsigned expected_components =
         vertex ? 4U : (main && index == 3U ? 2U : (main ? 4U : 2U));
      if (index >= expected_inputs ||
          var->data.location != expected_location ||
          var->data.location_frac != 0 ||
          glsl_get_base_type(var->type) != GLSL_TYPE_FLOAT ||
          glsl_get_bit_size(var->type) != 32 ||
          glsl_get_components(var->type) != expected_components ||
          var->data.precision !=
             (vertex ? GLSL_PRECISION_HIGH : GLSL_PRECISION_MEDIUM) ||
          (!vertex && var->data.interpolation != INTERP_MODE_SMOOTH)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain %s input ABI mismatch",
                                vertex ? "VS" : "FS");
      }
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
      ++outputs;
      if (!vertex) {
         if ((var->data.location != FRAG_RESULT_COLOR &&
              var->data.location != FRAG_RESULT_DATA0) ||
             glsl_get_components(var->type) != 4 ||
             var->data.precision != GLSL_PRECISION_MEDIUM) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "terrain FS output ABI mismatch");
         }
         continue;
      }
      if (var->data.location == VARYING_SLOT_POS) {
         if (glsl_get_components(var->type) != 4 ||
             var->data.precision != GLSL_PRECISION_HIGH) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "terrain VS position ABI mismatch");
         }
         continue;
      }
      const unsigned varying = var->data.location - VARYING_SLOT_VAR0;
      const unsigned expected_components =
         main && varying == 3U ? 2U : (main ? 4U : 2U);
      if (var->data.location < VARYING_SLOT_VAR0 ||
          varying >= (main ? 4U : 1U) || var->data.location_frac != 0 ||
          glsl_get_components(var->type) != expected_components ||
          var->data.precision != GLSL_PRECISION_MEDIUM) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain VS varying ABI mismatch");
      }
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_uniform) {
      ++uniforms;
      if (!glsl_type_is_sampler(var->type))
         continue;
      if (var->data.precision != GLSL_PRECISION_LOW ||
          glsl_get_sampler_dim(var->type) != GLSL_SAMPLER_DIM_2D ||
          glsl_sampler_type_is_array(var->type) ||
          glsl_sampler_type_is_shadow(var->type) || var->data.binding < 0 ||
          var->data.binding >= (int)(vertex ?
             pvrgpu_terrain_profiles[profile].vertex_texture_count :
             pvrgpu_terrain_profiles[profile].fragment_texture_count) ||
          (sampler_mask & BITFIELD_BIT(var->data.binding))) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain sampler variable ABI mismatch");
      }
      sampler_mask |= BITFIELD_BIT(var->data.binding);
   }

   const unsigned expected_samplers =
      vertex ? pvrgpu_terrain_profiles[profile].vertex_texture_count :
               pvrgpu_terrain_profiles[profile].fragment_texture_count;
   if (inputs != expected_inputs || outputs != expected_outputs ||
       uniforms != expected_uniforms ||
       sampler_mask != BITFIELD_MASK(expected_samplers)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain %s variable signature mismatch",
                             vertex ? "VS" : "FS");
   }
   return true;
}

static bool
pvrgpu_validate_terrain_nir(const nir_shader *nir,
                            enum pvrgpu_pco_terrain_profile profile,
                            mesa_shader_stage expected_stage,
                            char *error,
                            size_t error_size)
{
   if (!nir || nir->info.stage != expected_stage ||
       !pvrgpu_validate_terrain_variables(nir,
                                          profile,
                                          error,
                                          error_size))
      return false;

   struct pvrgpu_terrain_nir_signature observed = { 0 };
   unsigned functions = 0;
   unsigned texture_order[9] = { 0 };
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (!function->is_entrypoint || ++functions != 1)
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain requires one NIR entrypoint");
      nir_foreach_block (block, function->impl) {
         ++observed.blocks;
         nir_foreach_instr (instr, block) {
            ++observed.instructions;
            switch (instr->type) {
            case nir_instr_type_load_const:
               ++observed.constants;
               break;
            case nir_instr_type_deref:
               if (nir_instr_as_deref(instr)->deref_type !=
                   nir_deref_type_var) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain contains non-variable deref");
               }
               ++observed.derefs;
               break;
            case nir_instr_type_intrinsic: {
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               if (op == nir_intrinsic_load_uniform)
                  ++observed.uniform_loads;
               else if (op == nir_intrinsic_load_deref)
                  ++observed.load_derefs;
               else if (op == nir_intrinsic_store_deref)
                  ++observed.store_derefs;
               else
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain contains unsupported NIR intrinsic %s",
                     nir_intrinsic_infos[op].name);
               break;
            }
            case nir_instr_type_alu: {
               const nir_op op = nir_instr_as_alu(instr)->op;
#define PVRGPU_TERRAIN_COUNT(opcode, field)                               \
   case nir_op_##opcode:                                                  \
      ++observed.field;                                                   \
      break
               switch (op) {
               PVRGPU_TERRAIN_COUNT(b2f32, b2f32);
               PVRGPU_TERRAIN_COUNT(bcsel, bcsel);
               PVRGPU_TERRAIN_COUNT(fabs, fabs);
               PVRGPU_TERRAIN_COUNT(fadd, fadd);
               PVRGPU_TERRAIN_COUNT(fdiv, fdiv);
               PVRGPU_TERRAIN_COUNT(fdot3, fdot3);
               PVRGPU_TERRAIN_COUNT(fdot4, fdot4);
               PVRGPU_TERRAIN_COUNT(ffloor, ffloor);
               PVRGPU_TERRAIN_COUNT(fge, fge);
               PVRGPU_TERRAIN_COUNT(flrp, flrp);
               PVRGPU_TERRAIN_COUNT(flt, flt);
               PVRGPU_TERRAIN_COUNT(fmax, fmax);
               PVRGPU_TERRAIN_COUNT(fmin, fmin);
               PVRGPU_TERRAIN_COUNT(fmod, fmod);
               PVRGPU_TERRAIN_COUNT(fmul, fmul);
               PVRGPU_TERRAIN_COUNT(fneg, fneg);
               PVRGPU_TERRAIN_COUNT(fpow, fpow);
               PVRGPU_TERRAIN_COUNT(frcp, frcp);
               PVRGPU_TERRAIN_COUNT(frsq, frsq);
               PVRGPU_TERRAIN_COUNT(fsqrt, fsqrt);
               PVRGPU_TERRAIN_COUNT(mov, mov);
               PVRGPU_TERRAIN_COUNT(vec2, vec2);
               PVRGPU_TERRAIN_COUNT(vec3, vec3);
               PVRGPU_TERRAIN_COUNT(vec4, vec4);
               default:
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain contains unsupported NIR ALU op %s",
                                         nir_op_infos[op].name);
               }
#undef PVRGPU_TERRAIN_COUNT
               break;
            }
            case nir_instr_type_tex: {
               const nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (observed.textures >= ARRAY_SIZE(texture_order) ||
                   tex->texture_index != tex->sampler_index ||
                   tex->op != nir_texop_tex ||
                   tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
                   tex->dest_type != nir_type_float32 ||
                   tex->def.num_components != 4 || tex->def.bit_size != 32 ||
                   tex->num_srcs != 1 || tex->coord_components != 2 ||
                   tex->src[0].src_type != nir_tex_src_coord || tex->is_array ||
                   tex->is_shadow || tex->is_sparse ||
                   tex->texture_non_uniform || tex->sampler_non_uniform ||
                   tex->embedded_sampler || tex->offset_non_uniform ||
                   tex->backend_flags != 0) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain texture graph changed");
               }
               texture_order[observed.textures++] = tex->texture_index;
               break;
            }
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain contains unsupported NIR instruction type %u",
                                      instr->type);
            }
         }
      }
   }

   const struct pvrgpu_terrain_nir_signature expected =
      pvrgpu_terrain_expected_signature(profile, expected_stage);
   if (functions != 1 || memcmp(&observed, &expected, sizeof(expected)) != 0) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain %s canonical NIR signature mismatch "
                             "(instr=%u blocks=%u uniform=%u tex=%u)",
                             expected_stage == MESA_SHADER_VERTEX ? "VS" : "FS",
                             observed.instructions,
                             observed.blocks,
                             observed.uniform_loads,
                             observed.textures);
   }

   static const unsigned main_vs_order[2] = { 1U, 0U };
   static const unsigned main_fs_order[5] = { 2U, 0U, 1U, 4U, 3U };
   const unsigned *expected_order = NULL;
   unsigned expected_texture_ops = 0;
   if (expected_stage == MESA_SHADER_VERTEX &&
       profile == PVRGPU_PCO_TERRAIN_D3) {
      expected_order = main_vs_order;
      expected_texture_ops = ARRAY_SIZE(main_vs_order);
   } else if (expected_stage == MESA_SHADER_FRAGMENT &&
              profile == PVRGPU_PCO_TERRAIN_D3) {
      expected_order = main_fs_order;
      expected_texture_ops = ARRAY_SIZE(main_fs_order);
   } else if (expected_stage == MESA_SHADER_FRAGMENT) {
      expected_texture_ops =
         pvrgpu_terrain_profiles[profile].fragment_texture_ops;
   }
   for (unsigned texture = 0; texture < expected_texture_ops; ++texture) {
      const unsigned expected_index = expected_order ? expected_order[texture] : 0U;
      if (texture_order[texture] != expected_index) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain %s texture order changed at %u",
                                expected_stage == MESA_SHADER_VERTEX ? "VS" : "FS",
                                texture);
      }
   }
   return true;
}

static bool pvrgpu_conditionals_allowed_alu(nir_op op)
{
   switch (op) {
   case nir_op_mov:
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_ffract:
   case nir_op_fge:
   case nir_op_bcsel:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
      return true;
   default:
      return false;
   }
}

static bool pvrgpu_conditionals_allowed_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_deref:
   case nir_intrinsic_store_deref:
   case nir_intrinsic_load_input:
   case nir_intrinsic_store_output:
   case nir_intrinsic_load_frag_coord:
      return true;
   default:
      return false;
   }
}

static bool pvrgpu_validate_conditionals_variables(const nir_shader *nir,
                                                   char *error,
                                                   size_t error_size)
{
   unsigned input_count = 0;
   unsigned output_count = 0;

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_in) {
      ++input_count;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if (var->data.location != VERT_ATTRIB_GENERIC0 ||
             glsl_get_components(var->type) != 4) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "conditionals VS requires one vec4 "
                                   "GENERIC0 input");
         }
      } else if (var->data.location != VARYING_SLOT_POS ||
                 glsl_get_components(var->type) != 4) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "conditionals FS requires only vec4 "
                                "VARYING_SLOT_POS");
      }
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
      ++output_count;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if (var->data.location != VARYING_SLOT_POS ||
             glsl_get_components(var->type) != 4) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "conditionals VS requires only vec4 "
                                   "gl_Position output");
         }
      } else if ((var->data.location != FRAG_RESULT_COLOR &&
                  var->data.location != FRAG_RESULT_DATA0) ||
                 glsl_get_components(var->type) != 4) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "conditionals FS requires one vec4 color "
                                "output");
      }
   }

   if (input_count != 1 || output_count != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals %s I/O count mismatch (%u input, "
                             "%u output)",
                             nir->info.stage == MESA_SHADER_VERTEX ? "VS"
                                                                   : "FS",
                             input_count,
                             output_count);
   }

   return true;
}

static bool
pvrgpu_validate_conditionals_uniform(const nir_intrinsic_instr *intr,
                                     mesa_shader_stage stage,
                                     unsigned *seen_offsets,
                                     char *error,
                                     size_t error_size)
{
   const unsigned slot_count = stage == MESA_SHADER_VERTEX ? 4U : 1U;
   if (intr->def.bit_size != 32 || intr->def.num_components != 4 ||
       nir_intrinsic_base(intr) != 0 ||
       nir_intrinsic_range(intr) != slot_count ||
       nir_intrinsic_dest_type(intr) != nir_type_float32 ||
       !nir_src_is_const(intr->src[0])) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals %s load_uniform ABI mismatch",
                             stage == MESA_SHADER_VERTEX ? "VS" : "FS");
   }

   const unsigned slot = nir_src_as_uint(intr->src[0]);
   if (slot >= slot_count || (*seen_offsets & (1U << slot))) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals %s load_uniform slot %u is invalid "
                             "or duplicated",
                             stage == MESA_SHADER_VERTEX ? "VS" : "FS",
                             slot);
   }

   *seen_offsets |= 1U << slot;
   return true;
}

static bool pvrgpu_validate_conditionals_nir(const nir_shader *nir,
                                             mesa_shader_stage expected_stage,
                                             char *error,
                                             size_t error_size)
{
   if (!nir || nir->info.stage != expected_stage) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals shader stage mismatch");
   }
   if (expected_stage != MESA_SHADER_VERTEX &&
       expected_stage != MESA_SHADER_FRAGMENT) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals profile supports VS and FS only");
   }
   if (!pvrgpu_validate_conditionals_variables(nir, error, error_size))
      return false;

   struct pvrgpu_conditionals_profile profile = { 0 };
   unsigned seen_uniform_offsets = 0;
   unsigned implemented_functions = 0;

   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (!function->is_entrypoint || ++implemented_functions != 1) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "conditionals profile requires one entrypoint");
      }

      nir_foreach_block (block, function->impl) {
         ++profile.blocks;
         nir_foreach_instr (instr, block) {
            switch (instr->type) {
            case nir_instr_type_alu: {
               const nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (!pvrgpu_conditionals_allowed_alu(alu->op)) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "conditionals contains unsupported "
                                         "NIR ALU op %s",
                                         nir_op_infos[alu->op].name);
               }
               profile.fract_ops += alu->op == nir_op_ffract;
               profile.compare_ops += alu->op == nir_op_fge;
               profile.select_ops += alu->op == nir_op_bcsel;
               break;
            }
            case nir_instr_type_intrinsic: {
               const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (!pvrgpu_conditionals_allowed_intrinsic(intr->intrinsic)) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "conditionals contains unsupported NIR intrinsic %s",
                     nir_intrinsic_infos[intr->intrinsic].name);
               }
               if (intr->intrinsic == nir_intrinsic_load_uniform) {
                  if (!pvrgpu_validate_conditionals_uniform(
                         intr,
                         expected_stage,
                         &seen_uniform_offsets,
                         error,
                         error_size))
                     return false;
                  ++profile.uniform_loads;
               }
               break;
            }
            case nir_instr_type_load_const:
            case nir_instr_type_deref:
               break;
            case nir_instr_type_jump:
               if (nir_instr_as_jump(instr)->type != nir_jump_return) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "conditionals contains unsupported control flow");
               }
               break;
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "conditionals contains unsupported NIR "
                                      "instruction type %u",
                                      instr->type);
            }
         }
      }
   }

   const unsigned uniform_loads = expected_stage == MESA_SHADER_VERTEX ? 4U
                                                                       : 1U;
   const unsigned uniform_mask = expected_stage == MESA_SHADER_VERTEX ? 0x0fU
                                                                      : 0x01U;
   if (implemented_functions != 1 || profile.blocks == 0 ||
       profile.blocks > 2 || profile.uniform_loads != uniform_loads ||
       seen_uniform_offsets != uniform_mask || profile.fract_ops != 3 ||
       profile.compare_ops != 1 || profile.select_ops != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals %s NIR signature mismatch "
                             "(blocks=%u uniform=%u fract=%u fge=%u bcsel=%u)",
                             expected_stage == MESA_SHADER_VERTEX ? "VS" : "FS",
                             profile.blocks,
                             profile.uniform_loads,
                             profile.fract_ops,
                             profile.compare_ops,
                             profile.select_ops);
   }

   return true;
}

static bool
pvrgpu_lower_uniform_slots_to_push_constants(nir_shader *nir,
                                             unsigned expected_dwords,
                                             unsigned expected_loads,
                                             const char *profile_name,
                                             char *error,
                                             size_t error_size)
{
   unsigned rewritten = 0;

   nir_foreach_function_impl(impl, nir)
   {
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_uniform)
               continue;

            if (!nir_src_is_const(intr->src[0])) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "dynamic %s load_uniform is "
                                      "unsupported",
                                      profile_name);
            }

            const unsigned vec4_slot =
               nir_intrinsic_base(intr) + nir_src_as_uint(intr->src[0]);
            if (vec4_slot > (UINT32_MAX / 16U)) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s uniform byte offset "
                                      "overflow",
                                      profile_name);
            }

            /*
             * Gallium's non-packed load_uniform offset is a vec4 slot.  NIR
             * load_push_constant is byte-addressed; pco_nir_lower_io() later
             * shifts it right by two to obtain a shared-register DWORD index.
             */
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_src_rewrite(&intr->src[0], nir_imm_int(&b, vec4_slot * 16U));
            intr->intrinsic = nir_intrinsic_load_push_constant;
            nir_intrinsic_set_base(intr, 0);
            nir_intrinsic_set_range(intr, expected_dwords * 4U);
            nir_intrinsic_set_align_mul(intr, 16);
            nir_intrinsic_set_align_offset(intr, 0);
            ++rewritten;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   if (rewritten != expected_loads) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s %s rewrote %u uniform loads, "
                             "expected %u",
                             profile_name,
                             nir->info.stage == MESA_SHADER_VERTEX ? "VS"
                                                                   : "FS",
                             rewritten,
                             expected_loads);
   }
   return true;
}

static bool pvrgpu_canonicalize_fragment_output(nir_shader *nir,
                                                const char *profile_name,
                                                char *error,
                                                size_t error_size)
{
   unsigned stores = 0;

   nir_foreach_function_impl(impl, nir)
   {
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;

            nir_variable *var = nir_intrinsic_get_var(intr, 0);
            if (!var || var->data.mode != nir_var_shader_out ||
                var->data.location != FRAG_RESULT_DATA0 ||
                intr->src[1].ssa->num_components != 4 ||
                intr->src[1].ssa->bit_size != 32 ||
                nir_intrinsic_write_mask(intr) != 0x0f || stores++) {
               return pvrgpu_pco_fail(
                  error,
                  error_size,
                  "%s FS color-store ABI mismatch",
                  profile_name);
            }

            nir_builder b = nir_builder_at(nir_before_instr(instr));
            for (unsigned component = 0; component < 4; ++component) {
               nir_frag_store_pco(&b,
                                  nir_channel(&b, intr->src[1].ssa, component),
                                  .base = component);
            }

            /* Keep the now-dead variable valid until normal NIR DCE removes it.
             */
            var->data.mode = nir_var_shader_temp;
            var->data.location = 0;
            nir_instr_remove(&intr->instr);
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   if (stores != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s FS requires one RGBA color store",
                             profile_name);
   }

   nir->info.outputs_written &= ~BITFIELD64_BIT(FRAG_RESULT_DATA0);
   return true;
}

/*
 * The captured GLES fragment shader declares default mediump float, while its
 * FragCoord temporary is explicitly highp.  Therefore each assignment to the
 * mediump local `d` rounds to IEEE binary16 and is read back as float32:
 *
 *   d = fract(highp_expression);
 *   d = condition ? fract(2.0 * d) : fract(3.0 * d);
 *
 * Gallium gives this driver a legal full-float32 NIR because the generic
 * pvrgpu screen does not advertise 16-bit ALU.  PCO also does not advertise
 * native 16-bit ALU yet, so emulate every f16 ALU result with an explicit
 * round-to-nearest-even f2f16/f2f32 pair around otherwise equivalent f32
 * operations.  Mesa PCO translates those conversions to its public F16
 * pack/unpack instructions; this is compiler lowering, not a framebuffer
 * dither or a SystemC shader-name shortcut.
 */
static bool
pvrgpu_lower_conditionals_fragment_mediump(nir_shader *nir,
                                           char *error,
                                           size_t error_size)
{
   nir_alu_instr *phase = NULL;
   nir_alu_instr *selected = NULL;
   nir_function_impl *entrypoint = NULL;

   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (entrypoint) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "conditionals mediump lowering requires one "
                                "entrypoint");
      }
      entrypoint = function->impl;

      nir_foreach_block (block, function->impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_fge) {
               if (phase || alu->def.num_components != 1 ||
                   alu->src[0].src.ssa->bit_size != 32) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "conditionals mediump phase comparison changed");
               }
               nir_instr *producer = nir_def_instr(alu->src[0].src.ssa);
               if (!producer || producer->type != nir_instr_type_alu ||
                   nir_instr_as_alu(producer)->op != nir_op_ffract) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "conditionals mediump phase is not ffract");
               }
               phase = nir_instr_as_alu(producer);
            } else if (alu->op == nir_op_bcsel) {
               if (selected || alu->def.num_components != 1 ||
                   alu->def.bit_size != 32) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "conditionals mediump select changed");
               }
               selected = alu;
            }
         }
      }
   }

   if (!entrypoint || !phase || !selected) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals mediump assignment graph missing");
   }

   nir_builder b = nir_builder_at(nir_after_instr(&selected->instr));
#define PVRGPU_QUANTIZE_MEDIUMP(value)                                      \
   nir_f2f32(&b, nir_f2f16_rtne(&b, (value)))
   nir_def *phase_mediump = PVRGPU_QUANTIZE_MEDIUMP(&phase->def);
   nir_def *condition_mediump =
      nir_fge(&b, phase_mediump, nir_imm_float(&b, 0.5f));
   nir_def *twice_product = PVRGPU_QUANTIZE_MEDIUMP(
      nir_fmul_imm(&b, phase_mediump, 2.0f));
   nir_def *twice_mediump =
      PVRGPU_QUANTIZE_MEDIUMP(nir_ffract(&b, twice_product));
   nir_def *thrice_product = PVRGPU_QUANTIZE_MEDIUMP(
      nir_fmul_imm(&b, phase_mediump, 3.0f));
   nir_def *thrice_mediump =
      PVRGPU_QUANTIZE_MEDIUMP(nir_ffract(&b, thrice_product));
   /* bcsel preserves one already-quantized operand bit-for-bit. */
   nir_def *selected_mediump =
      nir_bcsel(&b, condition_mediump, twice_mediump, thrice_mediump);
#undef PVRGPU_QUANTIZE_MEDIUMP
   nir_def_rewrite_uses_after(&selected->def, selected_mediump);

   nir_progress(true, entrypoint, nir_metadata_control_flow);
   return true;
}

static void pvrgpu_init_conditionals_shader_data(pco_data *vertex_data,
                                                 pco_data *fragment_data,
                                                 enum pipe_format vertex_format)
{
   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC0] = vertex_format;
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC0] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->common.vtxins = 4;
   vertex_data->vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->vs.vtxouts = 4;

   /* The SystemC PBE consumes four float PIXOUT registers and owns RGBA8 pack.
    */
   fragment_data->fs.z_replicate = ~0U;
   fragment_data->fs.rasterization_samples = 1;
}

static bool
pvrgpu_allocate_push_constants(pco_data *data,
                               unsigned expected_dwords,
                               unsigned expected_used_dwords,
                               const char *profile_name,
                               const char *stage,
                               char *error,
                               size_t error_size)
{
   if (data->common.push_consts.used != expected_used_dwords ||
       data->common.shareds != 0) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "%s %s push ABI mismatch "
         "(used=%u shared=%u expected-used=%u expected-range=%u)",
         profile_name,
         stage,
         data->common.push_consts.used,
         data->common.shareds,
         expected_used_dwords,
         expected_dwords);
   }

   /* Preserve the complete Gallium CB0 ABI, including unused FS z/w words. */
   data->common.push_consts.range = (pco_range){
      .start = 0,
      .count = expected_dwords,
   };
   data->common.shareds = expected_dwords;
   return true;
}

static bool
pvrgpu_allocate_prefixed_push_constants(pco_data *data,
                                         unsigned descriptor_dwords,
                                         unsigned expected_dwords,
                                         unsigned expected_used_dwords,
                                         const char *profile_name,
                                         const char *stage,
                                         char *error,
                                         size_t error_size)
{
   if (data->common.push_consts.used != expected_used_dwords ||
       data->common.shareds != descriptor_dwords) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "%s %s descriptor/push ABI mismatch "
         "(used=%u shared=%u expected-used=%u descriptor=%u range=%u)",
         profile_name,
         stage,
         data->common.push_consts.used,
         data->common.shareds,
         expected_used_dwords,
         descriptor_dwords,
         expected_dwords);
   }

   data->common.push_consts.range = (pco_range){
      .start = descriptor_dwords,
      .count = expected_dwords,
   };
   data->common.shareds = descriptor_dwords + expected_dwords;
   return true;
}

static bool pvrgpu_copy_pco_stage(pco_shader *shader,
                                  bool vertex_stage,
                                  struct pvrgpu_pco_owned_binary *out,
                                  char *error,
                                  size_t error_size)
{
   const unsigned binary_size = pco_shader_binary_size(shader);
   const void *binary_data = pco_shader_binary_data(shader);
   if (!binary_size || !binary_data) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO produced an empty shader binary");
   }

   out->data = malloc(binary_size);
   if (!out->data)
      return pvrgpu_pco_fail(error, error_size, "out of memory copying PCO");
   memcpy(out->data, binary_data, binary_size);
   out->size = binary_size;

   const pco_data *data = pco_shader_data(shader);
   out->abi.temps = data->common.temps;
   out->abi.vertex_inputs = data->common.vtxins;
   out->abi.vertex_outputs = vertex_stage ? data->vs.vtxouts : 0;
   out->abi.coefficients = data->common.coeffs;
   out->abi.shareds = data->common.shareds;
   out->abi.push_constant_start = data->common.push_consts.range.start;
   out->abi.push_constant_count = data->common.push_consts.range.count;
   out->abi.entry_offset = data->common.entry_offset;
   return true;
}

struct pvrgpu_pco_compiler *pvrgpu_pco_compiler_create(char *error,
                                                       size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';

   struct pvrgpu_pco_compiler *compiler = calloc(1, sizeof(*compiler));
   if (!compiler) {
      pvrgpu_pco_fail(error, error_size, "out of memory creating PCO compiler");
      return NULL;
   }

   compiler->mem_ctx = ralloc_context(NULL);
   if (!compiler->mem_ctx) {
      pvrgpu_pco_fail(error, error_size, "out of memory creating PCO ralloc");
      free(compiler);
      return NULL;
   }

   if (!pvr_device_info_init_public_name(&compiler->dev_info,
                                         PVRGPU_PCO_PUBLIC_TARGET)) {
      pvrgpu_pco_fail(error,
                      error_size,
                      "Mesa does not recognize public PCO target %s",
                      PVRGPU_PCO_PUBLIC_TARGET);
      ralloc_free(compiler->mem_ctx);
      free(compiler);
      return NULL;
   }

   compiler->pco = pco_ctx_create(&compiler->dev_info,
                                  &compiler->runtime_info,
                                  compiler->mem_ctx);
   if (!compiler->pco) {
      pvrgpu_pco_fail(error, error_size, "failed to create gx6250 PCO context");
      ralloc_free(compiler->mem_ctx);
      free(compiler);
      return NULL;
   }

   return compiler;
}

void pvrgpu_pco_compiler_destroy(struct pvrgpu_pco_compiler *compiler)
{
   if (!compiler)
      return;
   ralloc_free(compiler->mem_ctx);
   free(compiler);
}

void pvrgpu_pco_graphics_binary_finish(struct pvrgpu_pco_graphics_binary *binary)
{
   if (!binary)
      return;
   free(binary->vertex.data);
   free(binary->fragment.data);
   memset(binary, 0, sizeof(*binary));
}

bool pvrgpu_pco_compile_conditionals(struct pvrgpu_pco_compiler *compiler,
                                     const nir_shader *vertex_nir,
                                     const nir_shader *fragment_nir,
                                     enum pipe_format vertex_format,
                                     struct pvrgpu_pco_graphics_binary *out,
                                     char *error,
                                     size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));

   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "missing compiler or conditionals NIR stage");
   }
   if (vertex_format != PIPE_FORMAT_R32G32B32_FLOAT) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "conditionals requires R32G32B32_FLOAT GENERIC0");
   }
   if (!pvrgpu_validate_conditionals_nir(vertex_nir,
                                         MESA_SHADER_VERTEX,
                                         error,
                                         error_size) ||
       !pvrgpu_validate_conditionals_nir(fragment_nir,
                                         MESA_SHADER_FRAGMENT,
                                         error,
                                         error_size)) {
      return false;
   }

   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");

   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "failed to clone conditionals NIR");
   }

   /*
    * This model profile intentionally omits Vulkan-only ISP feedback and the
    * default point-size export.  Raster/depth state is carried by the model's
    * draw-state ABI, while the PCO programs remain deterministic and match the
    * development fixture compiler profile.
    */
   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = pco_nir_options();
   fs->options = pco_nir_options();

   nir_lower_fragcolor(fs, 1);
   nir_foreach_variable_with_modes (var, fs, nir_var_shader_in) {
      if (var->data.location == VARYING_SLOT_POS)
         var->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
   }

   if (!pvrgpu_canonicalize_fragment_output(fs,
                                            "conditionals",
                                            error,
                                            error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   if (!pvrgpu_lower_conditionals_fragment_mediump(fs,
                                                   error,
                                                   error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   if (!pvrgpu_lower_uniform_slots_to_push_constants(
          vs,
          PVRGPU_CONDITIONALS_VS_UNIFORM_DWORDS,
          4,
          "conditionals",
          error,
          error_size) ||
       !pvrgpu_lower_uniform_slots_to_push_constants(
          fs,
          PVRGPU_CONDITIONALS_FS_UNIFORM_DWORDS,
          1,
          "conditionals",
          error,
          error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   pvrgpu_init_conditionals_shader_data(&vertex_data,
                                        &fragment_data,
                                        vertex_format);

   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);

   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   if (!pvrgpu_allocate_push_constants(
          &vertex_data,
          PVRGPU_CONDITIONALS_VS_UNIFORM_DWORDS,
          PVRGPU_CONDITIONALS_VS_UNIFORM_DWORDS,
          "conditionals",
          "VS",
          error,
          error_size) ||
       !pvrgpu_allocate_push_constants(
          &fragment_data,
          PVRGPU_CONDITIONALS_FS_UNIFORM_DWORDS,
          2,
          "conditionals",
          "FS",
          error,
          error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate conditionals NIR");
   }

   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);

   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment,
                              false,
                              &out->fragment,
                              error,
                              error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   out->position_output_start = vertex_data.vs.varyings[VARYING_SLOT_POS].start;
   out->position_output_count = vertex_data.vs.varyings[VARYING_SLOT_POS].count;
   out->fragment_position_start =
      fragment_data.fs.varyings[VARYING_SLOT_POS].start;
   out->fragment_position_count =
      fragment_data.fs.varyings[VARYING_SLOT_POS].count;

   if (out->vertex.abi.shareds != PVRGPU_CONDITIONALS_VS_UNIFORM_DWORDS ||
       out->vertex.abi.push_constant_count !=
          PVRGPU_CONDITIONALS_VS_UNIFORM_DWORDS ||
       out->fragment.abi.shareds != PVRGPU_CONDITIONALS_FS_UNIFORM_DWORDS ||
       out->fragment.abi.push_constant_count !=
          PVRGPU_CONDITIONALS_FS_UNIFORM_DWORDS ||
       out->position_output_start != 0 || out->position_output_count != 4) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "compiled conditionals PCO ABI changed");
   }

   ralloc_free(compile_mem_ctx);
   return true;
}

/* Shared registers one lowered stage may reserve for its constant buffer. */
#define PVRGPU_COLOR_PRIMITIVE_MAX_UNIFORM_DWORDS 64u

// Uniform loads a stage still performs once dead code has been removed.
static unsigned pvrgpu_count_uniform_loads(const nir_shader *nir)
{
   unsigned loads = 0;
   nir_foreach_function_impl(impl, (nir_shader *)nir)
   {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            if (nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_load_uniform)
               ++loads;
         }
      }
   }
   return loads;
}

/*
 * Reserve shared registers for the whole constant buffer the draw binds.  The
 * pinned profiles assert an exact used-DWORD count because their shader is
 * fixed; a generic shader only has to stay inside the buffer it was given.
 */
static bool pvrgpu_allocate_generic_push_constants(pco_data *data,
                                                   unsigned dwords,
                                                   const char *stage,
                                                   char *error,
                                                   size_t error_size)
{
   if (data->common.shareds != 0) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive %s already owns shared "
                             "registers",
                             stage);
   }
   if (dwords == 0 || dwords > PVRGPU_COLOR_PRIMITIVE_MAX_UNIFORM_DWORDS ||
       data->common.push_consts.used > dwords) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive %s uniform range is "
                             "unsupported (used=%u available=%u)",
                             stage,
                             data->common.push_consts.used,
                             dwords);
   }
   data->common.push_consts.range = (pco_range){
      .start = 0,
      .count = dwords,
   };
   data->common.shareds = dwords;
   return true;
}

/*
 * Reserve the stage's constant buffer after a descriptor block that already
 * owns the first shared registers.
 */
static bool pvrgpu_allocate_generic_push_constants_after(pco_data *data,
                                                         unsigned prefix_dwords,
                                                         unsigned dwords,
                                                         const char *stage,
                                                         char *error,
                                                         size_t error_size)
{
   if (data->common.shareds != prefix_dwords ||
       dwords == 0 ||
       dwords > PVRGPU_COLOR_PRIMITIVE_MAX_UNIFORM_DWORDS ||
       data->common.push_consts.used > dwords) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive %s uniform range is "
                             "unsupported after its descriptors",
                             stage);
   }
   data->common.push_consts.range = (pco_range){
      .start = prefix_dwords,
      .count = dwords,
   };
   data->common.shareds = prefix_dwords + dwords;
   return true;
}

/*
 * Rewrites each texture reference onto its own descriptor set so a shader
 * sampling several textures fits the public PCO descriptor ABI.
 */
static bool
pvrgpu_pack_terrain_texture_bindings(nir_shader *nir,
                                     unsigned texture_count,
                                     unsigned expected_texture_ops,
                                     const char *profile_name,
                                     char *error,
                                     size_t error_size);

static bool pvrgpu_color_primitive_allowed_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_deref:
   case nir_intrinsic_store_deref:
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_store_output:
      return true;
   default:
      return false;
   }
}

/*
 * The color-primitive PCO data below hard-wires one vec4 position plus one
 * vec4 color attribute into VTXIN0..7, and one vec4 varying between
 * VARYING_SLOT_VAR0 and the fragment color output.  A shader that reads a
 * uniform, writes gl_PointSize, samples a texture, or otherwise steps outside
 * that signature has no place in this layout, and pco_trans_nir() aborts the
 * process rather than reporting it.  Reject it here so the driver fails
 * closed on the draw instead.
 */
static bool pvrgpu_validate_color_primitive_nir(const nir_shader *nir,
                                                mesa_shader_stage expected_stage,
                                                unsigned render_target_count,
                                                unsigned attribute_count,
                                                uint64_t varying_mask,
                                                bool writes_point_size,
                                                unsigned texture_count,
                                                char *error,
                                                size_t error_size)
{
   if (!nir || nir->info.stage != expected_stage) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive shader stage mismatch");
   }

   uint64_t vertex_inputs = 0;
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute)
      vertex_inputs |= BITFIELD64_BIT(VERT_ATTRIB_GENERIC0 + attribute);
   /*
    * Varyings occupy whichever slots the shaders agreed on, not VAR0 upwards:
    * the placement into registers is this profile's choice, so a shader that
    * writes VAR3..VAR5 is as lowerable as one that writes VAR0..VAR2.
    */
   const uint64_t varying_slots = varying_mask;
   const uint64_t expected_inputs =
      expected_stage == MESA_SHADER_VERTEX ? vertex_inputs : varying_slots;
   const uint64_t expected_outputs =
      expected_stage == MESA_SHADER_VERTEX
         ? (BITFIELD64_BIT(VARYING_SLOT_POS) | varying_slots |
            (writes_point_size ? BITFIELD64_BIT(VARYING_SLOT_PSIZ) : 0))
         : 0;

   if (nir->info.inputs_read != expected_inputs) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive %s reads unsupported inputs "
                             "0x%llx",
                             expected_stage == MESA_SHADER_VERTEX ? "VS" : "FS",
                             (unsigned long long)nir->info.inputs_read);
   }
   if (expected_stage == MESA_SHADER_VERTEX) {
      if (nir->info.outputs_written != expected_outputs) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "color primitive VS writes unsupported "
                                "outputs 0x%llx",
                                (unsigned long long)nir->info.outputs_written);
      }
   } else {
      uint64_t allowed_fs_outputs = BITFIELD64_BIT(FRAG_RESULT_COLOR);
      for (unsigned target = 0; target < render_target_count; ++target)
         allowed_fs_outputs |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + target);
      if (nir->info.outputs_written == 0 ||
          (nir->info.outputs_written & ~allowed_fs_outputs) != 0) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "color primitive FS writes unsupported "
                                "outputs 0x%llx",
                                (unsigned long long)nir->info.outputs_written);
      }
   }

   const bool stage_uses_discard =
      expected_stage == MESA_SHADER_FRAGMENT && nir->info.fs.uses_discard;
   /*
    * Default-block uniforms become push constants, so they are allowed.
    * Anything reached through a descriptor is not.
    */
   if (nir->info.num_ubos != 0 || nir->info.num_ssbos != 0 ||
       nir->info.num_images != 0 || nir->info.shared_size != 0 ||
       stage_uses_discard ||
       nir->info.num_textures != texture_count) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive %s uses unsupported resources",
                             expected_stage == MESA_SHADER_VERTEX ? "VS" : "FS");
   }

   unsigned implemented_functions = 0;
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (!function->is_entrypoint || ++implemented_functions != 1) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "color primitive requires one entrypoint");
      }
      nir_foreach_block (block, function->impl) {
         nir_foreach_instr (instr, block) {
            switch (instr->type) {
            case nir_instr_type_alu:
            case nir_instr_type_load_const:
            case nir_instr_type_deref:
            /*
             * An undefined SSA value is ordinary NIR -- it stands for a value
             * the language leaves unspecified, such as an uninitialised
             * variable or a lane that cannot be reached.  PCO lowers it to
             * whatever register it allocates; nothing reads a defined result
             * from it.
             */
            case nir_instr_type_undef:
               break;
            case nir_instr_type_tex: {
               /* Only plain sampling of a bound 2D texture is lowered. */
               const nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (texture_count == 0 || tex->op != nir_texop_tex ||
                   tex->is_array || tex->is_shadow ||
                   tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
                   tex->texture_index != tex->sampler_index ||
                   tex->texture_index >= texture_count) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "color primitive contains an "
                                         "unsupported texture operation");
               }
               break;
            }
            case nir_instr_type_intrinsic: {
               const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (!pvrgpu_color_primitive_allowed_intrinsic(intr->intrinsic)) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "color primitive contains unsupported NIR intrinsic %s",
                     nir_intrinsic_infos[intr->intrinsic].name);
               }
               break;
            }
            case nir_instr_type_jump:
               if (nir_instr_as_jump(instr)->type != nir_jump_return) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "color primitive contains "
                                         "unsupported control flow");
               }
               break;
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "color primitive contains unsupported "
                                      "NIR instruction type %u",
                                      instr->type);
            }
         }
      }
   }
   if (implemented_functions != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive requires one entrypoint");
   }
   return true;
}

/*
 * A `gl_PointSize` export only reaches rasterization for POINTS, and then only
 * when the pipeline is asked to take the size from the shader.  The dEQP
 * rasterization shaders share one vertex shader across every topology, so a
 * triangle, line, or fixed-size point draw still carries the export fed by a
 * uniform.  Remove that dead store when nothing consumes it: the uniform load
 * dies with it and the shader collapses onto the plain position/color
 * signature this profile lowers.
 */
static bool pvrgpu_strip_dead_point_size(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir)
   {
      bool impl_progress = false;
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_output) {
               if (nir_intrinsic_io_semantics(intr).location !=
                   VARYING_SLOT_PSIZ)
                  continue;
            } else if (intr->intrinsic == nir_intrinsic_store_deref) {
               nir_variable *var = nir_intrinsic_get_var(intr, 0);
               if (!var || var->data.location != VARYING_SLOT_PSIZ)
                  continue;
            } else {
               continue;
            }
            nir_instr_remove(instr);
            impl_progress = true;
         }
      }
      if (impl_progress) {
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      }
   }

   if (!progress)
      return false;

   nir_foreach_variable_with_modes_safe (var, nir, nir_var_shader_out) {
      if (var->data.location == VARYING_SLOT_PSIZ)
         exec_node_remove(&var->node);
   }
   nir->info.outputs_written &= ~VARYING_BIT_PSIZ;

   /* The uniform that fed the export is now dead; collect it and its load. */
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_out, NULL);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_uniform, NULL);
   nir->num_uniforms = 0;
   return true;
}

/*
 * Components the vertex shader declares for each generic attribute.  The model
 * matches a program's VTXIN read mask against the attribute bindings exactly,
 * so the driver has to supply each attribute at the width its shader declares
 * rather than a uniform four.  Returns false when an attribute the draw binds
 * is absent from the shader or is not a plain float vector.
 */
bool pvrgpu_pco_vertex_attribute_components(const struct nir_shader *vertex_nir,
                                            unsigned attribute_count,
                                            unsigned *components)
{
   if (!vertex_nir || !components || attribute_count == 0 ||
       attribute_count > PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES)
      return false;
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute)
      components[attribute] = 0;

   nir_foreach_variable_with_modes (var, vertex_nir, nir_var_shader_in) {
      if (var->data.location < VERT_ATTRIB_GENERIC0)
         return false;
      const unsigned index = var->data.location - VERT_ATTRIB_GENERIC0;
      if (index >= attribute_count)
         return false;
      /*
       * A matrix attribute occupies one location per column, each holding as
       * many components as the matrix has rows -- which is how the
       * application bound it, one vertex element per column.  Treating it as
       * a single wide attribute is what made every draw using one
       * unlowerable.
       */
      if (glsl_type_is_matrix(var->type)) {
         const unsigned columns = glsl_get_matrix_columns(var->type);
         const unsigned rows = glsl_get_vector_elements(var->type);
         if (columns == 0 || rows == 0 || rows > 4 ||
             index + columns > attribute_count)
            return false;
         for (unsigned column = 0; column < columns; ++column)
            components[index + column] = rows;
         continue;
      }
      const unsigned count = glsl_get_components(var->type);
      if (count == 0 || count > 4 || !glsl_type_is_vector_or_scalar(var->type))
         return false;
      components[index] = count;
   }

   /*
    * An application may bind an attribute its shader never reads.  Give the
    * unread binding a single placeholder component so the packed layout keeps
    * one entry per bound attribute; the shader ignores it, and the draw stays
    * lowerable instead of being dropped for a binding nothing consumes.
    */
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
      if (components[attribute] == 0)
         components[attribute] = 1;
   }
   return true;
}

/*
 * Components of each varying slot the vertex shader passes to the fragment
 * shader, starting at VARYING_SLOT_VAR0 and packed consecutively.
 *
 * Reports the slot count separately from success so that a shader passing no
 * varyings at all -- a shape coloured from a uniform, which is most of what
 * the scissor and fragment-op groups draw -- is a valid layout rather than an
 * unsupported one.  Position still occupies the first four outputs.
 */
static bool pvrgpu_color_primitive_varyings(const nir_shader *vs,
                                            const nir_shader *fs,
                                            unsigned *components,
                                            unsigned *locations,
                                            bool *read_by_fragment,
                                            unsigned *out_slots,
                                            uint64_t *out_mask)
{
   if (!vs || !fs || !components || !locations || !read_by_fragment ||
       !out_slots || !out_mask)
      return false;
   *out_slots = 0;
   *out_mask = 0;
   /*
    * gl_PointSize is a vertex output the rasterizer consumes, not a value the
    * fragment stage interpolates, so it is not part of the varying set the
    * two stages have to agree on.
    */
   const uint64_t vs_varyings = vs->info.outputs_written &
                                ~BITFIELD64_BIT(VARYING_SLOT_POS) &
                                ~BITFIELD64_BIT(VARYING_SLOT_PSIZ);
   /*
    * A vertex shader may write a varying the fragment shader never reads --
    * legal GLSL, and common once a shader is shared across draws.  The
    * fragment stage may not read one the vertex stage does not write.
    */
   const uint64_t fs_varyings = fs->info.inputs_read;
   if ((fs_varyings & ~vs_varyings) != 0)
      return false;
   if (vs_varyings == 0)
      return true;

   /*
    * Slots are packed in ascending order rather than required to start at
    * VAR0 and run consecutively.  Which registers a varying occupies is this
    * profile's choice, so a shader that writes VAR3..VAR5 is placed at the
    * first three slots; only the two stages agreeing on the set matters.
    */
   unsigned slots = 0;
   uint64_t placed = 0;
   for (unsigned slot = 0; slot < 64; ++slot) {
      const unsigned location = VARYING_SLOT_VAR0 + slot;
      if (location >= 64)
         break;
      if ((vs_varyings & BITFIELD64_BIT(location)) == 0)
         continue;
      if (slots >= PVRGPU_PCO_MAX_VARYINGS)
         return false;
      components[slots] = 0;
      nir_foreach_variable_with_modes (var, fs, nir_var_shader_in) {
         if (var->data.location != (int)location)
            continue;
         if (!glsl_type_is_vector_or_scalar(var->type))
            return false;
         components[slots] = glsl_get_components(var->type);
      }
      if (components[slots] == 0) {
         /*
          * The fragment stage does not read this one.  It still occupies a
          * vertex output, because the vertex shader writes it; give it the
          * width the vertex stage declares so the layout stays consistent,
          * and no fragment coefficients below.
          */
         nir_foreach_variable_with_modes (var, vs, nir_var_shader_out) {
            if (var->data.location != (int)location)
               continue;
            if (!glsl_type_is_vector_or_scalar(var->type))
               return false;
            components[slots] = glsl_get_components(var->type);
         }
      }
      if (components[slots] == 0 || components[slots] > 4)
         return false;
      read_by_fragment[slots] =
         (fs_varyings & BITFIELD64_BIT(location)) != 0;
      locations[slots] = location;
      placed |= BITFIELD64_BIT(location);
      ++slots;
   }
   /* Every written varying has to be one this loop placed. */
   if (placed != vs_varyings)
      return false;
   *out_slots = slots;
   *out_mask = placed;
   return true;
}

bool pvrgpu_pco_compile_color_triangle(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   const enum pipe_format *attribute_formats,
   bool topology_uses_point_size,
   unsigned render_target_count,
   unsigned vertex_uniform_dwords,
   unsigned fragment_uniform_dwords,
   unsigned attribute_count,
   unsigned texture_count,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size)
{
   const unsigned expected_stage_textures_vs = 0;
   if (texture_count > PVRGPU_PCO_MAX_TEXTURES) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive texture count is unsupported");
   }
   if (!attribute_formats || attribute_count == 0 ||
       attribute_count > PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive attribute count is unsupported");
   }
   if (render_target_count == 0 || render_target_count > 4) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive render target count is "
                             "unsupported");
   }
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));

   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "missing compiler or NIR stage for color triangle");
   }

   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");

   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error, error_size, "failed to clone color triangle NIR");
   }

   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = pco_nir_options();
   fs->options = pco_nir_options();

   if (!topology_uses_point_size)
      pvrgpu_strip_dead_point_size(vs);

   /*
    * Validate the shader this profile will actually hand to PCO.  pco_trans_nir
    * aborts the process on a signature it cannot translate, so anything outside
    * the position/color layout has to be rejected before that point.
    */
   unsigned probe_components[PVRGPU_PCO_MAX_VARYINGS] = {0};
   unsigned probe_locations[PVRGPU_PCO_MAX_VARYINGS] = {0};
   bool probe_read_by_fragment[PVRGPU_PCO_MAX_VARYINGS] = {false};
   unsigned probe_varyings = 0;
   uint64_t probe_varying_mask = 0;
   if (!pvrgpu_color_primitive_varyings(vs, fs, probe_components,
                                        probe_locations,
                                        probe_read_by_fragment,
                                        &probe_varyings,
                                        &probe_varying_mask)) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive varyings are unsupported: "
                             "vs=0x%llx fs=0x%llx",
                             (unsigned long long)vs->info.outputs_written,
                             (unsigned long long)fs->info.inputs_read);
   }
   const bool probe_writes_point_size =
      topology_uses_point_size &&
      (vs->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_PSIZ)) != 0;
   if (!pvrgpu_validate_color_primitive_nir(vs,
                                            MESA_SHADER_VERTEX,
                                            render_target_count,
                                            attribute_count,
                                            probe_varying_mask,
                                            probe_writes_point_size,
                                            expected_stage_textures_vs,
                                            error,
                                            error_size) ||
       !pvrgpu_validate_color_primitive_nir(fs,
                                            MESA_SHADER_FRAGMENT,
                                            render_target_count,
                                            attribute_count,
                                            fs->info.inputs_read,
                                            false,
                                            texture_count,
                                            error,
                                            error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   nir_lower_fragcolor(fs, 1);

   /*
    * Whatever uniform loads survive dead-code removal address the draw's
    * constant buffer, so rewrite them onto push constants sized by the buffer
    * the driver bound for that stage.
    */
   const unsigned vertex_uniform_loads = pvrgpu_count_uniform_loads(vs);
   const unsigned fragment_uniform_loads = pvrgpu_count_uniform_loads(fs);
   if ((vertex_uniform_loads != 0 &&
        !pvrgpu_lower_uniform_slots_to_push_constants(vs,
                                                      vertex_uniform_dwords,
                                                      vertex_uniform_loads,
                                                      "color primitive",
                                                      error,
                                                      error_size)) ||
       (fragment_uniform_loads != 0 &&
        !pvrgpu_lower_uniform_slots_to_push_constants(fs,
                                                      fragment_uniform_dwords,
                                                      fragment_uniform_loads,
                                                      "color primitive",
                                                      error,
                                                      error_size))) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };

   /*
    * One vec4-aligned VTXIN slot per attribute, each with the source format
    * it is fetched from.  PCO's vertex-input lowering dereferences the format
    * of every attribute it is told about, so leaving one unset is not an
    * option.
    */
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
      if (attribute_formats[attribute] == PIPE_FORMAT_NONE) {
         ralloc_free(compile_mem_ctx);
         return pvrgpu_pco_fail(error,
                                error_size,
                                "color primitive attribute %u has no source "
                                "format",
                                attribute);
      }
      vertex_data.vs.attrib_formats[VERT_ATTRIB_GENERIC0 + attribute] =
         attribute_formats[attribute];
      vertex_data.vs.attribs[VERT_ATTRIB_GENERIC0 + attribute] =
         (pco_range){
            .start = attribute * 4,
            .count = 4,
         };
   }
   vertex_data.common.vtxins = attribute_count * 4;
   unsigned varying_components[PVRGPU_PCO_MAX_VARYINGS] = {0};
   unsigned varying_locations[PVRGPU_PCO_MAX_VARYINGS] = {0};
   bool varying_read_by_fragment[PVRGPU_PCO_MAX_VARYINGS] = {false};
   unsigned varying_slots = 0;
   uint64_t varying_mask = 0;
   if (!pvrgpu_color_primitive_varyings(vs, fs, varying_components,
                                        varying_locations,
                                        varying_read_by_fragment,
                                        &varying_slots, &varying_mask)) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "color primitive varyings are unsupported: "
                             "vs=0x%llx fs=0x%llx",
                             (unsigned long long)vs->info.outputs_written,
                             (unsigned long long)fs->info.inputs_read);
   }
   unsigned varying_component_total = 0;
   for (unsigned slot = 0; slot < varying_slots; ++slot)
      varying_component_total += varying_components[slot];

   /*
    * Position occupies the first four VTXOUTs; the varyings follow it packed
    * in slot order.  The fragment stage sees four interpolation coefficients
    * per component, after its own four for position.
    */
   vertex_data.vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   fragment_data.fs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   unsigned vertex_output = 4;
   unsigned fragment_coefficient = 4;
   /*
    * A point draw whose shader sizes each point writes gl_PointSize; give it
    * the output immediately after position so the capsule can name one index
    * and the rasterizer can read it per vertex.
    */
   const bool writes_point_size =
      topology_uses_point_size &&
      (vs->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_PSIZ)) != 0;
   if (writes_point_size) {
      vertex_data.vs.varyings[VARYING_SLOT_PSIZ] = (pco_range){
         .start = vertex_output,
         .count = 1,
      };
      out->point_size_output_start = vertex_output;
      out->point_size_output_count = 1;
      vertex_output += 1;
   }
   unsigned fragment_varying_total = 0;
   for (unsigned slot = 0; slot < varying_slots; ++slot) {
      /* Keyed by the slot the shaders actually use, packed in that order. */
      const unsigned location = varying_locations[slot];
      vertex_data.vs.varyings[location] = (pco_range){
         .start = vertex_output,
         .count = varying_components[slot],
      };
      vertex_output += varying_components[slot];
      /* A varying the fragment stage never reads gets no coefficients. */
      if (!varying_read_by_fragment[slot])
         continue;
      fragment_data.fs.varyings[location] = (pco_range){
         .start = fragment_coefficient,
         .count = varying_components[slot] * 4,
      };
      fragment_coefficient += varying_components[slot] * 4;
      fragment_varying_total += varying_components[slot];
   }
   vertex_data.vs.vtxouts = vertex_output;
   /* Smooth interpolation is a fragment-stage property. */
   vertex_data.vs.f32_smooth = fragment_varying_total;

   fragment_data.fs.uses.w = true;
   fragment_data.common.coeffs = fragment_coefficient;

   /*
    * Each bound texture becomes its own descriptor set holding one combined
    * image/sampler binding, and the descriptor block precedes any constant
    * buffer in shared registers.
    */
   unsigned descriptor_dwords = 0;
   if (texture_count != 0) {
      unsigned texture_ops = 0;
      nir_foreach_function_impl(impl, fs)
      {
         nir_foreach_block (block, impl) {
            nir_foreach_instr (instr, block) {
               if (instr->type == nir_instr_type_tex)
                  ++texture_ops;
            }
         }
      }
      if (!pvrgpu_pack_terrain_texture_bindings(fs,
                                                texture_count,
                                                texture_ops,
                                                "color primitive",
                                                error,
                                                error_size)) {
         ralloc_free(compile_mem_ctx);
         return false;
      }
      for (unsigned texture = 0; texture < texture_count; ++texture) {
         pco_descriptor_set_data *set =
            &fragment_data.common.desc_sets[texture];
         set->binding_count = 1;
         set->bindings = rzalloc_array(compile_mem_ctx, pco_binding_data, 1);
         if (!set->bindings) {
            ralloc_free(compile_mem_ctx);
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "out of memory allocating texture "
                                   "descriptor ABI");
         }
         set->range = (pco_range){
            .start = texture * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
            .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
         };
         set->used = true;
         set->bindings[0].range = (pco_range){
            .start = texture * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
            .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
            .stride = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
         };
         set->bindings[0].used = true;
         set->bindings[0].is_img_smp = true;
      }
      descriptor_dwords = texture_count * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
      fragment_data.common.shareds = descriptor_dwords;
   }
   fragment_data.fs.z_replicate = ~0U;
   fragment_data.fs.rasterization_samples = 1;
   /*
    * One PIXOUT range per colour attachment.  A single-attachment shader also
    * declares FRAG_RESULT_COLOR at the same range, because a shader writing
    * gl_FragColor lands there rather than on a numbered output.
    */
   for (unsigned target = 0; target < render_target_count; ++target) {
      fragment_data.fs.outputs[FRAG_RESULT_DATA0 + target] = (pco_range){
         .start = target * 4,
         .count = 4,
      };
      fragment_data.fs.output_formats[FRAG_RESULT_DATA0 + target] =
         PIPE_FORMAT_R32G32B32A32_FLOAT;
   }
   if (render_target_count == 1) {
      fragment_data.fs.outputs[FRAG_RESULT_COLOR] = (pco_range){
         .start = 0,
         .count = 4,
      };
      fragment_data.fs.output_formats[FRAG_RESULT_COLOR] =
         PIPE_FORMAT_R32G32B32A32_FLOAT;
   }

   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);
   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   if ((vertex_uniform_loads != 0 &&
        !pvrgpu_allocate_generic_push_constants(&vertex_data,
                                                vertex_uniform_dwords,
                                                "VS",
                                                error,
                                                error_size)) ||
       (fragment_uniform_loads != 0 &&
        !pvrgpu_allocate_generic_push_constants_after(&fragment_data,
                                                      descriptor_dwords,
                                                      fragment_uniform_dwords,
                                                      "FS",
                                                      error,
                                                      error_size))) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate color triangle NIR");
   }

   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);

   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment, false, &out->fragment, error, error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   /* Position, then gl_PointSize when the shader writes it, then varyings. */
   out->position_output_start = 0;
   out->position_output_count = 4;
   out->fragment_position_start = 0;
   out->fragment_position_count = 4;
   out->varying_output_start = 4 + out->point_size_output_count;
   out->varying_output_count = varying_component_total;
   out->fragment_varying_start = 4;
   out->fragment_varying_count = fragment_varying_total * 4;

   ralloc_free(compile_mem_ctx);
   return true;
}

/* The Terrain D1 fragment source is default-mediump, except for the
 * source-level permute(highp vec4) helper.  Gallium's NIR arrives here with
 * all float arithmetic widened to 32 bits, so recover that one exact highp
 * island from the already validated source graph and make every remaining
 * binary16 arithmetic boundary explicit for the fp32-only public PCO ISA.
 *
 * This is intentionally structural and fail-closed.  It is not a shader-name
 * switch or a final-color quantizer: the twelve source permute calls must
 * scalarize to 48 exact
 *
 *     mod(((x * 34.0) + 1.0) * x, 289.0)
 *
 * roots.  Their three producers stay fp32 and the result is rounded to half.
 * The four mediump mod(i, 289.0) calls scalarize to twelve roots and are
 * expanded before rounding so division, multiply, and add each retain their
 * own binary16 boundary, matching the GLES/Gallivm precision graph. */
#define PVRGPU_TERRAIN_D1_HIGHP_ROOTS 48U
#define PVRGPU_TERRAIN_D1_HIGHP_PRODUCERS \
   (PVRGPU_TERRAIN_D1_HIGHP_ROOTS * 3U)

struct pvrgpu_terrain_d1_highp_graph {
   nir_instr *roots[PVRGPU_TERRAIN_D1_HIGHP_ROOTS];
   nir_instr *producers[PVRGPU_TERRAIN_D1_HIGHP_PRODUCERS];
   unsigned root_count;
   unsigned producer_count;
};

static bool
pvrgpu_terrain_d1_instr_array_contains(nir_instr *const *instrs,
                                       unsigned count,
                                       const nir_instr *instr)
{
   for (unsigned i = 0; i < count; ++i) {
      if (instrs[i] == instr)
         return true;
   }
   return false;
}

static bool
pvrgpu_terrain_d1_instr_array_add(nir_instr **instrs,
                                  unsigned *count,
                                  unsigned capacity,
                                  nir_instr *instr)
{
   if (!instr || *count >= capacity ||
       pvrgpu_terrain_d1_instr_array_contains(instrs, *count, instr))
      return false;
   instrs[(*count)++] = instr;
   return true;
}

static nir_alu_instr *
pvrgpu_terrain_d1_scalar_alu_source(const nir_alu_src *source)
{
   if (!source || !source->src.ssa)
      return NULL;
   nir_scalar scalar =
      nir_scalar_resolved(source->src.ssa, source->swizzle[0]);
   if (!scalar.def || scalar.comp != 0 ||
       scalar.def->num_components != 1 ||
       nir_def_instr(scalar.def)->type != nir_instr_type_alu)
      return NULL;
   return nir_instr_as_alu(nir_def_instr(scalar.def));
}

static bool
pvrgpu_terrain_d1_scalar_sources_equal(const nir_alu_src *left,
                                        const nir_alu_src *right)
{
   if (!left || !right || !left->src.ssa || !right->src.ssa)
      return false;
   nir_scalar left_scalar =
      nir_scalar_resolved(left->src.ssa, left->swizzle[0]);
   nir_scalar right_scalar =
      nir_scalar_resolved(right->src.ssa, right->swizzle[0]);
   return left_scalar.def && right_scalar.def &&
          left_scalar.def == right_scalar.def &&
          left_scalar.comp == right_scalar.comp;
}

static bool
pvrgpu_terrain_d1_scalar_source_is_bits(const nir_alu_src *source,
                                         uint32_t bits)
{
   if (!source || !source->src.ssa)
      return false;
   nir_scalar scalar =
      nir_scalar_resolved(source->src.ssa, source->swizzle[0]);
   return nir_scalar_is_const(scalar) &&
          nir_scalar_as_uint(scalar) == bits;
}

static bool
pvrgpu_terrain_d1_match_highp_permute(
   nir_alu_instr *root,
   nir_alu_instr **inner_multiply,
   nir_alu_instr **add_one,
   nir_alu_instr **outer_multiply)
{
   if (!root || root->op != nir_op_fmod || root->def.bit_size != 32 ||
       root->def.num_components != 1 ||
       !pvrgpu_terrain_d1_scalar_source_is_bits(
          &root->src[1], UINT32_C(0x43908000)))
      return false;

   nir_alu_instr *outer =
      pvrgpu_terrain_d1_scalar_alu_source(&root->src[0]);
   if (!outer || outer->op != nir_op_fmul ||
       outer->def.bit_size != 32 || outer->def.num_components != 1)
      return false;

   for (unsigned factor_source = 0; factor_source < 2; ++factor_source) {
      nir_alu_instr *add = pvrgpu_terrain_d1_scalar_alu_source(
         &outer->src[factor_source]);
      if (!add || add->op != nir_op_fadd || add->def.bit_size != 32 ||
          add->def.num_components != 1)
         continue;

      const unsigned x_source = 1U - factor_source;
      for (unsigned multiply_source = 0; multiply_source < 2;
           ++multiply_source) {
         const unsigned one_source = 1U - multiply_source;
         if (!pvrgpu_terrain_d1_scalar_source_is_bits(
                &add->src[one_source], UINT32_C(0x3f800000)))
            continue;

         nir_alu_instr *inner = pvrgpu_terrain_d1_scalar_alu_source(
            &add->src[multiply_source]);
         if (!inner || inner->op != nir_op_fmul ||
             inner->def.bit_size != 32 || inner->def.num_components != 1)
            continue;

         for (unsigned inner_x_source = 0; inner_x_source < 2;
              ++inner_x_source) {
            const unsigned constant_source = 1U - inner_x_source;
            if (!pvrgpu_terrain_d1_scalar_source_is_bits(
                   &inner->src[constant_source],
                   UINT32_C(0x42080000)) ||
                !pvrgpu_terrain_d1_scalar_sources_equal(
                   &inner->src[inner_x_source],
                   &outer->src[x_source]))
               continue;

            *inner_multiply = inner;
            *add_one = add;
            *outer_multiply = outer;
            return true;
         }
      }
   }
   return false;
}

static nir_def *
pvrgpu_terrain_d1_round_half_rtne(nir_builder *builder, nir_def *value)
{
   return nir_f2f32(builder, nir_f2f16_rtne(builder, value));
}

#define PVRGPU_TERRAIN_D1_OCTAVES 4U

struct pvrgpu_terrain_d1_splat_dot {
   nir_alu_instr *dot;
   unsigned value_source;
   unsigned coefficient_source;
   nir_scalar values[3];
   nir_scalar base_values[3];
   nir_scalar coefficient;
   unsigned scale;
};

static nir_scalar
pvrgpu_terrain_d1_resolve_alu_component(const nir_alu_src *source,
                                         unsigned component)
{
   if (!source || !source->src.ssa || component >= 3)
      return (nir_scalar){ 0 };
   return nir_scalar_resolved(source->src.ssa,
                              source->swizzle[component]);
}

static bool
pvrgpu_terrain_d1_scalars_equal(nir_scalar left, nir_scalar right)
{
   return left.def && right.def && left.def == right.def &&
          left.comp == right.comp;
}

static bool
pvrgpu_terrain_d1_scalar_arrays_equal(const nir_scalar left[3],
                                       const nir_scalar right[3])
{
   for (unsigned component = 0; component < 3; ++component) {
      if (!pvrgpu_terrain_d1_scalars_equal(left[component],
                                            right[component]))
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d1_alu_source_is_splat_bits(const nir_alu_src *source,
                                            uint32_t bits)
{
   if (!source || !source->src.ssa)
      return false;
   for (unsigned component = 0; component < 3; ++component) {
      uint64_t actual = 0;
      if (!nir_alu_src_comp_get_uint(*source, component, &actual) ||
          actual != bits)
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d1_collect_splat_dot(
   nir_alu_instr *dot,
   uint32_t coefficient_bits,
   struct pvrgpu_terrain_d1_splat_dot *out)
{
   if (!dot || !out || dot->op != nir_op_fdot3 ||
       dot->def.bit_size != 32 || dot->def.num_components != 1)
      return false;

   const bool source0_coefficient =
      pvrgpu_terrain_d1_alu_source_is_splat_bits(&dot->src[0],
                                                 coefficient_bits);
   const bool source1_coefficient =
      pvrgpu_terrain_d1_alu_source_is_splat_bits(&dot->src[1],
                                                 coefficient_bits);
   if (source0_coefficient == source1_coefficient)
      return false;

   memset(out, 0, sizeof(*out));
   out->dot = dot;
   out->coefficient_source = source0_coefficient ? 0U : 1U;
   out->value_source = 1U - out->coefficient_source;
   for (unsigned component = 0; component < 3; ++component) {
      out->values[component] = pvrgpu_terrain_d1_resolve_alu_component(
         &dot->src[out->value_source], component);
      if (!out->values[component].def)
         return false;
   }
   out->coefficient = pvrgpu_terrain_d1_resolve_alu_component(
      &dot->src[out->coefficient_source], 0);
   return out->coefficient.def != NULL;
}

static bool
pvrgpu_terrain_d1_classify_skew_scale(
   struct pvrgpu_terrain_d1_splat_dot *dot)
{
   if (!dot || !dot->dot)
      return false;

   nir_def *value = dot->dot->src[dot->value_source].src.ssa;
   if (!value || value->bit_size != 32 || value->num_components != 3)
      return false;

   nir_instr *producer = nir_def_instr(value);
   if (producer->type != nir_instr_type_alu ||
       nir_instr_as_alu(producer)->op != nir_op_fmul) {
      dot->scale = 1U;
      memcpy(dot->base_values, dot->values, sizeof(dot->base_values));
      return true;
   }

   nir_alu_instr *multiply = nir_instr_as_alu(producer);
   if (multiply->def.bit_size != 32 ||
       multiply->def.num_components != 3)
      return false;

   static const struct {
      uint32_t bits;
      unsigned scale;
   } scales[] = {
      { UINT32_C(0x40000000), 2U },
      { UINT32_C(0x40800000), 4U },
      { UINT32_C(0x41000000), 8U },
   };

   int constant_source = -1;
   unsigned scale = 0;
   for (unsigned source = 0; source < 2; ++source) {
      for (unsigned candidate = 0; candidate < ARRAY_SIZE(scales);
           ++candidate) {
         if (!pvrgpu_terrain_d1_alu_source_is_splat_bits(
                &multiply->src[source], scales[candidate].bits))
            continue;
         if (constant_source >= 0)
            return false;
         constant_source = (int)source;
         scale = scales[candidate].scale;
      }
   }
   if (constant_source < 0)
      return false;

   const unsigned base_source = 1U - (unsigned)constant_source;
   for (unsigned component = 0; component < 3; ++component) {
      const unsigned value_component =
         dot->dot->src[dot->value_source].swizzle[component];
      if (value_component >= 3)
         return false;
      dot->base_values[component] = nir_scalar_resolved(
         multiply->src[base_source].src.ssa,
         multiply->src[base_source].swizzle[value_component]);
      if (!dot->base_values[component].def)
         return false;
   }
   dot->scale = scale;
   return true;
}

static bool
pvrgpu_terrain_d1_cell_dot_matches_skew(
   const struct pvrgpu_terrain_d1_splat_dot *cell,
   const struct pvrgpu_terrain_d1_splat_dot *skew)
{
   if (!cell || !cell->dot || !skew || !skew->dot)
      return false;

   nir_def *cell_value = cell->dot->src[cell->value_source].src.ssa;
   if (!cell_value || cell_value->bit_size != 32 ||
       cell_value->num_components != 3 ||
       nir_def_instr(cell_value)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *floor = nir_instr_as_alu(nir_def_instr(cell_value));
   if (floor->op != nir_op_ffloor || floor->def.num_components != 3 ||
       !floor->src[0].src.ssa ||
       nir_def_instr(floor->src[0].src.ssa)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *add =
      nir_instr_as_alu(nir_def_instr(floor->src[0].src.ssa));
   if (add->op != nir_op_fadd || add->def.bit_size != 32 ||
       add->def.num_components != 3)
      return false;

   for (unsigned coordinate_source = 0; coordinate_source < 2;
        ++coordinate_source) {
      const unsigned skew_source = 1U - coordinate_source;
      bool matches = true;
      for (unsigned component = 0; component < 3; ++component) {
         nir_scalar coordinate = pvrgpu_terrain_d1_resolve_alu_component(
            &add->src[coordinate_source], component);
         nir_scalar skew_value = pvrgpu_terrain_d1_resolve_alu_component(
            &add->src[skew_source], component);
         if (!pvrgpu_terrain_d1_scalars_equal(coordinate,
                                               skew->values[component]) ||
             !skew_value.def || skew_value.def != &skew->dot->def ||
             skew_value.comp != 0) {
            matches = false;
            break;
         }
      }
      if (matches)
         return true;
   }
   return false;
}

static nir_def *
pvrgpu_terrain_d1_scalar_def(nir_builder *builder, nir_scalar scalar)
{
   if (!scalar.def)
      return NULL;
   if (scalar.comp == 0 && scalar.def->num_components == 1)
      return scalar.def;
   return nir_channel(builder, scalar.def, scalar.comp);
}

static bool
pvrgpu_terrain_d1_reconstruct_splat_dots(nir_shader *nir,
                                          nir_function_impl *entrypoint,
                                          char *error,
                                          size_t error_size)
{
   struct pvrgpu_terrain_d1_splat_dot skew[PVRGPU_TERRAIN_D1_OCTAVES] =
      { 0 };
   struct pvrgpu_terrain_d1_splat_dot cell[PVRGPU_TERRAIN_D1_OCTAVES] =
      { 0 };
   unsigned skew_count = 0;
   unsigned cell_count = 0;
   unsigned dot3_count = 0;
   unsigned dot4_count = 0;

   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op == nir_op_fdot4) {
            ++dot4_count;
            continue;
         }
         if (alu->op != nir_op_fdot3)
            continue;
         ++dot3_count;

         struct pvrgpu_terrain_d1_splat_dot matched = { 0 };
         if (pvrgpu_terrain_d1_collect_splat_dot(
                alu, UINT32_C(0x3eaaaaab), &matched)) {
            if (skew_count >= ARRAY_SIZE(skew))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D1 skew dot count changed");
            skew[skew_count++] = matched;
         } else if (pvrgpu_terrain_d1_collect_splat_dot(
                       alu, UINT32_C(0x3e2aaaab), &matched)) {
            if (cell_count >= ARRAY_SIZE(cell))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D1 cell dot count changed");
            cell[cell_count++] = matched;
         }
      }
   }

   if (dot3_count != 56U || dot4_count != 4U ||
       skew_count != PVRGPU_TERRAIN_D1_OCTAVES ||
       cell_count != PVRGPU_TERRAIN_D1_OCTAVES) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D1 dot signature changed (dot3=%u dot4=%u skew=%u cell=%u)",
         dot3_count,
         dot4_count,
         skew_count,
         cell_count);
   }

   static const unsigned expected_scales[PVRGPU_TERRAIN_D1_OCTAVES] = {
      1U, 2U, 4U, 8U,
   };
   for (unsigned octave = 0; octave < PVRGPU_TERRAIN_D1_OCTAVES;
        ++octave) {
      if (!pvrgpu_terrain_d1_classify_skew_scale(&skew[octave]) ||
          skew[octave].scale != expected_scales[octave]) {
         return pvrgpu_pco_fail(
            error,
            error_size,
            "terrain D1 skew scale graph changed at octave %u",
            octave);
      }
      if (octave != 0 &&
          !pvrgpu_terrain_d1_scalar_arrays_equal(skew[0].base_values,
                                                  skew[octave].base_values)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D1 skew base graph changed");
      }
      if (!pvrgpu_terrain_d1_cell_dot_matches_skew(&cell[octave],
                                                    &skew[octave])) {
         return pvrgpu_pco_fail(
            error,
            error_size,
            "terrain D1 cell/skew association changed at octave %u",
            octave);
      }
   }

   nir_builder base_builder =
      nir_builder_at(nir_before_instr(&skew[0].dot->instr));
   base_builder.fp_math_ctrl = skew[0].dot->fp_math_ctrl;
   nir_def *base_z = pvrgpu_terrain_d1_scalar_def(
      &base_builder, skew[0].base_values[2]);
   nir_def *base_y = pvrgpu_terrain_d1_scalar_def(
      &base_builder, skew[0].base_values[1]);
   nir_def *base_x = pvrgpu_terrain_d1_scalar_def(
      &base_builder, skew[0].base_values[0]);
   if (!base_x || !base_y || !base_z)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D1 skew base scalar is missing");
   nir_def *base_sum = nir_fadd(&base_builder, base_z, base_y);
   base_sum = nir_fadd(&base_builder, base_sum, base_x);

   static const uint32_t skew_coefficient_bits
      [PVRGPU_TERRAIN_D1_OCTAVES] = {
         UINT32_C(0x3eaaaaab),
         UINT32_C(0x3f2aaaab),
         UINT32_C(0x3faaaaab),
         UINT32_C(0x402aaaab),
      };
   for (unsigned octave = 0; octave < PVRGPU_TERRAIN_D1_OCTAVES;
        ++octave) {
      nir_builder builder =
         nir_builder_at(nir_before_instr(&skew[octave].dot->instr));
      builder.fp_math_ctrl = skew[octave].dot->fp_math_ctrl;
      nir_def *coefficient =
         octave == 0
            ? pvrgpu_terrain_d1_scalar_def(&builder,
                                            skew[octave].coefficient)
            : nir_imm_int(&builder, skew_coefficient_bits[octave]);
      if (!coefficient)
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D1 skew coefficient is missing");
      nir_def_replace(&skew[octave].dot->def,
                      nir_fmul(&builder, base_sum, coefficient));
   }

   for (unsigned octave = 0; octave < PVRGPU_TERRAIN_D1_OCTAVES;
        ++octave) {
      nir_builder builder =
         nir_builder_at(nir_before_instr(&cell[octave].dot->instr));
      builder.fp_math_ctrl = cell[octave].dot->fp_math_ctrl;
      nir_def *cell_z =
         pvrgpu_terrain_d1_scalar_def(&builder, cell[octave].values[2]);
      nir_def *cell_y =
         pvrgpu_terrain_d1_scalar_def(&builder, cell[octave].values[1]);
      nir_def *cell_x =
         pvrgpu_terrain_d1_scalar_def(&builder, cell[octave].values[0]);
      nir_def *coefficient = pvrgpu_terrain_d1_scalar_def(
         &builder, cell[octave].coefficient);
      if (!cell_x || !cell_y || !cell_z || !coefficient)
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D1 cell scalar is missing");
      nir_def *cell_sum = nir_fadd(&builder, cell_z, cell_y);
      cell_sum = nir_fadd(&builder, cell_sum, cell_x);
      nir_def_replace(&cell[octave].dot->def,
                      nir_fmul(&builder, cell_sum, coefficient));
   }

   nir_opt_dce(nir);
   return true;
}

static bool
pvrgpu_lower_terrain_d1_fragment_mediump(nir_shader *nir,
                                          char *error,
                                          size_t error_size)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D1 mediump requires a fragment shader");

   nir_function_impl *entrypoint = NULL;
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (entrypoint) {
         return pvrgpu_pco_fail(
            error,
            error_size,
            "terrain D1 mediump requires one entrypoint");
      }
      entrypoint = function->impl;
   }
   if (!entrypoint)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D1 mediump entrypoint is missing");

   if (getenv("PVRGPU_TERRAIN_D1_PRECISION_DEBUG")) {
      fprintf(stderr, "terrain D1 precision graph before scalarization:\n");
      nir_print_shader(nir, stderr);
   }

   if (!pvrgpu_terrain_d1_reconstruct_splat_dots(nir,
                                                  entrypoint,
                                                  error,
                                                  error_size))
      return false;

   /* Gallivm lowers fdot3 as z+y+x and fdot4 as w+z+y+x.  PCO's generic
    * scalarizer uses the same reverse component order; exposing the scalar
    * products here lets every half multiply and intermediate add receive an
    * independent RTNE boundary below. */
   nir_lower_alu_to_scalar(nir, NULL, NULL);

   if (getenv("PVRGPU_TERRAIN_D1_PRECISION_DEBUG")) {
      fprintf(stderr, "terrain D1 precision graph after scalarization:\n");
      nir_print_shader(nir, stderr);
   }

   struct pvrgpu_terrain_d1_highp_graph highp = { 0 };
   unsigned total_fmods = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_fmod)
            continue;
         ++total_fmods;

         nir_alu_instr *inner = NULL;
         nir_alu_instr *add = NULL;
         nir_alu_instr *outer = NULL;
         if (!pvrgpu_terrain_d1_match_highp_permute(alu,
                                                     &inner,
                                                     &add,
                                                     &outer))
            continue;
         if (!pvrgpu_terrain_d1_instr_array_add(
                highp.roots,
                &highp.root_count,
                ARRAY_SIZE(highp.roots),
                instr) ||
             !pvrgpu_terrain_d1_instr_array_add(
                highp.producers,
                &highp.producer_count,
                ARRAY_SIZE(highp.producers),
                &inner->instr) ||
             !pvrgpu_terrain_d1_instr_array_add(
                highp.producers,
                &highp.producer_count,
                ARRAY_SIZE(highp.producers),
                &add->instr) ||
             !pvrgpu_terrain_d1_instr_array_add(
                highp.producers,
                &highp.producer_count,
                ARRAY_SIZE(highp.producers),
                &outer->instr)) {
            return pvrgpu_pco_fail(
               error,
               error_size,
               "terrain D1 highp permute graph is ambiguous");
         }
      }
   }
   if (total_fmods != 60U ||
       highp.root_count != PVRGPU_TERRAIN_D1_HIGHP_ROOTS ||
       highp.producer_count != PVRGPU_TERRAIN_D1_HIGHP_PRODUCERS) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D1 highp permute signature changed "
         "(fmod=%u roots=%u producers=%u)",
         total_fmods,
         highp.root_count,
         highp.producer_count);
   }

   /* Expand only the twelve scalar mediump modulo roots.  Leaving these to
    * generic fp32 fmod lowering would round only the final value and lose the
    * half division/multiply/add boundaries present in the reference graph. */
   unsigned expanded_mediump_fmods = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_fmod ||
             pvrgpu_terrain_d1_instr_array_contains(
                highp.roots, highp.root_count, instr))
            continue;
         if (alu->def.bit_size != 32 || alu->def.num_components != 1 ||
             !pvrgpu_terrain_d1_scalar_source_is_bits(
                &alu->src[1], UINT32_C(0x43908000))) {
            return pvrgpu_pco_fail(
               error,
               error_size,
               "terrain D1 mediump modulo graph changed");
         }

         nir_builder builder = nir_builder_at(nir_before_instr(instr));
         builder.fp_math_ctrl = alu->fp_math_ctrl;
         nir_def *value = nir_ssa_for_alu_src(&builder, alu, 0);
         nir_def *modulus = nir_ssa_for_alu_src(&builder, alu, 1);
         nir_def *quotient = nir_fdiv(&builder, value, modulus);
         nir_def *floored = nir_ffloor(&builder, quotient);
         nir_def *multiple = nir_fmul(&builder, modulus, floored);
         nir_def *result =
            nir_fadd(&builder, value, nir_fneg(&builder, multiple));
         nir_def_replace(&alu->def, result);
         ++expanded_mediump_fmods;
      }
   }
   if (expanded_mediump_fmods != 12U) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D1 mediump modulo count changed (%u)",
         expanded_mediump_fmods);
   }

   /* Constants are typeless in NIR.  Round only their ALU uses so a shared
    * zero used as an intrinsic byte offset remains an immediate integer, while
    * constants such as 1/3 and 1/6 enter mediump arithmetic as exact binary16
    * values.  The highp permute constants 1, 34, and 289 are all exactly
    * representable in binary16, so this does not introduce a numeric boundary
    * inside the recognized island. */
   unsigned rounded_constants = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_load_const)
            continue;
         nir_load_const_instr *constant = nir_instr_as_load_const(instr);
         if (constant->def.bit_size != 32)
            continue;

         bool has_alu_use = false;
         nir_foreach_use (use, &constant->def) {
            if (nir_src_use_instr(use)->type == nir_instr_type_alu) {
               has_alu_use = true;
               break;
            }
         }
         if (!has_alu_use)
            continue;

         nir_builder builder = nir_builder_at(nir_after_instr(instr));
         nir_def *half = nir_f2f16_rtne(&builder, &constant->def);
         nir_def *rounded = nir_f2f32(&builder, half);
         nir_instr *half_instr = nir_def_instr(half);
         nir_instr *rounded_instr = nir_def_instr(rounded);
         nir_foreach_use_safe (use, &constant->def) {
            nir_instr *use_instr = nir_src_use_instr(use);
            if (use_instr == half_instr || use_instr == rounded_instr ||
                use_instr->type != nir_instr_type_alu)
               continue;
            nir_src_rewrite(use, rounded);
         }
         ++rounded_constants;
      }
   }

   unsigned rounded_inputs = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         bool round_input = false;
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            nir_variable *var = nir_intrinsic_get_var(intr, 0);
            round_input = var && var->data.mode == nir_var_shader_in &&
                          var->data.precision == GLSL_PRECISION_MEDIUM;
         } else if (intr->intrinsic == nir_intrinsic_load_push_constant) {
            round_input = true;
         }
         if (!round_input)
            continue;
         if (intr->def.bit_size != 32) {
            return pvrgpu_pco_fail(
               error,
               error_size,
               "terrain D1 mediump input precision changed");
         }
         nir_builder builder = nir_builder_at(nir_after_instr(instr));
         nir_def *rounded =
            pvrgpu_terrain_d1_round_half_rtne(&builder, &intr->def);
         nir_def_rewrite_uses_after(&intr->def, rounded);
         ++rounded_inputs;
      }
   }

   unsigned rounded_adds = 0;
   unsigned rounded_multiplies = 0;
   unsigned rounded_divides = 0;
   unsigned rounded_permute_results = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         bool round_result = false;
         switch (alu->op) {
         case nir_op_fadd:
            if (!pvrgpu_terrain_d1_instr_array_contains(
                   highp.producers, highp.producer_count, instr)) {
               ++rounded_adds;
               round_result = true;
            }
            break;
         case nir_op_fmul:
            if (!pvrgpu_terrain_d1_instr_array_contains(
                   highp.producers, highp.producer_count, instr)) {
               ++rounded_multiplies;
               round_result = true;
            }
            break;
         case nir_op_fdiv:
            ++rounded_divides;
            round_result = true;
            break;
         case nir_op_fmod:
            if (!pvrgpu_terrain_d1_instr_array_contains(
                   highp.roots, highp.root_count, instr)) {
               return pvrgpu_pco_fail(
                  error,
                  error_size,
                  "terrain D1 retained an unclassified modulo");
            }
            ++rounded_permute_results;
            round_result = true;
            break;
         case nir_op_f2f16_rtne:
         case nir_op_f2f32:
            break;
         case nir_op_ffma:
         case nir_op_ffma_weak:
            return pvrgpu_pco_fail(
               error,
               error_size,
               "terrain D1 fused arithmetic changed");
         default:
            break;
         }
         if (!round_result)
            continue;
         if (alu->def.bit_size != 32) {
            return pvrgpu_pco_fail(
               error,
               error_size,
               "terrain D1 arithmetic precision changed");
         }
         nir_builder builder = nir_builder_at(nir_after_instr(instr));
         nir_def *rounded =
            pvrgpu_terrain_d1_round_half_rtne(&builder, &alu->def);
         nir_def_rewrite_uses_after(&alu->def, rounded);
      }
   }

   if (rounded_inputs != 2U || rounded_constants != 26U ||
       rounded_adds != 513U || rounded_multiplies != 452U ||
       rounded_divides != 12U ||
       rounded_permute_results != PVRGPU_TERRAIN_D1_HIGHP_ROOTS) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D1 mediump lowered graph signature changed "
         "(inputs=%u constants=%u add=%u mul=%u div=%u permute=%u)",
         rounded_inputs,
         rounded_constants,
         rounded_adds,
         rounded_multiplies,
         rounded_divides,
         rounded_permute_results);
   }

   nir_progress(true, entrypoint, nir_metadata_control_flow);
   return true;
}

static bool
pvrgpu_terrain_d2_offset_coord_matches(const nir_tex_instr *texture,
                                        const nir_intrinsic_instr *varying,
                                        const nir_intrinsic_instr *resolution,
                                        unsigned active_component)
{
   if (!texture || !varying || !resolution || active_component >= 2)
      return false;
   const int coord_index =
      nir_tex_instr_src_index(texture, nir_tex_src_coord);
   if (coord_index < 0 || !texture->src[coord_index].src.ssa ||
       nir_def_instr(texture->src[coord_index].src.ssa)->type !=
          nir_instr_type_alu)
      return false;
   nir_alu_instr *add = nir_instr_as_alu(
      nir_def_instr(texture->src[coord_index].src.ssa));
   if (add->op != nir_op_fadd || add->def.bit_size != 32 ||
       add->def.num_components != 2)
      return false;

   for (unsigned varying_source = 0; varying_source < 2;
        ++varying_source) {
      const unsigned offset_source = 1U - varying_source;
      bool matches = true;
      for (unsigned component = 0; component < 2; ++component) {
         nir_scalar varying_value = nir_scalar_resolved(
            add->src[varying_source].src.ssa,
            add->src[varying_source].swizzle[component]);
         nir_scalar expected_varying = {
            .def = (nir_def *)&varying->def,
            .comp = component,
         };
         if (!pvrgpu_terrain_d1_scalars_equal(varying_value,
                                               expected_varying)) {
            matches = false;
            break;
         }

         nir_scalar offset = nir_scalar_resolved(
            add->src[offset_source].src.ssa,
            add->src[offset_source].swizzle[component]);
         if (component != active_component) {
            if (!nir_scalar_is_const(offset) ||
                nir_scalar_as_uint(offset) != 0U) {
               matches = false;
               break;
            }
            continue;
         }

         if (!offset.def || offset.comp != 0 ||
             offset.def->num_components != 1 ||
             nir_def_instr(offset.def)->type != nir_instr_type_alu) {
            matches = false;
            break;
         }
         nir_alu_instr *reciprocal =
            nir_instr_as_alu(nir_def_instr(offset.def));
         if (reciprocal->op != nir_op_frcp ||
             reciprocal->def.bit_size != 32 ||
             reciprocal->def.num_components != 1) {
            matches = false;
            break;
         }
         nir_scalar reciprocal_source = nir_scalar_resolved(
            reciprocal->src[0].src.ssa,
            reciprocal->src[0].swizzle[0]);
         nir_scalar expected_resolution = {
            .def = (nir_def *)&resolution->def,
            .comp = active_component,
         };
         if (!pvrgpu_terrain_d1_scalars_equal(reciprocal_source,
                                               expected_resolution)) {
            matches = false;
            break;
         }
      }
      if (matches)
         return true;
   }
   return false;
}

static nir_def *
pvrgpu_terrain_d2_round_texture_x_rtz(nir_tex_instr *texture)
{
   nir_builder builder =
      nir_builder_at(nir_after_instr(&texture->instr));
   nir_def *components[4];
   components[0] = nir_f2f32(
      &builder,
      nir_f2f16_rtz(&builder, nir_channel(&builder, &texture->def, 0)));
   for (unsigned component = 1; component < 4; ++component)
      components[component] =
         nir_channel(&builder, &texture->def, component);
   nir_def *replacement = nir_vec(&builder, components, 4);
   nir_def_rewrite_uses_after(&texture->def, replacement);
   return replacement;
}

/* Terrain D2 reconstructs a normal map from three height-map samples.  The
 * source variables are mediump, but the Gallium-facing NIR is widened to
 * fp32 before it reaches PCO.  Restore only the captured GLES precision graph:
 * the center texture coordinate and each untouched offset component remain
 * fp32, while active UV/resolution components enter half arithmetic through
 * RTNE; normalized texture results enter through RTZ.  Every half add,
 * multiply, and reciprocal receives its own RTNE boundary.  LLVM implements
 * half rsqrt as a rounded half sqrt followed by a rounded half reciprocal, so
 * expose both steps rather than rounding one fp32 FRSQ result. */
static bool
pvrgpu_lower_terrain_d2_fragment_mediump(nir_shader *nir,
                                          char *error,
                                          size_t error_size)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D2 mediump requires a fragment shader");

   nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
   if (!entrypoint)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D2 mediump entrypoint is missing");

   nir_intrinsic_instr *varying = NULL;
   nir_intrinsic_instr *height = NULL;
   nir_intrinsic_instr *resolution = NULL;
   nir_tex_instr *textures[3] = { 0 };
   unsigned texture_count = 0;
   unsigned source_adds = 0;
   unsigned source_multiplies = 0;
   unsigned source_reciprocals = 0;
   unsigned source_rsqrt = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               nir_variable *var = nir_intrinsic_get_var(intr, 0);
               if (!var || var->data.mode != nir_var_shader_in ||
                   var->data.location != VARYING_SLOT_VAR0 || varying ||
                   intr->def.bit_size != 32 ||
                   intr->def.num_components != 2) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D2 varying graph changed");
               }
               varying = intr;
            } else if (intr->intrinsic ==
                       nir_intrinsic_load_push_constant) {
               if (intr->def.bit_size != 32)
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D2 push precision changed");
               if (intr->def.num_components == 1 && !height)
                  height = intr;
               else if (intr->def.num_components == 2 && !resolution)
                  resolution = intr;
               else
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D2 push graph changed");
            }
         } else if (instr->type == nir_instr_type_tex) {
            if (texture_count >= ARRAY_SIZE(textures))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D2 texture count changed");
            textures[texture_count++] = nir_instr_as_tex(instr);
         } else if (instr->type == nir_instr_type_alu) {
            switch (nir_instr_as_alu(instr)->op) {
            case nir_op_fadd:
               ++source_adds;
               break;
            case nir_op_fmul:
               ++source_multiplies;
               break;
            case nir_op_frcp:
               ++source_reciprocals;
               break;
            case nir_op_frsq:
               ++source_rsqrt;
               break;
            case nir_op_ffma:
            case nir_op_ffma_weak:
               return pvrgpu_pco_fail(
                  error,
                  error_size,
                  "terrain D2 fused arithmetic changed");
            default:
               break;
            }
         }
      }
   }

   if (!varying || !height || !resolution || texture_count != 3U ||
       source_adds != 5U || source_multiplies != 2U ||
       source_reciprocals != 2U || source_rsqrt != 1U) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D2 source graph changed (tex=%u add=%u mul=%u rcp=%u rsq=%u)",
         texture_count,
         source_adds,
         source_multiplies,
         source_reciprocals,
         source_rsqrt);
   }

   const int center_coord_index =
      nir_tex_instr_src_index(textures[0], nir_tex_src_coord);
   if (center_coord_index < 0 ||
       textures[0]->src[center_coord_index].src.ssa != &varying->def ||
       !pvrgpu_terrain_d2_offset_coord_matches(textures[1],
                                                varying,
                                                resolution,
                                                0) ||
       !pvrgpu_terrain_d2_offset_coord_matches(textures[2],
                                                varying,
                                                resolution,
                                                1)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D2 texture coordinate graph changed");
   }

   nir_builder varying_builder =
      nir_builder_at(nir_after_instr(&varying->instr));
   nir_def *uv_half[2];
   for (unsigned component = 0; component < 2; ++component) {
      uv_half[component] = pvrgpu_terrain_d1_round_half_rtne(
         &varying_builder,
         nir_channel(&varying_builder, &varying->def, component));
   }

   nir_builder resolution_builder =
      nir_builder_at(nir_after_instr(&resolution->instr));
   nir_def *reciprocal[2];
   for (unsigned component = 0; component < 2; ++component) {
      nir_def *rounded_resolution = pvrgpu_terrain_d1_round_half_rtne(
         &resolution_builder,
         nir_channel(&resolution_builder, &resolution->def, component));
      reciprocal[component] = nir_frcp(&resolution_builder,
                                        rounded_resolution);
   }

   nir_builder coord_u_builder =
      nir_builder_at(nir_before_instr(&textures[1]->instr));
   nir_def *coord_u_x =
      nir_fadd(&coord_u_builder, uv_half[0], reciprocal[0]);
   nir_def *coord_u = nir_vec2(
      &coord_u_builder,
      coord_u_x,
      nir_channel(&coord_u_builder, &varying->def, 1));
   const int coord_u_index =
      nir_tex_instr_src_index(textures[1], nir_tex_src_coord);
   nir_src_rewrite(&textures[1]->src[coord_u_index].src, coord_u);

   nir_builder coord_v_builder =
      nir_builder_at(nir_before_instr(&textures[2]->instr));
   nir_def *coord_v_y =
      nir_fadd(&coord_v_builder, uv_half[1], reciprocal[1]);
   nir_def *coord_v = nir_vec2(
      &coord_v_builder,
      nir_channel(&coord_v_builder, &varying->def, 0),
      coord_v_y);
   const int coord_v_index =
      nir_tex_instr_src_index(textures[2], nir_tex_src_coord);
   nir_src_rewrite(&textures[2]->src[coord_v_index].src, coord_v);

   nir_builder height_builder =
      nir_builder_at(nir_after_instr(&height->instr));
   nir_def *rounded_height = pvrgpu_terrain_d1_round_half_rtne(
      &height_builder, &height->def);
   nir_def_rewrite_uses_after(&height->def, rounded_height);

   for (unsigned texture = 0; texture < ARRAY_SIZE(textures); ++texture)
      pvrgpu_terrain_d2_round_texture_x_rtz(textures[texture]);

   nir_opt_dce(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);

   nir_alu_instr *rsqrt = NULL;
   unsigned rounded_adds = 0;
   unsigned rounded_multiplies = 0;
   unsigned rounded_reciprocals = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         bool round_result = false;
         switch (alu->op) {
         case nir_op_fadd:
            ++rounded_adds;
            round_result = true;
            break;
         case nir_op_fmul:
            ++rounded_multiplies;
            round_result = true;
            break;
         case nir_op_frcp:
            ++rounded_reciprocals;
            round_result = true;
            break;
         case nir_op_frsq:
            if (rsqrt)
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D2 rsqrt count changed");
            rsqrt = alu;
            break;
         case nir_op_ffma:
         case nir_op_ffma_weak:
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "terrain D2 fused arithmetic changed");
         default:
            break;
         }
         if (!round_result)
            continue;
         if (alu->def.bit_size != 32 || alu->def.num_components != 1)
            return pvrgpu_pco_fail(
               error,
               error_size,
               "terrain D2 scalar arithmetic changed");
         nir_builder builder =
            nir_builder_at(nir_after_instr(&alu->instr));
         nir_def *rounded = pvrgpu_terrain_d1_round_half_rtne(
            &builder, &alu->def);
         nir_def_rewrite_uses_after(&alu->def, rounded);
      }
   }

   if (!rsqrt || rounded_adds != 9U || rounded_multiplies != 9U ||
       rounded_reciprocals != 2U) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D2 lowered graph changed (add=%u mul=%u rcp=%u rsq=%u)",
         rounded_adds,
         rounded_multiplies,
         rounded_reciprocals,
         rsqrt ? 1U : 0U);
   }

   nir_builder rsqrt_builder =
      nir_builder_at(nir_after_instr(&rsqrt->instr));
   nir_def *sqrt_value = nir_frcp(&rsqrt_builder, &rsqrt->def);
   nir_def *sqrt_rounded = pvrgpu_terrain_d1_round_half_rtne(
      &rsqrt_builder, sqrt_value);
   nir_def *scale = nir_frcp(&rsqrt_builder, sqrt_rounded);
   nir_def *scale_rounded = pvrgpu_terrain_d1_round_half_rtne(
      &rsqrt_builder, scale);
   nir_def_rewrite_uses_after(&rsqrt->def, scale_rounded);

   nir_progress(true, entrypoint, nir_metadata_control_flow);
   return true;
}

static nir_def *
pvrgpu_terrain_d3_round_texture_rtz(nir_tex_instr *texture)
{
   nir_builder builder =
      nir_builder_at(nir_after_instr(&texture->instr));
   nir_def *components[4];
   for (unsigned component = 0; component < ARRAY_SIZE(components);
        ++component) {
      components[component] = nir_f2f32(
         &builder,
         nir_f2f16_rtz(
            &builder,
            nir_channel(&builder, &texture->def, component)));
   }
   nir_def *replacement = nir_vec(&builder, components, 4);
   nir_def_rewrite_uses_after(&texture->def, replacement);
   return replacement;
}

static bool
pvrgpu_terrain_d3_scalar_source_is_texture_component(
   const nir_alu_src *source,
   const nir_tex_instr *texture,
   unsigned component)
{
   if (!source || !source->src.ssa || !texture || component >= 4)
      return false;
   const nir_scalar scalar =
      nir_scalar_resolved(source->src.ssa, source->swizzle[0]);
   return scalar.def == &texture->def && scalar.comp == component;
}

static bool
pvrgpu_terrain_d3_scalar_source_is_def(const nir_alu_src *source,
                                        const nir_def *def)
{
   if (!source || !source->src.ssa || !def || def->num_components != 1)
      return false;
   const nir_scalar scalar =
      nir_scalar_resolved(source->src.ssa, source->swizzle[0]);
   return scalar.def == def && scalar.comp == 0;
}

/* Terrain D3's vertex source is highp except for normalized texture results.
 * Gallivm converts all four channels of both UNORM samples to binary16 with
 * RTZ.  The displacement sample is widened immediately and all of its math
 * remains highp.  Only the normal-map RGB `sample * 2.0 - 1.0` chain has a
 * mediump ALU result boundary: three RTNE multiplies and three RTNE adds.
 * Varying stores intentionally remain fp32; the fragment pass restores their
 * mediump boundary at the consumer. */
static bool
pvrgpu_lower_terrain_d3_vertex_mediump(nir_shader *nir,
                                        char *error,
                                        size_t error_size)
{
   if (!nir || nir->info.stage != MESA_SHADER_VERTEX)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 mediump requires a vertex shader");

   nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
   if (!entrypoint)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 vertex entrypoint is missing");

   nir_tex_instr *textures[2] = { 0 };
   unsigned texture_count = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         if (texture_count >= ARRAY_SIZE(textures))
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "terrain D3 VS texture count changed");
         textures[texture_count++] = nir_instr_as_tex(instr);
      }
   }

   if (texture_count != ARRAY_SIZE(textures) ||
       textures[0]->texture_index != 0U ||
       textures[0]->sampler_index != 0U ||
       textures[1]->texture_index != (1U << 16U) ||
       textures[1]->sampler_index != (1U << 16U) ||
       textures[0]->def.bit_size != 32 ||
       textures[0]->def.num_components != 4 ||
       textures[1]->def.bit_size != 32 ||
       textures[1]->def.num_components != 4) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 VS texture ABI changed");
   }

   const int displacement_coord =
      nir_tex_instr_src_index(textures[0], nir_tex_src_coord);
   const int normal_coord =
      nir_tex_instr_src_index(textures[1], nir_tex_src_coord);
   if (displacement_coord < 0 || normal_coord < 0 ||
       !textures[0]->src[displacement_coord].src.ssa ||
       textures[0]->src[displacement_coord].src.ssa !=
          textures[1]->src[normal_coord].src.ssa ||
       textures[0]->src[displacement_coord].src.ssa->bit_size != 32 ||
       textures[0]->src[displacement_coord].src.ssa->num_components != 2) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 VS texture coordinate graph changed");
   }

   nir_lower_alu_to_scalar(nir, NULL, NULL);

   nir_alu_instr *normal_multiplies[3] = { 0 };
   nir_alu_instr *normal_adds[3] = { 0 };
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->def.bit_size != 32 || alu->def.num_components != 1)
            continue;
         if (alu->op == nir_op_fmul) {
            for (unsigned component = 0;
                 component < ARRAY_SIZE(normal_multiplies);
                 ++component) {
               const bool match =
                  (pvrgpu_terrain_d3_scalar_source_is_texture_component(
                      &alu->src[0], textures[1], component) &&
                   pvrgpu_terrain_d1_scalar_source_is_bits(
                      &alu->src[1], UINT32_C(0x40000000))) ||
                  (pvrgpu_terrain_d3_scalar_source_is_texture_component(
                      &alu->src[1], textures[1], component) &&
                   pvrgpu_terrain_d1_scalar_source_is_bits(
                      &alu->src[0], UINT32_C(0x40000000)));
               if (!match)
                  continue;
               if (normal_multiplies[component])
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D3 VS normal multiply graph is ambiguous");
               normal_multiplies[component] = alu;
            }
         }
      }
   }

   for (unsigned component = 0; component < ARRAY_SIZE(normal_multiplies);
        ++component) {
      if (!normal_multiplies[component])
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D3 VS normal multiply graph changed");
   }

   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_fadd || alu->def.bit_size != 32 ||
             alu->def.num_components != 1)
            continue;
         for (unsigned component = 0;
              component < ARRAY_SIZE(normal_adds);
              ++component) {
            const bool match =
               (pvrgpu_terrain_d3_scalar_source_is_def(
                   &alu->src[0], &normal_multiplies[component]->def) &&
                pvrgpu_terrain_d1_scalar_source_is_bits(
                   &alu->src[1], UINT32_C(0xbf800000))) ||
               (pvrgpu_terrain_d3_scalar_source_is_def(
                   &alu->src[1], &normal_multiplies[component]->def) &&
                pvrgpu_terrain_d1_scalar_source_is_bits(
                   &alu->src[0], UINT32_C(0xbf800000)));
            if (!match)
               continue;
            if (normal_adds[component])
               return pvrgpu_pco_fail(
                  error,
                  error_size,
                  "terrain D3 VS normal add graph is ambiguous");
            normal_adds[component] = alu;
         }
      }
   }

   for (unsigned component = 0; component < ARRAY_SIZE(normal_adds);
        ++component) {
      if (!normal_adds[component])
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D3 VS normal add graph changed");
   }

   for (unsigned texture = 0; texture < ARRAY_SIZE(textures); ++texture)
      pvrgpu_terrain_d3_round_texture_rtz(textures[texture]);

   for (unsigned component = 0; component < ARRAY_SIZE(normal_multiplies);
        ++component) {
      nir_builder builder =
         nir_builder_at(nir_after_instr(&normal_multiplies[component]->instr));
      nir_def *rounded = pvrgpu_terrain_d1_round_half_rtne(
         &builder, &normal_multiplies[component]->def);
      nir_def_rewrite_uses_after(&normal_multiplies[component]->def,
                                 rounded);
   }
   for (unsigned component = 0; component < ARRAY_SIZE(normal_adds);
        ++component) {
      nir_builder builder =
         nir_builder_at(nir_after_instr(&normal_adds[component]->instr));
      nir_def *rounded = pvrgpu_terrain_d1_round_half_rtne(
         &builder, &normal_adds[component]->def);
      nir_def_rewrite_uses_after(&normal_adds[component]->def, rounded);
   }

   nir_opt_dce(nir);
   nir_progress(true, entrypoint, nir_metadata_control_flow);
   return true;
}


static bool
pvrgpu_validate_lit_mesh_variables(const nir_shader *nir,
                                   unsigned varying_components,
                                   const char *profile_name,
                                   char *error,
                                   size_t error_size)
{
   unsigned inputs = 0;
   unsigned outputs = 0;

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_in) {
      ++inputs;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if ((var->data.location != VERT_ATTRIB_GENERIC0 &&
              var->data.location != VERT_ATTRIB_GENERIC1) ||
             glsl_get_components(var->type) != 4) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "%s VS input ABI mismatch",
                                   profile_name);
         }
      } else if (var->data.location != VARYING_SLOT_VAR0 ||
                 glsl_get_components(var->type) != varying_components) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "%s FS varying ABI mismatch",
                                profile_name);
      }
   }

   nir_foreach_variable_with_modes (var, nir, nir_var_shader_out) {
      ++outputs;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if (var->data.location == VARYING_SLOT_POS) {
            if (glsl_get_components(var->type) != 4) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s VS position ABI mismatch",
                                      profile_name);
            }
         } else if (var->data.location != VARYING_SLOT_VAR0 ||
                    glsl_get_components(var->type) != varying_components) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "%s VS varying ABI mismatch",
                                   profile_name);
         }
      } else if ((var->data.location != FRAG_RESULT_COLOR &&
                  var->data.location != FRAG_RESULT_DATA0) ||
                 glsl_get_components(var->type) != 4) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "%s FS output ABI mismatch",
                                profile_name);
      }
   }

   const unsigned expected_inputs =
      nir->info.stage == MESA_SHADER_VERTEX ? 2U : 1U;
   const unsigned expected_outputs =
      nir->info.stage == MESA_SHADER_VERTEX ? 2U : 1U;
   if (inputs != expected_inputs || outputs != expected_outputs) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s %s I/O count mismatch (%u/%u)",
                             profile_name,
                             nir->info.stage == MESA_SHADER_VERTEX ? "VS" :
                                                                    "FS",
                             inputs,
                             outputs);
   }
   return true;
}

/* GLES mediump permits an implementation to retain fp32 precision.  Keep the
 * captured fp32 SSA values intact so the PCO backend and llvmpipe golden run
 * consume the same arithmetic; these gates validate the strict store shape
 * without injecting an implementation-specific f16 round trip. */
static bool
pvrgpu_validate_lit_mesh_varying_store(nir_shader *nir,
                                       unsigned varying_components,
                                       const char *profile_name,
                                       char *error,
                                       size_t error_size)
{
   unsigned stores = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_variable *var = nir_intrinsic_get_var(intr, 0);
            if (!var || var->data.mode != nir_var_shader_out ||
                var->data.location != VARYING_SLOT_VAR0)
               continue;
            if (stores++ || intr->src[1].ssa->bit_size != 32 ||
                intr->src[1].ssa->num_components != varying_components) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s floating-point varying store changed",
                                      profile_name);
            }
         }
      }
   }
   if (stores != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s requires one floating-point varying store",
                             profile_name);
   }
   return true;
}

/* Gallivm lowers the captured GLSL dot products as a z/y/x sequence of
 * individually rounded binary32 multiplies and adds.  PCO normally contracts
 * those adds back into FMADs.  Keep only the profile-selected reductions
 * split: applying prefers_split to the whole vertex shader also changes the
 * otherwise bit-exact position matrix arithmetic and can move triangle edges.
 */
static bool
pvrgpu_split_lit_mesh_dot3(nir_shader *nir,
                           unsigned split_mask,
                           unsigned expected_dots,
                           const char *profile_name,
                           char *error,
                           size_t error_size)
{
   unsigned dot_index = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *dot = nir_instr_as_alu(instr);
            if (dot->op != nir_op_fdot3)
               continue;

            const bool split = split_mask & BITFIELD_BIT(dot_index++);
            if (!split)
               continue;
            if (dot->def.bit_size != 32 || dot->def.num_components != 1) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s dot3 precision changed",
                                      profile_name);
            }

            nir_builder builder = nir_builder_at(nir_before_instr(instr));
            builder.fp_math_ctrl = dot->fp_math_ctrl | nir_fp_exact;
            nir_def *left = nir_ssa_for_alu_src(&builder, dot, 0);
            nir_def *right = nir_ssa_for_alu_src(&builder, dot, 1);
            if (!left || !right || left->num_components != 3 ||
                right->num_components != 3) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s dot3 sources changed",
                                      profile_name);
            }

            nir_def *sum = NULL;
            for (int component = 2; component >= 0; --component) {
               nir_def *product = nir_fmul(
                  &builder,
                  nir_channel(&builder, left, (unsigned)component),
                  nir_channel(&builder, right, (unsigned)component));
               sum = sum ? nir_fadd(&builder, product, sum) : product;
            }
            nir_def_replace(&dot->def, sum);
         }
      }
   }

   if (dot_index != expected_dots ||
       (split_mask & ~BITFIELD_MASK(dot_index))) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s dot3 graph changed (%u)",
                             profile_name,
                             dot_index);
   }
   return true;
}

static bool
pvrgpu_split_shadow_scene_dot2(nir_shader *nir,
                               const char *profile_name,
                               char *error,
                               size_t error_size)
{
   nir_alu_instr *lighting_dot = NULL;
   unsigned dots = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_fdot2) {
               lighting_dot = alu;
               ++dots;
            }
         }
      }
   }
   if (dots != 1U || !lighting_dot || lighting_dot->def.bit_size != 32 ||
       lighting_dot->def.num_components != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s lighting dot2 graph changed (%u)",
                             profile_name,
                             dots);
   }

   nir_builder builder =
      nir_builder_at(nir_before_instr(&lighting_dot->instr));
   builder.fp_math_ctrl = lighting_dot->fp_math_ctrl | nir_fp_exact;
   nir_def *left = nir_ssa_for_alu_src(&builder, lighting_dot, 0);
   nir_def *right = nir_ssa_for_alu_src(&builder, lighting_dot, 1);
   if (!left || !right || left->num_components != 2 ||
       right->num_components != 2) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s lighting dot2 sources changed",
                             profile_name);
   }

   nir_def *high = nir_fmul(&builder,
                            nir_channel(&builder, left, 1),
                            nir_channel(&builder, right, 1));
   nir_def *low = nir_fmul(&builder,
                           nir_channel(&builder, left, 0),
                           nir_channel(&builder, right, 0));
   nir_def *sum = nir_fadd(&builder, low, high);
   nir_def_replace(&lighting_dot->def, sum);
   return true;
}

static bool
pvrgpu_validate_lit_mesh_fragment_output(nir_shader *nir,
                                         const char *profile_name,
                                         char *error,
                                         size_t error_size)
{
   unsigned stores = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_variable *var = nir_intrinsic_get_var(intr, 0);
            if (!var || var->data.mode != nir_var_shader_out ||
                (var->data.location != FRAG_RESULT_COLOR &&
                 var->data.location != FRAG_RESULT_DATA0))
               continue;
            if (stores++ || intr->src[1].ssa->bit_size != 32 ||
                intr->src[1].ssa->num_components != 4) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s floating-point fragment store changed",
                                      profile_name);
            }
         }
      }
   }
   if (stores != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s requires one floating-point fragment store",
                             profile_name);
   }
   return true;
}

/* The texture fragment source is mediump, and Gallivm's GLES lowering keeps
 * the architectural half-precision boundaries visible in its generated IR:
 * all four UNORM sample channels convert to binary16 with RTZ, the scalar
 * lighting varying converts with RTNE, and each of the three RGB products is
 * rounded to binary16 with RTNE before conversion back to the fp32 fragment
 * output ABI.  Alpha is the RTZ-rounded sample alpha and is not multiplied.
 *
 * Preserve that operation-level graph explicitly.  This is intentionally
 * narrower than blanket fragment quantization: texture coordinates and the
 * interpolation machinery remain fp32, as do unrelated shader profiles. */
static bool
pvrgpu_lower_texture_fragment_mediump(nir_shader *nir,
                                      char *error,
                                      size_t error_size)
{
   nir_intrinsic_instr *varying_load = NULL;
   nir_intrinsic_instr *color_store = NULL;
   nir_tex_instr *sample = NULL;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_deref) {
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (var && var->data.mode == nir_var_shader_in &&
                      var->data.location == VARYING_SLOT_VAR0) {
                     if (varying_load)
                        return pvrgpu_pco_fail(
                           error,
                           error_size,
                           "texture mediump has duplicate varying load");
                     varying_load = intr;
                  }
               } else if (intr->intrinsic == nir_intrinsic_store_deref) {
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (var && var->data.mode == nir_var_shader_out &&
                      (var->data.location == FRAG_RESULT_COLOR ||
                       var->data.location == FRAG_RESULT_DATA0)) {
                     if (color_store)
                        return pvrgpu_pco_fail(
                           error,
                           error_size,
                           "texture mediump has duplicate color store");
                     color_store = intr;
                  }
               }
            } else if (instr->type == nir_instr_type_tex) {
               if (sample)
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "texture mediump has duplicate sample");
               sample = nir_instr_as_tex(instr);
            }
         }
      }
   }

   if (!varying_load || varying_load->def.bit_size != 32 ||
       varying_load->def.num_components != 3 || !sample ||
       sample->def.bit_size != 32 || sample->def.num_components != 4 ||
       !color_store || color_store->src[1].ssa->bit_size != 32 ||
       color_store->src[1].ssa->num_components != 4) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "texture mediump I/O graph changed");
   }

   nir_builder b = nir_builder_at(nir_before_instr(&color_store->instr));
   nir_def *light = nir_f2f32(
      &b,
      nir_f2f16_rtne(&b, nir_channel(&b, &varying_load->def, 0)));
   nir_def *sample_half[4];
   for (unsigned component = 0; component < 4; ++component) {
      sample_half[component] = nir_f2f32(
         &b,
         nir_f2f16_rtz(&b, nir_channel(&b, &sample->def, component)));
   }

   nir_def *color[4];
   for (unsigned component = 0; component < 3; ++component) {
      nir_def *product = nir_fmul(&b, sample_half[component], light);
      color[component] =
         nir_f2f32(&b, nir_f2f16_rtne(&b, product));
   }
   color[3] = sample_half[3];
   nir_src_rewrite(&color_store->src[1], nir_vec(&b, color, 4));

   nir_progress(true,
                nir_shader_get_entrypoint(nir),
                nir_metadata_control_flow);
   return true;
}

/* The bump fragment source has a default mediump float precision.  Its vertex
 * shader intentionally remains fp32, including the smooth varying producer,
 * while the fragment consumer and every fragment arithmetic result round to
 * binary16.  The incoming and outgoing ABI is still fp32, so express each
 * fragment precision boundary explicitly as an RTNE f16 round trip before the
 * PCO backend lowers the fail-closed graph. */
static bool
pvrgpu_lower_bump_fragment_mediump(nir_shader *nir,
                                   char *error,
                                   size_t error_size)
{
   nir_lower_alu(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);

   /* PCO expands fpow after this profile pass.  Expand the single captured
    * power here so Gallivm's mediump boundaries are applied independently to
    * log2, exponent multiplication, and exp2 rather than only to the result. */
   unsigned powers = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op != nir_op_fpow)
               continue;
            if (powers++ || alu->def.bit_size != 32 ||
                alu->def.num_components != 1) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "bump mediump power graph changed");
            }

            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_def *base = nir_ssa_for_alu_src(&b, alu, 0);
            nir_def *exponent = nir_ssa_for_alu_src(&b, alu, 1);
            nir_def *zero = nir_imm_float(&b, 0.0f);
            nir_def *is_zero = nir_feq(&b, base, zero);
            nir_def *logarithm = nir_flog2(&b, base);
            nir_def *scaled = nir_fmul(&b, logarithm, exponent);
            nir_def *power = nir_fexp2(&b, scaled);
            nir_def *selected = nir_bcsel(&b, is_zero, zero, power);
            nir_def_replace(&alu->def, selected);
         }
      }
   }
   if (powers != 1U) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "bump mediump power graph is missing");
   }

   unsigned varying_loads = 0;
   unsigned stores = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            nir_def *value = NULL;
            if (instr->type == nir_instr_type_load_const) {
               nir_load_const_instr *constant =
                  nir_instr_as_load_const(instr);
               if (constant->def.bit_size != 32) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "bump fragment constant precision changed");
               }
               value = &constant->def;
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_deref) {
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (!var || var->data.mode != nir_var_shader_in ||
                      var->data.location != VARYING_SLOT_VAR0 ||
                      varying_loads++ || intr->def.bit_size != 32 ||
                      intr->def.num_components != 3) {
                     return pvrgpu_pco_fail(
                        error,
                        error_size,
                        "bump fragment varying load changed");
                  }
                  value = &intr->def;
               } else if (intr->intrinsic == nir_intrinsic_store_deref) {
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (!var || var->data.mode != nir_var_shader_out ||
                      (var->data.location != FRAG_RESULT_COLOR &&
                       var->data.location != FRAG_RESULT_DATA0))
                     continue;
                  if (stores++ || intr->src[1].ssa->bit_size != 32 ||
                      intr->src[1].ssa->num_components != 4) {
                     return pvrgpu_pco_fail(
                        error,
                        error_size,
                        "bump fragment result assignment changed");
                  }
               }
            } else if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               switch (alu->op) {
               case nir_op_fadd:
               case nir_op_fmul:
               case nir_op_fmax:
               case nir_op_frcp:
               case nir_op_frsq:
               case nir_op_flog2:
               case nir_op_fexp2:
                  if (alu->def.bit_size != 32) {
                     return pvrgpu_pco_fail(
                        error,
                        error_size,
                        "bump fragment arithmetic precision changed");
                  }
                  value = &alu->def;
                  break;
               case nir_op_ffma:
               case nir_op_ffma_weak:
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "bump fragment fused arithmetic changed");
               default:
                  break;
               }
            }

            if (!value)
               continue;
            nir_builder b = nir_builder_at(nir_after_instr(instr));
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == nir_op_frsq) {
               /* LLVM's f16 frsq is a rounded half sqrt followed by a rounded
                * half reciprocal.  Preserve both architectural boundaries
                * while using the public fp32 Rogue operations. */
               nir_def *sqrt_value = nir_frcp(&b, value);
               nir_def *sqrt_rounded =
                  nir_f2f32(&b, nir_f2f16_rtne(&b, sqrt_value));
               nir_def *reciprocal = nir_frcp(&b, sqrt_rounded);
               nir_def *rounded =
                  nir_f2f32(&b, nir_f2f16_rtne(&b, reciprocal));
               nir_def_rewrite_uses_after(value, rounded);
               continue;
            }
            nir_def *half =
               instr->type == nir_instr_type_load_const
                  ? nir_f2f16_rtz(&b, value)
                  : nir_f2f16_rtne(&b, value);
            nir_def *rounded = nir_f2f32(&b, half);
            nir_def_rewrite_uses_after(value, rounded);
         }
      }
   }
   if (varying_loads != 1 || stores != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "bump mediump I/O signature changed");
   }
   return true;
}

/* Gallivm lowers the strict Refract composite fragment shader to an fp16
 * arithmetic graph even though the Gallium-facing NIR and the PCO fragment
 * ABI are fp32.  Keep those two interfaces unchanged and make each
 * architectural half boundary explicit on the private compile clone:
 *
 *  - interpolated mediump varyings enter the graph through RTNE f16;
 *  - every arithmetic result is rounded to f16 with RTNE;
 *  - normalized texture results enter the graph through RTZ f16; and
 *  - texture coordinates and the final color leave the graph as fp32.
 *
 * The source signature is validated before this profile-local pass runs.  In
 * particular, this is not a generic fragment-output quantizer: fdot is first
 * expanded in Gallivm's reverse component order, fpow exposes its half
 * log2/multiply/exp2 boundaries, and half frsq preserves the rounded sqrt and
 * reciprocal steps observed in the reference NIR/LLVM pipeline. */
static bool
pvrgpu_lower_refract_composite_fragment_mediump(nir_shader *nir,
                                                 char *error,
                                                 size_t error_size)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract mediump requires a fragment shader");
   }

   nir_function_impl *entrypoint = NULL;
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      if (entrypoint) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "refract mediump requires one entrypoint");
      }
      entrypoint = function->impl;
   }
   if (!entrypoint) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract mediump entrypoint is missing");
   }

   /* The captured Gallium NIR distributes the source-level scalar
    * `2.0 * dot(normal, view)` into a vec3 splat multiply.  If that form were
    * scalarized directly, it would introduce three independent fp16 rounding
    * boundaries.  Gallivm's mediump NIR instead computes the scalar multiply
    * once and then uses it for all three normal components.  Recognize only
    * that exact dot3-times-2.0 splat and restore its source-level scalar
    * boundary before general ALU scalarization. */
   unsigned reconstructed_reflection_scales = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_fmul || alu->def.bit_size != 32 ||
             alu->def.num_components != 3)
            continue;

         bool splat[2] = { true, true };
         for (unsigned src = 0; src < 2; ++src) {
            for (unsigned component = 1; component < 3; ++component) {
               if (alu->src[src].swizzle[component] !=
                   alu->src[src].swizzle[0])
                  splat[src] = false;
            }
         }
         if (!splat[0] || !splat[1])
            continue;

         uint64_t src_bits[2] = { 0, 0 };
         const bool src_is_const[2] = {
            nir_alu_src_comp_get_uint(alu->src[0], 0, &src_bits[0]),
            nir_alu_src_comp_get_uint(alu->src[1], 0, &src_bits[1]),
         };
         int factor_src = -1;
         if (src_is_const[0] && src_bits[0] == UINT32_C(0x40000000))
            factor_src = 0;
         if (src_is_const[1] && src_bits[1] == UINT32_C(0x40000000)) {
            if (factor_src >= 0)
               continue;
            factor_src = 1;
         }
         if (factor_src < 0)
            continue;

         const unsigned dot_src = 1U - (unsigned)factor_src;
         nir_def *dot_def = alu->src[dot_src].src.ssa;
         if (!dot_def || dot_def->bit_size != 32 ||
             dot_def->num_components != 1 ||
             nir_def_instr(dot_def)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(dot_def))->op != nir_op_fdot3) {
            continue;
         }
         if (reconstructed_reflection_scales++) {
            return pvrgpu_pco_fail(
               error,
               error_size,
               "refract mediump reflection scale is ambiguous");
         }

         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_def *dot = nir_channel(&b,
                                    dot_def,
                                    alu->src[dot_src].swizzle[0]);
         nir_def *factor = nir_channel(
            &b,
            alu->src[factor_src].src.ssa,
            alu->src[factor_src].swizzle[0]);
         nir_def *scalar_scale = nir_fmul(&b, dot, factor);
         nir_def_replace(&alu->def,
                         nir_replicate(&b, scalar_scale, 3));
      }
   }
   if (reconstructed_reflection_scales != 1) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "refract mediump reflection scale graph changed");
   }

   /* This expands each fdot3 as z*z + y*y + x*x, matching Gallivm's lowered
    * mediump NIR.  The profile's prefers_split compiler option keeps the
    * products and adds distinct. */
   nir_lower_alu_to_scalar(nir, NULL, NULL);

   /* PCO normally lowers fpow after this profile pass.  Expose the reference
    * implementation here so the three fp16 operation boundaries are not
    * collapsed into one final conversion. */
   unsigned powers = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_fpow)
            continue;
         if (powers++ || alu->def.bit_size != 32 ||
             alu->def.num_components != 1) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "refract mediump power graph changed");
         }

         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_def *base = nir_ssa_for_alu_src(&b, alu, 0);
         nir_def *exponent = nir_ssa_for_alu_src(&b, alu, 1);
         nir_def *zero = nir_imm_float(&b, 0.0f);
         nir_def *is_zero = nir_feq(&b, base, zero);
         nir_def *logarithm = nir_flog2(&b, base);
         nir_def *scaled = nir_fmul(&b, logarithm, exponent);
         nir_def *power = nir_fexp2(&b, scaled);
         nir_def *selected = nir_bcsel(&b, is_zero, zero, power);
         nir_def_replace(&alu->def, selected);
      }
   }
   if (powers != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract mediump power graph is missing");
   }

   unsigned input_mask = 0;
   unsigned input_loads = 0;
   unsigned textures = 0;
   unsigned stores = 0;
   unsigned rounded_arithmetic = 0;
   unsigned rounded_constants = 0;
   unsigned reconstructed_constants = 0;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr_safe (instr, block) {
         nir_def *value = NULL;
         bool texture_value = false;
         bool reconstruct_inverse_ior_squared = false;

         switch (instr->type) {
         case nir_instr_type_load_const: {
            nir_load_const_instr *constant =
               nir_instr_as_load_const(instr);
            if (constant->def.bit_size != 32) {
               return pvrgpu_pco_fail(
                  error,
                  error_size,
                  "refract mediump constant precision changed");
            }
            value = &constant->def;
            /* The GLES precision pass sees (1.0 / 1.2)^2 before the fp32
             * optimizer folds it to 0x3f31c71c.  It therefore rounds 1/1.2
             * to half first and folds the half product to 0x398f.  Rebuild
             * that one source-level provenance boundary explicitly. */
            if (constant->def.num_components == 1 &&
                constant->value[0].u32 == UINT32_C(0x3f31c71c)) {
               reconstruct_inverse_ior_squared = true;
               ++reconstructed_constants;
            }
            ++rounded_constants;
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               nir_variable *var = nir_intrinsic_get_var(intr, 0);
               if (!var || var->data.mode != nir_var_shader_in ||
                   var->data.location < VARYING_SLOT_VAR0 ||
                   var->data.location > VARYING_SLOT_VAR2 ||
                   intr->def.bit_size != 32) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "refract mediump varying load changed");
               }
               const unsigned index =
                  var->data.location - VARYING_SLOT_VAR0;
               const unsigned expected_components = index < 2 ? 4 : 3;
               if ((input_mask & BITFIELD_BIT(index)) ||
                   intr->def.num_components != expected_components) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "refract mediump varying signature changed");
               }
               input_mask |= BITFIELD_BIT(index);
               ++input_loads;
               value = &intr->def;
            } else if (intr->intrinsic == nir_intrinsic_store_deref) {
               nir_variable *var = nir_intrinsic_get_var(intr, 0);
               if (!var || var->data.mode != nir_var_shader_out ||
                   (var->data.location != FRAG_RESULT_COLOR &&
                    var->data.location != FRAG_RESULT_DATA0))
                  continue;
               if (stores++ || intr->src[1].ssa->bit_size != 32 ||
                   intr->src[1].ssa->num_components != 4) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "refract mediump fragment store changed");
               }
            }
            break;
         }
         case nir_instr_type_tex: {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->texture_index >= PVRGPU_REFRACT_TEXTURE_COUNT ||
                tex->sampler_index != tex->texture_index ||
                (textures & BITFIELD_BIT(tex->texture_index)) ||
                tex->def.bit_size != 32 || tex->def.num_components != 4) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "refract mediump texture graph changed");
            }
            textures |= BITFIELD_BIT(tex->texture_index);
            value = &tex->def;
            texture_value = true;
            break;
         }
         case nir_instr_type_alu: {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            switch (alu->op) {
            case nir_op_fadd:
            case nir_op_fmul:
            case nir_op_fdiv:
            case nir_op_fsqrt:
            case nir_op_flog2:
            case nir_op_fexp2:
               if (alu->def.bit_size != 32) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "refract mediump arithmetic precision changed");
               }
               value = &alu->def;
               ++rounded_arithmetic;
               break;
            case nir_op_frsq: {
               if (alu->def.bit_size != 32 ||
                   alu->def.num_components != 1) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "refract mediump reciprocal sqrt changed");
               }
               /* LLVM implements half frsq as a rounded half sqrt followed
                * by a rounded half division of 1.0 by that sqrt. */
               nir_builder b = nir_builder_at(nir_after_instr(instr));
               nir_def *sqrt_value = nir_frcp(&b, &alu->def);
               nir_def *sqrt_rounded = nir_f2f32(
                  &b, nir_f2f16_rtne(&b, sqrt_value));
               nir_def *reciprocal = nir_frcp(&b, sqrt_rounded);
               nir_def *rounded = nir_f2f32(
                  &b, nir_f2f16_rtne(&b, reciprocal));
               nir_def_rewrite_uses_after(&alu->def, rounded);
               ++rounded_arithmetic;
               continue;
            }
            case nir_op_fneg:
            case nir_op_fmax:
            case nir_op_flt:
            case nir_op_feq:
            case nir_op_bcsel:
            case nir_op_mov:
            case nir_op_vec2:
            case nir_op_vec3:
            case nir_op_vec4:
               /* These either produce a Boolean or select/rearrange an
                * already representable half value without new rounding. */
               break;
            case nir_op_ffma:
            case nir_op_ffma_weak:
            case nir_op_fdot3:
            case nir_op_fpow:
               return pvrgpu_pco_fail(
                  error,
                  error_size,
                  "refract mediump retained fused arithmetic %s",
                  nir_op_infos[alu->op].name);
            default:
               return pvrgpu_pco_fail(
                  error,
                  error_size,
                  "refract mediump contains unsupported arithmetic %s",
                  nir_op_infos[alu->op].name);
            }
            break;
         }
         case nir_instr_type_deref:
            break;
         default:
            return pvrgpu_pco_fail(
               error,
               error_size,
               "refract mediump contains unsupported instruction type %u",
               instr->type);
         }

         if (!value)
            continue;

         nir_builder b = nir_builder_at(nir_after_instr(instr));
         nir_def *rounded;
         if (texture_value) {
            nir_def *channels[4];
            for (unsigned component = 0; component < 4; ++component) {
               channels[component] = nir_f2f32(
                  &b,
                  nir_f2f16_rtz(
                     &b, nir_channel(&b, value, component)));
            }
            rounded = nir_vec(&b, channels, 4);
         } else {
            nir_def *round_source = value;
            if (reconstruct_inverse_ior_squared) {
               round_source =
                  nir_imm_float(&b, 0.69482421875f); /* binary16 0x398f */
            }
            rounded = nir_f2f32(
               &b, nir_f2f16_rtne(&b, round_source));
         }
         nir_def_rewrite_uses_after(value, rounded);
      }
   }

   if (input_loads != 3 || input_mask != 0x7 || textures != 0x7 ||
       stores != 1 || rounded_constants != 13 ||
       rounded_arithmetic != 127 || reconstructed_constants != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "refract mediump lowered graph signature changed "
                             "(inputs=%u/%x textures=%x stores=%u constants=%u "
                             "arithmetic=%u reconstructed=%u)",
                             input_loads,
                             input_mask,
                             textures,
                             stores,
                             rounded_constants,
                             rounded_arithmetic,
                             reconstructed_constants);
   }

   nir_progress(true, entrypoint, nir_metadata_control_flow);
   return true;
}

static void
pvrgpu_init_lit_mesh_shader_data(pco_data *vertex_data,
                                 pco_data *fragment_data,
                                 unsigned varying_components)
{
   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC0] =
      PIPE_FORMAT_R32G32B32_FLOAT;
   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC1] =
      PIPE_FORMAT_R32G32B32_FLOAT;
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC0] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC1] = (pco_range){
      .start = 4,
      .count = 4,
   };
   vertex_data->common.vtxins = 8;
   vertex_data->vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->vs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = 4,
      .count = varying_components,
   };
   vertex_data->vs.vtxouts = 4 + varying_components;
   vertex_data->vs.f32_smooth = varying_components;

   /* Smooth interpolation consumes one four-DWORD W plane followed by one
    * four-DWORD coefficient set per scalar varying component. */
   fragment_data->fs.uses.w = true;
   fragment_data->fs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   fragment_data->fs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = 4,
      .count = varying_components * 4,
   };
   fragment_data->common.coeffs = 4 + varying_components * 4;
   fragment_data->fs.z_replicate = ~0U;
   fragment_data->fs.rasterization_samples = 1;
}

static bool
pvrgpu_init_texture_shader_data(pco_data *vertex_data,
                                pco_data *fragment_data,
                                void *compile_mem_ctx,
                                char *error,
                                size_t error_size)
{
   pvrgpu_init_lit_mesh_shader_data(vertex_data, fragment_data, 3);
   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC2] =
      PIPE_FORMAT_R32G32_FLOAT;
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC2] = (pco_range){
      .start = 8,
      .count = 4,
   };
   vertex_data->common.vtxins = 12;

   pco_descriptor_set_data *set = &fragment_data->common.desc_sets[0];
   set->binding_count = 1;
   set->bindings = rzalloc_array(compile_mem_ctx, pco_binding_data, 1);
   if (!set->bindings)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "out of memory allocating texture descriptor ABI");

   set->range = (pco_range){
      .start = 0,
      .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
   };
   set->used = true;
   set->bindings[0].range = (pco_range){
      .start = 0,
      .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
      .stride = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
   };
   set->bindings[0].used = true;
   set->bindings[0].is_img_smp = true;
   fragment_data->common.shareds = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   return true;
}

static bool
pvrgpu_init_refract_shader_data(enum pvrgpu_pco_refract_profile profile,
                                pco_data *vertex_data,
                                pco_data *fragment_data,
                                void *compile_mem_ctx,
                                char *error,
                                size_t error_size)
{
   if (profile == PVRGPU_PCO_REFRACT_PREPASS) {
      pvrgpu_init_lit_mesh_shader_data(vertex_data, fragment_data, 3);
      return true;
   }

   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC0] =
      PIPE_FORMAT_R32G32B32_FLOAT;
   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC1] =
      PIPE_FORMAT_R32G32B32_FLOAT;
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC0] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC1] = (pco_range){
      .start = 4,
      .count = 4,
   };
   vertex_data->common.vtxins = 8;
   vertex_data->vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->vs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = 4,
      .count = 4,
   };
   vertex_data->vs.varyings[VARYING_SLOT_VAR1] = (pco_range){
      .start = 8,
      .count = 4,
   };
   vertex_data->vs.varyings[VARYING_SLOT_VAR2] = (pco_range){
      .start = 12,
      .count = 3,
   };
   vertex_data->vs.vtxouts = 15;
   vertex_data->vs.f32_smooth = 11;

   fragment_data->fs.uses.w = true;
   fragment_data->fs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   fragment_data->fs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = 4,
      .count = 16,
   };
   fragment_data->fs.varyings[VARYING_SLOT_VAR1] = (pco_range){
      .start = 20,
      .count = 16,
   };
   fragment_data->fs.varyings[VARYING_SLOT_VAR2] = (pco_range){
      .start = 36,
      .count = 12,
   };
   fragment_data->common.coeffs = 48;
   fragment_data->fs.z_replicate = ~0U;
   fragment_data->fs.rasterization_samples = 1;

   /* Gallium NIR represents texture unit N as descriptor set N, binding 0.
    * Keep the three image/sampler descriptors contiguous in shared storage,
    * while describing that set layout exactly to PCO lowering. */
   for (unsigned texture = 0; texture < PVRGPU_REFRACT_TEXTURE_COUNT;
        ++texture) {
      pco_descriptor_set_data *set =
         &fragment_data->common.desc_sets[texture];
      set->binding_count = 1;
      set->bindings = rzalloc_array(compile_mem_ctx, pco_binding_data, 1);
      if (!set->bindings) {
         return pvrgpu_pco_fail(
            error,
            error_size,
            "out of memory allocating refract descriptor set %u",
            texture);
      }
      set->range = (pco_range){
         .start = texture * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
         .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
      };
      set->used = true;
      set->bindings[0].range = (pco_range){
         .start = texture * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
         .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
         .stride = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
      };
      set->bindings[0].used = true;
      set->bindings[0].is_img_smp = true;
   }
   fragment_data->common.shareds = PVRGPU_REFRACT_DESCRIPTOR_DWORDS;
   return true;
}

static bool
pvrgpu_init_shadow_shader_data(enum pvrgpu_pco_shadow_profile profile,
                               pco_data *vertex_data,
                               pco_data *fragment_data,
                               void *compile_mem_ctx,
                               char *error,
                               size_t error_size)
{
   if (profile == PVRGPU_PCO_SHADOW_SCENE) {
      pvrgpu_init_lit_mesh_shader_data(vertex_data, fragment_data, 1);
      return true;
   }
   if (profile != PVRGPU_PCO_SHADOW_MASK)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "invalid non-depth shadow shader-data profile");

   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC0] =
      PIPE_FORMAT_R32G32_FLOAT;
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC0] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->common.vtxins = 4;
   vertex_data->vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->vs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = 4,
      .count = 4,
   };
   vertex_data->vs.vtxouts = 8;
   vertex_data->vs.f32_smooth = 4;

   fragment_data->fs.uses.w = true;
   fragment_data->fs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   fragment_data->fs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = 4,
      .count = 16,
   };
   fragment_data->common.coeffs = 20;
   fragment_data->fs.z_replicate = ~0U;
   fragment_data->fs.rasterization_samples = 1;

   pco_descriptor_set_data *set = &fragment_data->common.desc_sets[0];
   set->binding_count = 1;
   set->bindings = rzalloc_array(compile_mem_ctx, pco_binding_data, 1);
   if (!set->bindings)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "out of memory allocating shadow descriptor ABI");
   set->range = (pco_range){
      .start = 0,
      .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
   };
   set->used = true;
   set->bindings[0].range = (pco_range){
      .start = 0,
      .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
      .stride = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
   };
   set->bindings[0].used = true;
   set->bindings[0].is_img_smp = true;
   fragment_data->common.shareds = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   return true;
}

static bool
pvrgpu_init_terrain_descriptors(pco_data *data,
                                unsigned texture_count,
                                void *compile_mem_ctx,
                                const char *profile_name,
                                const char *stage,
                                char *error,
                                size_t error_size)
{
   if (!texture_count)
      return true;
   pco_descriptor_set_data *set = &data->common.desc_sets[0];
   set->binding_count = texture_count;
   set->bindings =
      rzalloc_array(compile_mem_ctx, pco_binding_data, texture_count);
   if (!set->bindings) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "out of memory allocating %s %s descriptors",
                             profile_name,
                             stage);
   }
   set->range = (pco_range){
      .start = 0,
      .count = texture_count * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
   };
   set->used = true;
   for (unsigned texture = 0; texture < texture_count; ++texture) {
      set->bindings[texture].range = (pco_range){
         .start = texture * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
         .count = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
         .stride = PVRGPU_TEXTURE_DESCRIPTOR_DWORDS,
      };
      set->bindings[texture].used = true;
      set->bindings[texture].is_img_smp = true;
   }
   data->common.shareds =
      texture_count * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   return true;
}

static bool
pvrgpu_init_terrain_shader_data(enum pvrgpu_pco_terrain_profile profile,
                                pco_data *vertex_data,
                                pco_data *fragment_data,
                                void *compile_mem_ctx,
                                char *error,
                                size_t error_size)
{
   const struct pvrgpu_terrain_pco_desc *desc =
      &pvrgpu_terrain_profiles[profile];
   const bool main = profile == PVRGPU_PCO_TERRAIN_D3;
   const unsigned vertex_inputs = main ? 4U : 1U;
   for (unsigned input = 0; input < vertex_inputs; ++input) {
      const enum pipe_format format =
         main && input == 3U ? PIPE_FORMAT_R32G32_FLOAT :
                               PIPE_FORMAT_R32G32B32_FLOAT;
      const unsigned start = input * 4U;
      vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC0 + input] = format;
      vertex_data->vs.attribs[VERT_ATTRIB_GENERIC0 + input] = (pco_range){
         .start = start,
         .count = 4,
      };
   }
   vertex_data->common.vtxins = vertex_inputs * 4U;
   vertex_data->vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   if (main) {
      vertex_data->vs.varyings[VARYING_SLOT_VAR0] = (pco_range){ 4, 4 };
      vertex_data->vs.varyings[VARYING_SLOT_VAR1] = (pco_range){ 8, 4 };
      vertex_data->vs.varyings[VARYING_SLOT_VAR2] = (pco_range){ 12, 4 };
      vertex_data->vs.varyings[VARYING_SLOT_VAR3] = (pco_range){ 16, 2 };
      vertex_data->vs.vtxouts = 18;
      vertex_data->vs.f32_smooth = 14;
   } else {
      vertex_data->vs.varyings[VARYING_SLOT_VAR0] = (pco_range){ 4, 2 };
      vertex_data->vs.vtxouts = 6;
      vertex_data->vs.f32_smooth = 2;
   }

   fragment_data->fs.uses.w = true;
   fragment_data->fs.varyings[VARYING_SLOT_POS] = (pco_range){ 0, 4 };
   if (main) {
      fragment_data->fs.varyings[VARYING_SLOT_VAR0] = (pco_range){ 4, 16 };
      fragment_data->fs.varyings[VARYING_SLOT_VAR1] = (pco_range){ 20, 16 };
      fragment_data->fs.varyings[VARYING_SLOT_VAR2] = (pco_range){ 36, 16 };
      fragment_data->fs.varyings[VARYING_SLOT_VAR3] = (pco_range){ 52, 8 };
   } else {
      fragment_data->fs.varyings[VARYING_SLOT_VAR0] = (pco_range){ 4, 8 };
   }
   fragment_data->common.coeffs = 4 + desc->varying_components * 4U;
   fragment_data->fs.z_replicate = ~0U;
   fragment_data->fs.rasterization_samples = 1;

   return pvrgpu_init_terrain_descriptors(vertex_data,
                                           desc->vertex_texture_count,
                                           compile_mem_ctx,
                                           desc->name,
                                           "VS",
                                           error,
                                           error_size) &&
          pvrgpu_init_terrain_descriptors(fragment_data,
                                           desc->fragment_texture_count,
                                           compile_mem_ctx,
                                           desc->name,
                                           "FS",
                                           error,
                                           error_size);
}

static bool
pvrgpu_pack_terrain_texture_bindings(nir_shader *nir,
                                     unsigned texture_count,
                                     unsigned expected_texture_ops,
                                     const char *profile_name,
                                     char *error,
                                     size_t error_size)
{
   /* The driver command packs sampled resources in first-use order.  PCO
    * descriptor sets must use the same order; raw GLSL binding numbers are
    * not necessarily monotonic (Terrain D3 uses VS {1, 0} and FS
    * {2, 0, 1, 4, 3}). */
   uint16_t binding_to_set[9];
   if (texture_count > ARRAY_SIZE(binding_to_set)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s has too many unique texture bindings",
                             profile_name);
   }
   for (unsigned binding = 0; binding < ARRAY_SIZE(binding_to_set);
        ++binding)
      binding_to_set[binding] = UINT16_MAX;

   unsigned rewritten = 0;
   unsigned unique_bindings = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->texture_index != tex->sampler_index ||
                tex->texture_index >= texture_count ||
                tex->texture_index > UINT16_MAX) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s texture binding is out of range",
                                      profile_name);
            }
            const unsigned binding = tex->texture_index;
            if (binding_to_set[binding] == UINT16_MAX)
               binding_to_set[binding] = unique_bindings++;
            const uint32_t packed = binding_to_set[binding] << 16U;
            tex->texture_index = packed;
            tex->sampler_index = packed;
            ++rewritten;
         }
      }
      nir_progress(rewritten != 0, impl, nir_metadata_control_flow);
   }
   if (rewritten != expected_texture_ops ||
       unique_bindings != texture_count) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "%s rewrote %u texture ops/%u bindings, "
                             "expected %u/%u",
                             profile_name,
                             rewritten,
                             unique_bindings,
                             expected_texture_ops,
                             texture_count);
   }
   return true;
}

bool pvrgpu_pco_compile_lit_mesh(
   struct pvrgpu_pco_compiler *compiler,
   const nir_shader *vertex_nir,
   const nir_shader *fragment_nir,
   enum pvrgpu_pco_lit_mesh_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));

   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir ||
       profile < PVRGPU_PCO_LIT_MESH_BUILD ||
       profile > PVRGPU_PCO_LIT_MESH_SHADING) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "missing compiler or invalid lit-mesh profile");
   }
   const struct pvrgpu_lit_mesh_desc *desc =
      &pvrgpu_lit_mesh_profiles[profile];
   if (vertex_nir->info.stage != MESA_SHADER_VERTEX ||
       fragment_nir->info.stage != MESA_SHADER_FRAGMENT ||
       !pvrgpu_source_hash_matches(vertex_nir, desc->vertex_source_hash) ||
       !pvrgpu_source_hash_matches(fragment_nir, desc->fragment_source_hash) ||
       !pvrgpu_validate_lit_mesh_variables(vertex_nir,
                                           desc->varying_components,
                                           desc->name,
                                           error,
                                           error_size) ||
       !pvrgpu_validate_lit_mesh_variables(fragment_nir,
                                           desc->varying_components,
                                           desc->name,
                                           error,
                                           error_size)) {
      if (error && error_size && error[0] == '\0')
         snprintf(error, error_size, "%s NIR source signature mismatch", desc->name);
      return false;
   }

   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");
   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "failed to clone %s NIR",
                             desc->name);
   }

   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = pco_nir_options();
   fs->options = pco_nir_options();
   nir_shader_compiler_options bump_fragment_options;
   if (profile == PVRGPU_PCO_LIT_MESH_BUMP) {
      bump_fragment_options = *pco_nir_options();
      bump_fragment_options.float_mul_add32 &=
         ~nir_float_muladd_support_fuse;
      bump_fragment_options.float_mul_add32 |=
         nir_float_muladd_support_prefers_split;
      fs->options = &bump_fragment_options;
   }
   nir_lower_fragcolor(fs, 1);

   if (!pvrgpu_validate_lit_mesh_varying_store(vs,
                                               desc->varying_components,
                                               desc->name,
                                               error,
                                               error_size) ||
       ((profile == PVRGPU_PCO_LIT_MESH_BUILD ||
         profile == PVRGPU_PCO_LIT_MESH_BUMP) &&
        !pvrgpu_split_lit_mesh_dot3(
           vs,
           BITFIELD_BIT(0),
           profile == PVRGPU_PCO_LIT_MESH_BUILD ? 2U : 1U,
           desc->name,
           error,
           error_size)) ||
       !pvrgpu_validate_lit_mesh_fragment_output(fs,
                                                 desc->name,
                                                 error,
                                                 error_size) ||
       (profile == PVRGPU_PCO_LIT_MESH_BUMP &&
        !pvrgpu_lower_bump_fragment_mediump(fs, error, error_size)) ||
       !pvrgpu_canonicalize_fragment_output(fs,
                                            desc->name,
                                            error,
                                            error_size) ||
       !pvrgpu_lower_uniform_slots_to_push_constants(
          vs,
          PVRGPU_LIT_MESH_VS_UNIFORM_DWORDS,
          8,
          desc->name,
          error,
          error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   pvrgpu_init_lit_mesh_shader_data(&vertex_data,
                                    &fragment_data,
                                    desc->varying_components);

   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);
   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   if (!pvrgpu_allocate_push_constants(&vertex_data,
                                       PVRGPU_LIT_MESH_VS_UNIFORM_DWORDS,
                                       31,
                                       desc->name,
                                       "VS",
                                       error,
                                       error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate %s NIR",
                             desc->name);
   }
   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);

   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment,
                              false,
                              &out->fragment,
                              error,
                              error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   out->position_output_start =
      vertex_data.vs.varyings[VARYING_SLOT_POS].start;
   out->position_output_count =
      vertex_data.vs.varyings[VARYING_SLOT_POS].count;
   out->fragment_position_start =
      fragment_data.fs.varyings[VARYING_SLOT_POS].start;
   out->fragment_position_count =
      fragment_data.fs.varyings[VARYING_SLOT_POS].count;
   out->varying_output_start =
      vertex_data.vs.varyings[VARYING_SLOT_VAR0].start;
   out->varying_output_count =
      vertex_data.vs.varyings[VARYING_SLOT_VAR0].count;
   out->fragment_varying_start =
      fragment_data.fs.varyings[VARYING_SLOT_VAR0].start;
   out->fragment_varying_count =
      fragment_data.fs.varyings[VARYING_SLOT_VAR0].count;

   if (out->vertex.abi.vertex_inputs != 8 ||
       out->vertex.abi.vertex_outputs != 4 + desc->varying_components ||
       out->vertex.abi.shareds != PVRGPU_LIT_MESH_VS_UNIFORM_DWORDS ||
       out->vertex.abi.push_constant_count !=
          PVRGPU_LIT_MESH_VS_UNIFORM_DWORDS ||
       out->fragment.abi.shareds != 0 ||
       out->fragment.abi.push_constant_count != 0 ||
       out->position_output_start != 0 || out->position_output_count != 4 ||
       out->varying_output_start != 4 ||
       out->varying_output_count != desc->varying_components ||
       out->fragment_position_start != 0 ||
       out->fragment_position_count != 4 ||
       out->fragment_varying_start != 4 ||
       out->fragment_varying_count != desc->varying_components * 4) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "compiled %s PCO ABI changed",
                             desc->name);
   }

   ralloc_free(compile_mem_ctx);
   return true;
}

bool pvrgpu_pco_compile_texture(
   struct pvrgpu_pco_compiler *compiler,
   const nir_shader *vertex_nir,
   const nir_shader *fragment_nir,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));

   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "missing compiler or texture NIR stage");
   }
   if (!pvrgpu_source_hash_matches(vertex_nir,
                                   pvrgpu_texture_vertex_source_hash) ||
       !pvrgpu_source_hash_matches(fragment_nir,
                                   pvrgpu_texture_fragment_source_hash) ||
       !pvrgpu_validate_texture_nir(vertex_nir,
                                    MESA_SHADER_VERTEX,
                                    error,
                                    error_size) ||
       !pvrgpu_validate_texture_nir(fragment_nir,
                                    MESA_SHADER_FRAGMENT,
                                    error,
                                    error_size)) {
      if (error && error_size && error[0] == '\0')
         snprintf(error, error_size, "texture NIR source signature mismatch");
      return false;
   }

   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");
   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "failed to clone texture NIR");
   }

   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = pco_nir_options();
   fs->options = pco_nir_options();
   nir_lower_fragcolor(fs, 1);

   if (!pvrgpu_validate_lit_mesh_varying_store(vs,
                                               3,
                                               "texture",
                                               error,
                                               error_size) ||
       !pvrgpu_validate_lit_mesh_fragment_output(fs,
                                                 "texture",
                                                 error,
                                                 error_size) ||
       !pvrgpu_lower_texture_fragment_mediump(fs, error, error_size) ||
       !pvrgpu_canonicalize_fragment_output(fs,
                                            "texture",
                                            error,
                                            error_size) ||
       !pvrgpu_lower_uniform_slots_to_push_constants(
          vs,
          PVRGPU_TEXTURE_VS_UNIFORM_DWORDS,
          8,
          "texture",
          error,
          error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   if (!pvrgpu_init_texture_shader_data(&vertex_data,
                                        &fragment_data,
                                        compile_mem_ctx,
                                        error,
                                        error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);
   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   if (!pvrgpu_allocate_push_constants(&vertex_data,
                                       PVRGPU_TEXTURE_VS_UNIFORM_DWORDS,
                                       31,
                                       "texture",
                                       "VS",
                                       error,
                                       error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate texture NIR");
   }
   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);

   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment,
                              false,
                              &out->fragment,
                              error,
                              error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   out->position_output_start =
      vertex_data.vs.varyings[VARYING_SLOT_POS].start;
   out->position_output_count =
      vertex_data.vs.varyings[VARYING_SLOT_POS].count;
   out->fragment_position_start =
      fragment_data.fs.varyings[VARYING_SLOT_POS].start;
   out->fragment_position_count =
      fragment_data.fs.varyings[VARYING_SLOT_POS].count;
   out->varying_output_start =
      vertex_data.vs.varyings[VARYING_SLOT_VAR0].start;
   out->varying_output_count =
      vertex_data.vs.varyings[VARYING_SLOT_VAR0].count;
   out->fragment_varying_start =
      fragment_data.fs.varyings[VARYING_SLOT_VAR0].start;
   out->fragment_varying_count =
      fragment_data.fs.varyings[VARYING_SLOT_VAR0].count;
   out->fragment_texture_descriptor_start =
      fragment_data.common.desc_sets[0].bindings[0].range.start;
   out->fragment_texture_descriptor_count =
      fragment_data.common.desc_sets[0].bindings[0].range.count;
   out->fragment_texture_descriptor_stride =
      fragment_data.common.desc_sets[0].bindings[0].range.stride;

   if (out->vertex.abi.vertex_inputs != 12 ||
       out->vertex.abi.vertex_outputs != 7 ||
       out->vertex.abi.coefficients != 0 ||
       out->vertex.abi.shareds != PVRGPU_TEXTURE_VS_UNIFORM_DWORDS ||
       out->vertex.abi.push_constant_start != 0 ||
       out->vertex.abi.push_constant_count !=
          PVRGPU_TEXTURE_VS_UNIFORM_DWORDS ||
       out->fragment.abi.coefficients != 16 ||
       out->fragment.abi.shareds != PVRGPU_TEXTURE_DESCRIPTOR_DWORDS ||
       out->fragment.abi.push_constant_count != 0 ||
       out->position_output_start != 0 || out->position_output_count != 4 ||
       out->varying_output_start != 4 || out->varying_output_count != 3 ||
       out->fragment_position_start != 0 ||
       out->fragment_position_count != 4 ||
       out->fragment_varying_start != 4 ||
       out->fragment_varying_count != 12 ||
       out->fragment_texture_descriptor_start != 0 ||
       out->fragment_texture_descriptor_count !=
          PVRGPU_TEXTURE_DESCRIPTOR_DWORDS ||
       out->fragment_texture_descriptor_stride !=
          PVRGPU_TEXTURE_DESCRIPTOR_DWORDS) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "compiled texture PCO ABI changed");
   }

   ralloc_free(compile_mem_ctx);
   return true;
}

bool pvrgpu_pco_compile_refract(
   struct pvrgpu_pco_compiler *compiler,
   const nir_shader *vertex_nir,
   const nir_shader *fragment_nir,
   enum pvrgpu_pco_refract_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));

   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir ||
       profile < PVRGPU_PCO_REFRACT_PREPASS ||
       profile > PVRGPU_PCO_REFRACT_COMPOSITE) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "missing compiler or invalid refract profile");
   }
   const struct pvrgpu_refract_pco_desc *desc =
      &pvrgpu_refract_profiles[profile];
   if (!pvrgpu_source_hash_matches(vertex_nir, desc->vertex_source_hash) ||
       !pvrgpu_source_hash_matches(fragment_nir,
                                   desc->fragment_source_hash) ||
       !pvrgpu_validate_refract_nir(vertex_nir,
                                    profile,
                                    MESA_SHADER_VERTEX,
                                    error,
                                    error_size) ||
       !pvrgpu_validate_refract_nir(fragment_nir,
                                    profile,
                                    MESA_SHADER_FRAGMENT,
                                    error,
                                    error_size)) {
      if (error && error_size && error[0] == '\0') {
         snprintf(error,
                  error_size,
                  "%s NIR source signature mismatch",
                  desc->name);
      }
      return false;
   }

   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");
   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "failed to clone %s NIR",
                             desc->name);
   }

   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = pco_nir_options();
   fs->options = pco_nir_options();
   nir_shader_compiler_options vertex_options;
   nir_shader_compiler_options fragment_options;
   /* Gallivm preserves both Refract vertex shaders' matrix and normal chains
    * as distinct multiply and add operations.  Keep that boundary local to
    * these strict profiles instead of changing global PCO policy. */
   vertex_options = *pco_nir_options();
   vertex_options.float_mul_add32 &= ~nir_float_muladd_support_fuse;
   vertex_options.float_mul_add32 |=
      nir_float_muladd_support_prefers_split;
   vs->options = &vertex_options;
   if (profile == PVRGPU_PCO_REFRACT_COMPOSITE) {
      /* The GLES source spells every product and sum separately.  Retain
       * those IEEE operation boundaries through generic NIR optimization. */
      fragment_options = *pco_nir_options();
      fragment_options.float_mul_add32 &= ~nir_float_muladd_support_fuse;
      fragment_options.float_mul_add32 |=
         nir_float_muladd_support_prefers_split;
      fs->options = &fragment_options;
   }
   nir_lower_fragcolor(fs, 1);

   if (!pvrgpu_validate_lit_mesh_fragment_output(fs,
                                                 desc->name,
                                                 error,
                                                 error_size) ||
       (profile == PVRGPU_PCO_REFRACT_COMPOSITE &&
        !pvrgpu_lower_refract_composite_fragment_mediump(fs,
                                                         error,
                                                         error_size)) ||
       !pvrgpu_canonicalize_fragment_output(fs,
                                            desc->name,
                                            error,
                                            error_size) ||
       !pvrgpu_lower_uniform_slots_to_push_constants(
          vs,
          desc->vertex_uniform_dwords,
          desc->vertex_uniform_loads,
          desc->name,
          error,
          error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   if (!pvrgpu_init_refract_shader_data(profile,
                                        &vertex_data,
                                        &fragment_data,
                                        compile_mem_ctx,
                                        error,
                                        error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);
   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   if (!pvrgpu_allocate_push_constants(&vertex_data,
                                       desc->vertex_uniform_dwords,
                                       desc->vertex_uniform_dwords,
                                       desc->name,
                                       "VS",
                                       error,
                                       error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate %s NIR",
                             desc->name);
   }
   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);

   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment,
                              false,
                              &out->fragment,
                              error,
                              error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   out->position_output_start = 0;
   out->position_output_count = 4;
   out->fragment_position_start = 0;
   out->fragment_position_count = 4;
   out->varying_output_start = 4;
   out->varying_output_count = desc->varying_components;
   out->fragment_varying_start = 4;
   out->fragment_varying_count = desc->varying_components * 4;
   if (profile == PVRGPU_PCO_REFRACT_COMPOSITE) {
      out->fragment_texture_descriptor_start = 0;
      out->fragment_texture_descriptor_count =
         PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
      out->fragment_texture_descriptor_stride =
         PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   }

   const unsigned expected_fragment_shareds =
      profile == PVRGPU_PCO_REFRACT_COMPOSITE ?
         PVRGPU_REFRACT_DESCRIPTOR_DWORDS : 0;
   if (out->vertex.abi.vertex_inputs != 8 ||
       out->vertex.abi.vertex_outputs != 4 + desc->varying_components ||
       out->vertex.abi.coefficients != 0 ||
       out->vertex.abi.shareds != desc->vertex_uniform_dwords ||
       out->vertex.abi.push_constant_start != 0 ||
       out->vertex.abi.push_constant_count != desc->vertex_uniform_dwords ||
       out->fragment.abi.coefficients !=
          4 + desc->varying_components * 4 ||
       out->fragment.abi.shareds != expected_fragment_shareds ||
       out->fragment.abi.push_constant_count != 0 ||
       out->position_output_start != 0 || out->position_output_count != 4 ||
       out->varying_output_start != 4 ||
       out->varying_output_count != desc->varying_components ||
       out->fragment_position_start != 0 ||
       out->fragment_position_count != 4 ||
       out->fragment_varying_start != 4 ||
       out->fragment_varying_count != desc->varying_components * 4 ||
       (profile == PVRGPU_PCO_REFRACT_COMPOSITE &&
        (out->fragment_texture_descriptor_start != 0 ||
         out->fragment_texture_descriptor_count !=
            PVRGPU_TEXTURE_DESCRIPTOR_DWORDS ||
         out->fragment_texture_descriptor_stride !=
            PVRGPU_TEXTURE_DESCRIPTOR_DWORDS))) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "compiled %s PCO ABI changed",
                             desc->name);
   }

   ralloc_free(compile_mem_ctx);
   return true;
}

bool pvrgpu_pco_compile_shadow(
   struct pvrgpu_pco_compiler *compiler,
   const nir_shader *vertex_nir,
   const nir_shader *fragment_nir,
   enum pvrgpu_pco_shadow_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));

   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir ||
       profile < PVRGPU_PCO_SHADOW_DEPTH ||
       profile > PVRGPU_PCO_SHADOW_SCENE) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "missing compiler or invalid shadow profile");
   }
   if (profile == PVRGPU_PCO_SHADOW_DEPTH) {
      return pvrgpu_pco_compile_refract(compiler,
                                        vertex_nir,
                                        fragment_nir,
                                        PVRGPU_PCO_REFRACT_PREPASS,
                                        out,
                                        error,
                                        error_size);
   }

   const struct pvrgpu_shadow_pco_desc *desc =
      &pvrgpu_shadow_profiles[profile];
   if (!pvrgpu_source_hash_matches(vertex_nir, desc->vertex_source_hash) ||
       !pvrgpu_source_hash_matches(fragment_nir,
                                   desc->fragment_source_hash) ||
       !pvrgpu_validate_shadow_nir(vertex_nir,
                                   profile,
                                   MESA_SHADER_VERTEX,
                                   error,
                                   error_size) ||
       !pvrgpu_validate_shadow_nir(fragment_nir,
                                   profile,
                                   MESA_SHADER_FRAGMENT,
                                   error,
                                   error_size)) {
      if (error && error_size && error[0] == '\0') {
         snprintf(error,
                  error_size,
                  "%s NIR source signature mismatch",
                  desc->name);
      }
      return false;
   }

   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");
   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "failed to clone %s NIR",
                             desc->name);
   }

   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = pco_nir_options();
   fs->options = pco_nir_options();
   nir_lower_fragcolor(fs, 1);
   if (!pvrgpu_validate_lit_mesh_fragment_output(fs,
                                                 desc->name,
                                                 error,
                                                 error_size) ||
       (profile == PVRGPU_PCO_SHADOW_SCENE &&
        !pvrgpu_split_shadow_scene_dot2(vs,
                                        desc->name,
                                        error,
                                        error_size)) ||
       !pvrgpu_canonicalize_fragment_output(fs,
                                            desc->name,
                                            error,
                                            error_size) ||
       !pvrgpu_lower_uniform_slots_to_push_constants(
          vs,
          desc->vertex_uniform_dwords,
          desc->vertex_uniform_loads,
          desc->name,
          error,
          error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   if (!pvrgpu_init_shadow_shader_data(profile,
                                       &vertex_data,
                                       &fragment_data,
                                       compile_mem_ctx,
                                       error,
                                       error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);
   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   if (!pvrgpu_allocate_push_constants(&vertex_data,
                                       desc->vertex_uniform_dwords,
                                       desc->vertex_uniform_used_dwords,
                                       desc->name,
                                       "VS",
                                       error,
                                       error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate %s NIR",
                             desc->name);
   }
   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);

   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment,
                              false,
                              &out->fragment,
                              error,
                              error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   out->position_output_start = 0;
   out->position_output_count = 4;
   out->fragment_position_start = 0;
   out->fragment_position_count = 4;
   out->varying_output_start = 4;
   out->varying_output_count = desc->varying_components;
   out->fragment_varying_start = 4;
   out->fragment_varying_count = desc->varying_components * 4;
   if (profile == PVRGPU_PCO_SHADOW_MASK) {
      out->fragment_texture_descriptor_start = 0;
      out->fragment_texture_descriptor_count =
         PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
      out->fragment_texture_descriptor_stride =
         PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   }

   const unsigned expected_vertex_inputs =
      profile == PVRGPU_PCO_SHADOW_MASK ? 4U : 8U;
   const unsigned expected_fragment_shareds =
      profile == PVRGPU_PCO_SHADOW_MASK ?
         PVRGPU_TEXTURE_DESCRIPTOR_DWORDS : 0U;
   if (out->vertex.abi.vertex_inputs != expected_vertex_inputs ||
       out->vertex.abi.vertex_outputs != 4 + desc->varying_components ||
       out->vertex.abi.coefficients != 0 ||
       out->vertex.abi.shareds != desc->vertex_uniform_dwords ||
       out->vertex.abi.push_constant_start != 0 ||
       out->vertex.abi.push_constant_count != desc->vertex_uniform_dwords ||
       out->fragment.abi.coefficients !=
          4 + desc->varying_components * 4 ||
       out->fragment.abi.shareds != expected_fragment_shareds ||
       out->fragment.abi.push_constant_count != 0 ||
       out->position_output_start != 0 || out->position_output_count != 4 ||
       out->varying_output_start != 4 ||
       out->varying_output_count != desc->varying_components ||
       out->fragment_position_start != 0 ||
       out->fragment_position_count != 4 ||
       out->fragment_varying_start != 4 ||
       out->fragment_varying_count != desc->varying_components * 4 ||
       (profile == PVRGPU_PCO_SHADOW_MASK &&
        (out->fragment_texture_descriptor_start != 0 ||
         out->fragment_texture_descriptor_count !=
            PVRGPU_TEXTURE_DESCRIPTOR_DWORDS ||
         out->fragment_texture_descriptor_stride !=
            PVRGPU_TEXTURE_DESCRIPTOR_DWORDS))) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "compiled %s PCO ABI changed",
                             desc->name);
   }

   ralloc_free(compile_mem_ctx);
   return true;
}

#define PVRGPU_TERRAIN_D3_FS_VARYINGS 4U
#define PVRGPU_TERRAIN_D3_FS_PUSH_SLOTS 16U
#define PVRGPU_TERRAIN_D3_FS_TEXTURES 5U
#define PVRGPU_TERRAIN_D3_FS_FADDS 13U
#define PVRGPU_TERRAIN_D3_FS_FMULS 24U
#define PVRGPU_TERRAIN_D3_FS_DOTS 7U
#define PVRGPU_TERRAIN_D3_FS_RSQRTS 5U

struct pvrgpu_terrain_d3_fragment_graph {
   nir_intrinsic_instr *varying[PVRGPU_TERRAIN_D3_FS_VARYINGS];
   nir_intrinsic_instr *push[PVRGPU_TERRAIN_D3_FS_PUSH_SLOTS];
   nir_tex_instr *texture[PVRGPU_TERRAIN_D3_FS_TEXTURES];
   nir_alu_instr *fadd[PVRGPU_TERRAIN_D3_FS_FADDS];
   nir_alu_instr *fmul[PVRGPU_TERRAIN_D3_FS_FMULS];
   nir_alu_instr *dot[PVRGPU_TERRAIN_D3_FS_DOTS];
   nir_alu_instr *rsqrt[PVRGPU_TERRAIN_D3_FS_RSQRTS];
   nir_alu_instr *fneg[2];
   nir_alu_instr *fmax[2];
   nir_alu_instr *flrp;
   nir_alu_instr *flt;
   nir_alu_instr *bcsel;
   nir_alu_instr *fdiv;
   nir_alu_instr *fmin;
   nir_alu_instr *fsqrt;
   nir_alu_instr *fpow;
   nir_alu_instr *attenuation_neg;
   nir_alu_instr *attenuation_add;
   nir_alu_instr *light_direction;
   nir_alu_instr *diffuse_max;
   nir_alu_instr *halfway_add;
   nir_alu_instr *halfway_normalize;
   nir_alu_instr *halfway_max;
   nir_alu_instr *material_surface;
   unsigned material_source;
   unsigned surface_source;
   unsigned fadd_components;
   unsigned fmul_components;
};

static bool
pvrgpu_terrain_d3_alu_source_matches_slice(const nir_alu_src *source,
                                            const nir_def *expected,
                                            unsigned first_component,
                                            unsigned components)
{
   if (!source || !source->src.ssa || !expected ||
       first_component + components > expected->num_components)
      return false;
   for (unsigned component = 0; component < components; ++component) {
      const nir_scalar actual = nir_scalar_resolved(
         source->src.ssa, source->swizzle[component]);
      if (actual.def != expected ||
          actual.comp != first_component + component)
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d3_def_matches_slice(const nir_def *value,
                                     const nir_def *expected,
                                     unsigned first_component,
                                     unsigned components)
{
   if (!value || !expected || value->num_components < components ||
       first_component + components > expected->num_components)
      return false;
   for (unsigned component = 0; component < components; ++component) {
      const nir_scalar actual =
         nir_scalar_resolved((nir_def *)value, component);
      if (actual.def != expected ||
          actual.comp != first_component + component)
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d3_alu_has_def(const nir_alu_instr *alu,
                               const nir_def *def)
{
   if (!alu || !def)
      return false;
   const unsigned inputs = nir_op_infos[alu->op].num_inputs;
   for (unsigned source = 0; source < inputs; ++source) {
      if (alu->src[source].src.ssa == def)
         return true;
   }
   return false;
}

static bool
pvrgpu_terrain_d3_material_source_matches(const nir_alu_src *source,
                                           const nir_def *opacity)
{
   if (!source || !source->src.ssa || !opacity)
      return false;

   for (unsigned component = 0; component < 4U; ++component) {
      const nir_scalar scalar = nir_scalar_resolved(
         source->src.ssa, source->swizzle[component]);
      if (component < 3U) {
         if (!nir_scalar_is_const(scalar) ||
             nir_scalar_as_uint(scalar) != 0x3f800000U)
            return false;
      } else if (scalar.def != opacity || scalar.comp != 0U) {
         return false;
      }
   }
   return true;
}

static nir_def *
pvrgpu_terrain_d3_texture_coord(nir_tex_instr *texture)
{
   if (!texture)
      return NULL;
   const int index = nir_tex_instr_src_index(texture, nir_tex_src_coord);
   return index < 0 ? NULL : texture->src[index].src.ssa;
}

static bool
pvrgpu_terrain_d3_collect_fragment_graph(
   nir_shader *nir,
   nir_function_impl *entrypoint,
   struct pvrgpu_terrain_d3_fragment_graph *graph,
   char *error,
   size_t error_size)
{
   memset(graph, 0, sizeof(*graph));
   nir_opt_shrink_vectors(nir, true);
   nir_opt_dce(nir);

   unsigned textures = 0;
   unsigned fadds = 0;
   unsigned fmuls = 0;
   unsigned dots = 0;
   unsigned rsqrts = 0;
   unsigned fnegs = 0;
   unsigned fmaxes = 0;
   unsigned flrps = 0;
   unsigned flts = 0;
   unsigned bcsels = 0;
   unsigned fdivs = 0;
   unsigned fmins = 0;
   unsigned fsqrts = 0;
   unsigned fpows = 0;

   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               nir_variable *var = nir_intrinsic_get_var(intr, 0);
               if (!var || var->data.mode != nir_var_shader_in ||
                   var->data.location < VARYING_SLOT_VAR0 ||
                   var->data.location >=
                      VARYING_SLOT_VAR0 + PVRGPU_TERRAIN_D3_FS_VARYINGS) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D3 FS varying graph changed");
               }
               const unsigned slot =
                  var->data.location - VARYING_SLOT_VAR0;
               if (graph->varying[slot])
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D3 FS varying load is duplicated");
               graph->varying[slot] = intr;
            } else if (intr->intrinsic ==
                       nir_intrinsic_load_push_constant) {
               if (!nir_src_is_const(intr->src[0]) ||
                   nir_intrinsic_base(intr) != 0) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D3 FS push address changed");
               }
               const unsigned byte_offset = nir_src_as_uint(intr->src[0]);
               if ((byte_offset & 15U) != 0 ||
                   byte_offset / 16U >= PVRGPU_TERRAIN_D3_FS_PUSH_SLOTS ||
                   graph->push[byte_offset / 16U]) {
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D3 FS push layout changed");
               }
               graph->push[byte_offset / 16U] = intr;
            }
            continue;
         }

         if (instr->type == nir_instr_type_tex) {
            if (textures >= ARRAY_SIZE(graph->texture))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D3 FS texture count changed");
            graph->texture[textures++] = nir_instr_as_tex(instr);
            continue;
         }

         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         switch (alu->op) {
         case nir_op_fadd:
            if (fadds >= ARRAY_SIZE(graph->fadd))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D3 FS fadd count changed");
            graph->fadd[fadds++] = alu;
            graph->fadd_components += alu->def.num_components;
            break;
         case nir_op_fmul:
            if (fmuls >= ARRAY_SIZE(graph->fmul))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D3 FS fmul count changed");
            graph->fmul[fmuls++] = alu;
            graph->fmul_components += alu->def.num_components;
            break;
         case nir_op_fdot3:
            if (dots >= ARRAY_SIZE(graph->dot))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D3 FS dot count changed");
            graph->dot[dots++] = alu;
            break;
         case nir_op_frsq:
            if (rsqrts >= ARRAY_SIZE(graph->rsqrt))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D3 FS rsqrt count changed");
            graph->rsqrt[rsqrts++] = alu;
            break;
         case nir_op_fneg:
            if (fnegs >= ARRAY_SIZE(graph->fneg))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D3 FS fneg count changed");
            graph->fneg[fnegs++] = alu;
            break;
         case nir_op_fmax:
            if (fmaxes >= ARRAY_SIZE(graph->fmax))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D3 FS fmax count changed");
            graph->fmax[fmaxes++] = alu;
            break;
         case nir_op_flrp:
            graph->flrp = alu;
            ++flrps;
            break;
         case nir_op_flt:
            graph->flt = alu;
            ++flts;
            break;
         case nir_op_bcsel:
            graph->bcsel = alu;
            ++bcsels;
            break;
         case nir_op_fdiv:
            graph->fdiv = alu;
            ++fdivs;
            break;
         case nir_op_fmin:
            graph->fmin = alu;
            ++fmins;
            break;
         case nir_op_fsqrt:
            graph->fsqrt = alu;
            ++fsqrts;
            break;
         case nir_op_fpow:
            graph->fpow = alu;
            ++fpows;
            break;
         case nir_op_ffma:
         case nir_op_ffma_weak:
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "terrain D3 FS retained fused arithmetic");
         default:
            break;
         }
      }
   }

   if (textures != ARRAY_SIZE(graph->texture) ||
       fadds != ARRAY_SIZE(graph->fadd) ||
       fmuls != ARRAY_SIZE(graph->fmul) ||
       dots != ARRAY_SIZE(graph->dot) ||
       rsqrts != ARRAY_SIZE(graph->rsqrt) || fnegs != 2U ||
       fmaxes != 2U || flrps != 1U || flts != 1U || bcsels != 1U ||
       fdivs != 1U || fmins != 1U || fsqrts != 1U || fpows != 1U ||
       graph->fadd_components != 37U ||
       graph->fmul_components != 69U) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D3 FS source graph changed "
         "(tex=%u add=%u/%u mul=%u/%u dot=%u rsq=%u)",
         textures,
         fadds,
         graph->fadd_components,
         fmuls,
         graph->fmul_components,
         dots,
         rsqrts);
   }

   static const unsigned varying_components[4] = { 4, 4, 4, 2 };
   for (unsigned slot = 0; slot < ARRAY_SIZE(graph->varying); ++slot) {
      if (!graph->varying[slot] ||
          graph->varying[slot]->def.bit_size != 32 ||
          graph->varying[slot]->def.num_components !=
             varying_components[slot]) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D3 FS varying ABI changed");
      }
   }

   static const unsigned push_components[16] = {
      3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1,
   };
   for (unsigned slot = 0; slot < ARRAY_SIZE(graph->push); ++slot) {
      if (!graph->push[slot] || graph->push[slot]->def.bit_size != 32 ||
          graph->push[slot]->def.num_components != push_components[slot]) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D3 FS push ABI changed at slot %u",
                                slot);
      }
   }

   for (unsigned multiply = 0; multiply < ARRAY_SIZE(graph->fmul);
        ++multiply) {
      nir_alu_instr *alu = graph->fmul[multiply];
      if (alu->def.num_components != 4U)
         continue;
      for (unsigned surface_source = 0; surface_source < 2U;
           ++surface_source) {
         const unsigned material_source = 1U - surface_source;
         if (!pvrgpu_terrain_d3_alu_source_matches_slice(
                &alu->src[surface_source], &graph->flrp->def, 0, 4) ||
             !pvrgpu_terrain_d3_material_source_matches(
                &alu->src[material_source], &graph->push[4]->def)) {
            continue;
         }
         if (graph->material_surface)
            return pvrgpu_pco_fail(
               error,
               error_size,
               "terrain D3 FS material multiply is ambiguous");
         graph->material_surface = alu;
         graph->material_source = material_source;
         graph->surface_source = surface_source;
      }
   }
   if (!graph->material_surface)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS material multiply changed");

   nir_def *detail_coord =
      pvrgpu_terrain_d3_texture_coord(graph->texture[0]);
   nir_def *base_coord =
      pvrgpu_terrain_d3_texture_coord(graph->texture[3]);
   if (!detail_coord || !base_coord || detail_coord->bit_size != 32 ||
       detail_coord->num_components != 2 || base_coord->bit_size != 32 ||
       base_coord->num_components != 2 ||
       pvrgpu_terrain_d3_texture_coord(graph->texture[1]) != detail_coord ||
       pvrgpu_terrain_d3_texture_coord(graph->texture[2]) != detail_coord ||
       pvrgpu_terrain_d3_texture_coord(graph->texture[4]) != detail_coord ||
       !pvrgpu_terrain_d3_def_matches_slice(
          base_coord, &graph->varying[0]->def, 0, 2)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS texture coordinate graph changed");
   }

   for (unsigned texture = 0; texture < ARRAY_SIZE(graph->texture);
        ++texture) {
      const uint32_t expected_index = texture << 16U;
      if (graph->texture[texture]->texture_index != expected_index ||
          graph->texture[texture]->sampler_index != expected_index ||
          graph->texture[texture]->def.bit_size != 32 ||
          graph->texture[texture]->def.num_components != 4) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D3 FS texture ABI changed at %u",
                                texture);
      }
   }

   if (nir_def_instr(detail_coord)->type != nir_instr_type_alu)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS detail coordinate is not ALU");
   nir_alu_instr *detail_add =
      nir_instr_as_alu(nir_def_instr(detail_coord));
   if (detail_add->op != nir_op_fadd ||
       detail_add->def.num_components != 2)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS detail add changed");

   bool detail_graph_matches = false;
   for (unsigned offset_source = 0; offset_source < 2; ++offset_source) {
      const unsigned multiply_source = 1U - offset_source;
      if (!pvrgpu_terrain_d3_alu_source_matches_slice(
             &detail_add->src[offset_source],
             &graph->push[7]->def,
             0,
             2) ||
          !detail_add->src[multiply_source].src.ssa ||
          nir_def_instr(detail_add->src[multiply_source].src.ssa)->type !=
             nir_instr_type_alu)
         continue;
      nir_alu_instr *detail_multiply = nir_instr_as_alu(
         nir_def_instr(detail_add->src[multiply_source].src.ssa));
      if (detail_multiply->op != nir_op_fmul ||
          detail_multiply->def.num_components != 2)
         continue;
      for (unsigned repeat_source = 0; repeat_source < 2;
           ++repeat_source) {
         const unsigned varying_source = 1U - repeat_source;
         if (pvrgpu_terrain_d3_alu_source_matches_slice(
                &detail_multiply->src[repeat_source],
                &graph->push[6]->def,
                0,
                2) &&
             pvrgpu_terrain_d3_alu_source_matches_slice(
                &detail_multiply->src[varying_source],
                &graph->varying[0]->def,
                0,
                2)) {
            detail_graph_matches = true;
         }
      }
   }
   if (!detail_graph_matches)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS detail coordinate inputs changed");

   for (unsigned dot = 0; dot < ARRAY_SIZE(graph->dot); ++dot) {
      if (graph->dot[dot]->def.bit_size != 32 ||
          graph->dot[dot]->def.num_components != 1)
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D3 FS dot ABI changed");
   }
   static const unsigned rsqrt_dot[5] = { 0, 1, 2, 3, 5 };
   for (unsigned rsqrt = 0; rsqrt < ARRAY_SIZE(graph->rsqrt); ++rsqrt) {
      if (graph->rsqrt[rsqrt]->def.bit_size != 32 ||
          graph->rsqrt[rsqrt]->def.num_components != 1 ||
          !pvrgpu_terrain_d3_alu_has_def(
             graph->rsqrt[rsqrt], &graph->dot[rsqrt_dot[rsqrt]]->def)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D3 FS rsqrt/dot association changed");
      }
   }
   if (!pvrgpu_terrain_d3_alu_has_def(graph->fsqrt,
                                      &graph->dot[3]->def) ||
       !pvrgpu_terrain_d3_alu_has_def(graph->fdiv,
                                      &graph->fsqrt->def) ||
       !pvrgpu_terrain_d3_alu_has_def(graph->fdiv,
                                      &graph->push[15]->def) ||
       !pvrgpu_terrain_d3_alu_has_def(graph->flt,
                                      &graph->push[15]->def) ||
       !pvrgpu_terrain_d3_alu_has_def(graph->fmin,
                                      &graph->fdiv->def)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS distance highp island changed");
   }

   for (unsigned neg = 0; neg < ARRAY_SIZE(graph->fneg); ++neg) {
      if (pvrgpu_terrain_d3_alu_has_def(graph->fneg[neg],
                                        &graph->fmin->def)) {
         graph->attenuation_neg = graph->fneg[neg];
      }
   }
   if (!graph->attenuation_neg)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS attenuation neg changed");
   for (unsigned add = 0; add < ARRAY_SIZE(graph->fadd); ++add) {
      if (pvrgpu_terrain_d3_alu_has_def(
             graph->fadd[add], &graph->attenuation_neg->def)) {
         if (graph->attenuation_add)
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "terrain D3 FS attenuation add ambiguous");
         graph->attenuation_add = graph->fadd[add];
      }
   }
   if (!graph->attenuation_add ||
       !pvrgpu_terrain_d3_alu_has_def(graph->bcsel, &graph->flt->def) ||
       !pvrgpu_terrain_d3_alu_has_def(
          graph->bcsel, &graph->attenuation_add->def)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS attenuation select changed");
   }

   nir_def *light_vector = graph->dot[3]->src[0].src.ssa;
   nir_def *halfway = graph->dot[5]->src[0].src.ssa;
   if (!light_vector || graph->dot[3]->src[1].src.ssa != light_vector ||
       !halfway || graph->dot[5]->src[1].src.ssa != halfway ||
       nir_def_instr(halfway)->type != nir_instr_type_alu) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS highp vector graph changed");
   }
   graph->halfway_add = nir_instr_as_alu(nir_def_instr(halfway));
   if (graph->halfway_add->op != nir_op_fadd ||
       graph->halfway_add->def.num_components != 3)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS halfway add changed");

   for (unsigned multiply = 0; multiply < ARRAY_SIZE(graph->fmul);
        ++multiply) {
      nir_alu_instr *alu = graph->fmul[multiply];
      if (pvrgpu_terrain_d3_alu_has_def(alu, light_vector) &&
          pvrgpu_terrain_d3_alu_has_def(alu, &graph->rsqrt[3]->def)) {
         graph->light_direction = alu;
      }
      if (pvrgpu_terrain_d3_alu_has_def(alu, halfway) &&
          pvrgpu_terrain_d3_alu_has_def(alu, &graph->rsqrt[4]->def)) {
         graph->halfway_normalize = alu;
      }
   }
   if (!graph->light_direction || !graph->halfway_normalize ||
       graph->light_direction->def.num_components != 3 ||
       graph->halfway_normalize->def.num_components != 3 ||
       !pvrgpu_terrain_d3_alu_has_def(graph->dot[4],
                                      &graph->light_direction->def) ||
       !pvrgpu_terrain_d3_alu_has_def(graph->dot[6],
                                      &graph->halfway_normalize->def)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS highp normalize graph changed");
   }

   for (unsigned maximum = 0; maximum < ARRAY_SIZE(graph->fmax);
        ++maximum) {
      if (pvrgpu_terrain_d3_alu_has_def(graph->fmax[maximum],
                                        &graph->dot[4]->def))
         graph->diffuse_max = graph->fmax[maximum];
      if (pvrgpu_terrain_d3_alu_has_def(graph->fmax[maximum],
                                        &graph->dot[6]->def))
         graph->halfway_max = graph->fmax[maximum];
   }
   if (!graph->diffuse_max || !graph->halfway_max ||
       !pvrgpu_terrain_d3_alu_has_def(graph->fpow,
                                      &graph->halfway_max->def) ||
       !pvrgpu_terrain_d3_alu_has_def(graph->fpow,
                                      &graph->push[3]->def)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS lighting weight graph changed");
   }

   const unsigned highp_add_components =
      graph->attenuation_add->def.num_components +
      graph->halfway_add->def.num_components;
   const unsigned highp_multiply_components =
      graph->light_direction->def.num_components +
      graph->halfway_normalize->def.num_components;
   if (highp_add_components != 4U || highp_multiply_components != 6U)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS highp scalar count changed");
   return true;
}

static nir_def *
pvrgpu_terrain_d3_round_alu_result_rtne(nir_alu_instr *alu)
{
   nir_builder builder = nir_builder_at(nir_after_instr(&alu->instr));
   nir_def *rounded =
      pvrgpu_terrain_d1_round_half_rtne(&builder, &alu->def);
   nir_def_rewrite_uses_after(&alu->def, rounded);
   return rounded;
}

static bool
pvrgpu_terrain_d3_replace_dot(nir_alu_instr *dot,
                               bool mediump,
                               char *error,
                               size_t error_size)
{
   if (!dot || dot->op != nir_op_fdot3 || dot->def.bit_size != 32 ||
       dot->def.num_components != 1)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS dot replacement changed");

   nir_builder builder = nir_builder_at(nir_before_instr(&dot->instr));
   builder.fp_math_ctrl = dot->fp_math_ctrl;
   nir_def *left = nir_ssa_for_alu_src(&builder, dot, 0);
   nir_def *right = nir_ssa_for_alu_src(&builder, dot, 1);
   if (!left || !right || left->num_components != 3 ||
       right->num_components != 3)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS dot sources changed");

   /* Gallivm scalarizes dot3 in z, y, x order. */
   nir_def *sum = NULL;
   for (int component = 2; component >= 0; --component) {
      nir_def *product = nir_fmul(
         &builder,
         nir_channel(&builder, left, (unsigned)component),
         nir_channel(&builder, right, (unsigned)component));
      if (mediump)
         product = pvrgpu_terrain_d1_round_half_rtne(&builder, product);
      if (!sum) {
         sum = product;
         continue;
      }
      sum = nir_fadd(&builder, sum, product);
      if (mediump)
         sum = pvrgpu_terrain_d1_round_half_rtne(&builder, sum);
   }
   nir_def_replace(&dot->def, sum);
   return true;
}

static bool
pvrgpu_terrain_d3_replace_flrp(
   struct pvrgpu_terrain_d3_fragment_graph *graph,
   char *error,
   size_t error_size)
{
   nir_alu_instr *flrp = graph->flrp;
   if (!flrp || flrp->op != nir_op_flrp || flrp->def.bit_size != 32 ||
       flrp->def.num_components != 4 ||
       flrp->src[0].src.ssa != &graph->texture[1]->def ||
       flrp->src[1].src.ssa != &graph->texture[2]->def ||
       !flrp->src[2].src.ssa ||
       nir_def_instr(flrp->src[2].src.ssa)->type != nir_instr_type_alu) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS flrp sources changed");
   }

   nir_alu_instr *mix_factor =
      nir_instr_as_alu(nir_def_instr(flrp->src[2].src.ssa));
   nir_alu_instr *mix_neg = NULL;
   for (unsigned neg = 0; neg < ARRAY_SIZE(graph->fneg); ++neg) {
      if (pvrgpu_terrain_d3_alu_has_def(graph->fneg[neg],
                                        &graph->texture[3]->def)) {
         mix_neg = graph->fneg[neg];
      }
   }
   if (!mix_neg || mix_factor->op != nir_op_fadd ||
       mix_factor->def.num_components != 4 ||
       !pvrgpu_terrain_d3_alu_has_def(mix_factor, &mix_neg->def)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS mix factor changed");
   }

   nir_builder builder = nir_builder_at(nir_before_instr(&flrp->instr));
   builder.fp_math_ctrl = flrp->fp_math_ctrl;
   nir_def *first = nir_ssa_for_alu_src(&builder, flrp, 0);
   nir_def *second = nir_ssa_for_alu_src(&builder, flrp, 1);
   nir_def *factor = nir_ssa_for_alu_src(&builder, flrp, 2);
   nir_def *delta = nir_fadd(&builder, second, nir_fneg(&builder, first));
   delta = pvrgpu_terrain_d1_round_half_rtne(&builder, delta);
   nir_def *weighted = nir_fmul(&builder, factor, delta);
   weighted = pvrgpu_terrain_d1_round_half_rtne(&builder, weighted);
   nir_def *surface = nir_fadd(&builder, first, weighted);
   surface = pvrgpu_terrain_d1_round_half_rtne(&builder, surface);
   nir_def_replace(&flrp->def, surface);
   return true;
}

static bool
pvrgpu_terrain_d3_replace_material_surface(
   struct pvrgpu_terrain_d3_fragment_graph *graph,
   char *error,
   size_t error_size)
{
   nir_alu_instr *multiply = graph->material_surface;
   if (!multiply || multiply->op != nir_op_fmul ||
       multiply->def.bit_size != 32 ||
       multiply->def.num_components != 4 ||
       graph->material_source >= 2U || graph->surface_source >= 2U ||
       graph->material_source == graph->surface_source) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS material replacement changed");
   }

   nir_builder builder =
      nir_builder_at(nir_before_instr(&multiply->instr));
   builder.fp_math_ctrl = multiply->fp_math_ctrl;
   nir_def *material = nir_ssa_for_alu_src(
      &builder, multiply, graph->material_source);
   nir_def *surface = nir_ssa_for_alu_src(
      &builder, multiply, graph->surface_source);
   if (!material || !surface || material->num_components != 4U ||
       surface->num_components != 4U) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS material sources changed");
   }

   /* GLSL's material RGB is exactly one.  Gallivm folds those three
    * multiplications; only the opacity product crosses a mediump boundary. */
   nir_def *alpha = nir_fmul(&builder,
                             nir_channel(&builder, material, 3),
                             nir_channel(&builder, surface, 3));
   alpha = pvrgpu_terrain_d1_round_half_rtne(&builder, alpha);
   nir_def *result = nir_vec4(&builder,
                              nir_channel(&builder, surface, 0),
                              nir_channel(&builder, surface, 1),
                              nir_channel(&builder, surface, 2),
                              alpha);
   nir_def_replace(&multiply->def, result);
   return true;
}

static bool
pvrgpu_terrain_d3_replace_power(
   struct pvrgpu_terrain_d3_fragment_graph *graph,
   char *error,
   size_t error_size)
{
   nir_alu_instr *power = graph->fpow;
   if (!power || power->op != nir_op_fpow ||
       power->def.bit_size != 32 || power->def.num_components != 1 ||
       !pvrgpu_terrain_d3_alu_has_def(power, &graph->halfway_max->def) ||
       !pvrgpu_terrain_d3_alu_has_def(power, &graph->push[3]->def)) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS power sources changed");
   }

   nir_builder builder = nir_builder_at(nir_before_instr(&power->instr));
   builder.fp_math_ctrl = power->fp_math_ctrl;
   nir_def *base = nir_ssa_for_alu_src(&builder, power, 0);
   nir_def *exponent = nir_ssa_for_alu_src(&builder, power, 1);
   nir_def *logarithm = nir_flog2(&builder, base);
   logarithm =
      pvrgpu_terrain_d1_round_half_rtne(&builder, logarithm);
   nir_def *scaled = nir_fmul(&builder, logarithm, exponent);
   scaled = pvrgpu_terrain_d1_round_half_rtne(&builder, scaled);
   nir_def *result = nir_fexp2(&builder, scaled);
   result = pvrgpu_terrain_d1_round_half_rtne(&builder, result);
   nir_def_replace(&power->def, result);
   return true;
}

/* Terrain D3 is default-mediump with two source-level highp lighting islands.
 * Restore the captured GLES precision graph without changing raw texture
 * coordinates or promoting the whole shader to half.  The pass is deliberately
 * profile-local and validates every structural anchor before rewriting:
 *  - 14 interpolants and 38 push values enter mediump with RTNE;
 *  - all five RGBA UNORM samples enter with RTZ;
 *  - flrp is rebuilt in source order a + (1-d) * (b-a);
 *  - mediump dot3 uses z/y/x scalar order and per-operation RTNE;
 *  - three mediump rsqrt operations expose rounded sqrt and reciprocal; and
 *  - fpow exposes rounded log2, multiply, and exp2 boundaries.
 * The distance and halfway-vector islands remain highp until their explicit
 * five-component return boundary. */
static bool
pvrgpu_lower_terrain_d3_fragment_mediump(nir_shader *nir,
                                          char *error,
                                          size_t error_size)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 mediump requires a fragment shader");
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
   if (!entrypoint)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 fragment entrypoint is missing");

   struct pvrgpu_terrain_d3_fragment_graph graph;
   if (!pvrgpu_terrain_d3_collect_fragment_graph(
          nir, entrypoint, &graph, error, error_size) ||
       !pvrgpu_terrain_d3_replace_flrp(&graph, error, error_size) ||
       !pvrgpu_terrain_d3_replace_material_surface(
          &graph, error, error_size) ||
       !pvrgpu_terrain_d3_replace_power(&graph, error, error_size)) {
      return false;
   }

   unsigned half_dot_adds = 0;
   unsigned half_dot_multiplies = 0;
   for (unsigned dot = 0; dot < ARRAY_SIZE(graph.dot); ++dot) {
      const bool mediump = dot < 3U || dot == 6U;
      if (!pvrgpu_terrain_d3_replace_dot(
             graph.dot[dot], mediump, error, error_size)) {
         return false;
      }
      if (mediump) {
         half_dot_adds += 2U;
         half_dot_multiplies += 3U;
      }
   }

   unsigned rounded_varying_components = 0;
   for (unsigned slot = 0; slot < ARRAY_SIZE(graph.varying); ++slot) {
      nir_intrinsic_instr *load = graph.varying[slot];
      nir_builder builder = nir_builder_at(nir_after_instr(&load->instr));
      nir_def *rounded =
         pvrgpu_terrain_d1_round_half_rtne(&builder, &load->def);
      nir_def_rewrite_uses_after(&load->def, rounded);
      rounded_varying_components += load->def.num_components;
   }

   /* The displacement sample alone consumes the original highp vUv. */
   const int displacement_coord_index =
      nir_tex_instr_src_index(graph.texture[3], nir_tex_src_coord);
   if (displacement_coord_index < 0)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS displacement coord disappeared");
   nir_builder base_builder =
      nir_builder_at(nir_before_instr(&graph.texture[3]->instr));
   nir_def *raw_base_coord = nir_vec2(
      &base_builder,
      nir_channel(&base_builder, &graph.varying[0]->def, 0),
      nir_channel(&base_builder, &graph.varying[0]->def, 1));
   nir_src_rewrite(&graph.texture[3]->src[displacement_coord_index].src,
                   raw_base_coord);

   unsigned rounded_push_components = 0;
   for (unsigned slot = 0; slot < 15U; ++slot) {
      nir_intrinsic_instr *load = graph.push[slot];
      nir_builder builder = nir_builder_at(nir_after_instr(&load->instr));
      nir_def *rounded =
         pvrgpu_terrain_d1_round_half_rtne(&builder, &load->def);
      nir_def_rewrite_uses_after(&load->def, rounded);
      rounded_push_components += load->def.num_components;
   }

   nir_intrinsic_instr *distance = graph.push[15];
   unsigned distance_uses = 0;
   unsigned distance_compare_uses = 0;
   unsigned distance_divide_uses = 0;
   nir_foreach_use (use, &distance->def) {
      nir_instr *use_instr = nir_src_use_instr(use);
      if (use_instr == &graph.flt->instr)
         ++distance_compare_uses;
      else if (use_instr == &graph.fdiv->instr)
         ++distance_divide_uses;
      ++distance_uses;
   }
   if (distance_uses != 2U || distance_compare_uses != 1U ||
       distance_divide_uses != 1U) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS distance dual-use changed");
   }
   nir_builder distance_builder =
      nir_builder_at(nir_after_instr(&distance->instr));
   nir_def *rounded_distance = pvrgpu_terrain_d1_round_half_rtne(
      &distance_builder, &distance->def);
   bool rewrote_distance_compare = false;
   for (unsigned source = 0; source < 2; ++source) {
      if (graph.flt->src[source].src.ssa != &distance->def)
         continue;
      nir_src_rewrite(&graph.flt->src[source].src, rounded_distance);
      rewrote_distance_compare = true;
   }
   if (!rewrote_distance_compare)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D3 FS distance comparison changed");
   ++rounded_push_components;

   for (unsigned texture = 0; texture < ARRAY_SIZE(graph.texture);
        ++texture)
      pvrgpu_terrain_d3_round_texture_rtz(graph.texture[texture]);

   unsigned rounded_original_adds = 0;
   for (unsigned add = 0; add < ARRAY_SIZE(graph.fadd); ++add) {
      nir_alu_instr *alu = graph.fadd[add];
      if (alu == graph.attenuation_add || alu == graph.halfway_add)
         continue;
      rounded_original_adds += alu->def.num_components;
      pvrgpu_terrain_d3_round_alu_result_rtne(alu);
   }

   /* The material vec4 multiply was reduced above to its one non-identity
    * alpha lane and rounded at that source-level operation boundary. */
   unsigned rounded_original_multiplies = 1U;
   for (unsigned multiply = 0; multiply < ARRAY_SIZE(graph.fmul);
        ++multiply) {
      nir_alu_instr *alu = graph.fmul[multiply];
      if (alu == graph.material_surface ||
          alu == graph.light_direction ||
          alu == graph.halfway_normalize)
         continue;
      rounded_original_multiplies += alu->def.num_components;
      pvrgpu_terrain_d3_round_alu_result_rtne(alu);
   }

   /* Explicit highp-to-mediump return boundaries. */
   pvrgpu_terrain_d3_round_alu_result_rtne(graph.attenuation_add);
   pvrgpu_terrain_d3_round_alu_result_rtne(graph.diffuse_max);
   pvrgpu_terrain_d3_round_alu_result_rtne(graph.halfway_normalize);
   const unsigned highp_return_components = 5U;

   unsigned rounded_rsqrt_steps = 0;
   for (unsigned index = 0; index < 3U; ++index) {
      nir_alu_instr *rsqrt = graph.rsqrt[index];
      nir_builder builder = nir_builder_at(nir_after_instr(&rsqrt->instr));
      nir_def *sqrt_value = nir_frcp(&builder, &rsqrt->def);
      nir_def *sqrt_rounded =
         pvrgpu_terrain_d1_round_half_rtne(&builder, sqrt_value);
      nir_def *scale = nir_frcp(&builder, sqrt_rounded);
      nir_def *scale_rounded =
         pvrgpu_terrain_d1_round_half_rtne(&builder, scale);
      nir_def_rewrite_uses_after(&rsqrt->def, scale_rounded);
      rounded_rsqrt_steps += 2U;
   }

   const unsigned flrp_adds = 8U;
   const unsigned flrp_multiplies = 4U;
   const unsigned half_adds =
      rounded_original_adds + half_dot_adds + flrp_adds;
   const unsigned half_multiplies =
      rounded_original_multiplies + half_dot_multiplies +
      flrp_multiplies;
   const unsigned input_and_return_rounds =
      rounded_varying_components + rounded_push_components +
      highp_return_components;
   const unsigned power_rounds = 3U;
   const unsigned rtne_rounds = input_and_return_rounds + half_adds +
                                half_multiplies + rounded_rsqrt_steps +
                                power_rounds;
   const unsigned rtz_rounds = PVRGPU_TERRAIN_D3_FS_TEXTURES * 4U;
   if (rounded_varying_components != 14U ||
       rounded_push_components != 38U ||
       rounded_original_adds != 33U ||
       rounded_original_multiplies != 60U || half_adds != 49U ||
       half_multiplies != 76U || input_and_return_rounds != 57U ||
       rounded_rsqrt_steps != 6U || rtne_rounds != 191U ||
       rtz_rounds != 20U) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D3 FS mediump signature changed "
         "(in=%u add=%u mul=%u rsq=%u pow=%u rtne=%u rtz=%u)",
         input_and_return_rounds,
         half_adds,
         half_multiplies,
         rounded_rsqrt_steps,
         power_rounds,
         rtne_rounds,
         rtz_rounds);
   }

   nir_opt_dce(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_opt_dce(nir);
   nir_progress(true, entrypoint, nir_metadata_control_flow);
   return true;
}

/*
 * Terrain D4/D5/D6 GLES-mediump precision recovery.  This pass runs after
 * fragment-output canonicalization, uniform-to-push lowering, and
 * texture-binding packing, but before pco_preprocess_nir().
 *
 * Capture-derived native scalar operation counts (one logical RGBA output;
 * exclude the reference replay's seven duplicate MRT stores):
 *
 *   D4/D5: tex=5 fabs=1 fmul=18 fdiv=1 fneg=1 fadd=16
 *            native input f2f16 RTNE=2/1, texture f16 RTZ=15,
 *            native f2f32=7 (four off-center coordinates plus RGB output).
 *   D6:    tex=1 fmul=4, native opacity f2f16 RTNE=1,
 *            texture f16 RTZ=4, native f2f32=4.
 *   Native NIR total instructions are D4=68, D5=67, D6=17 per logical
 *            output, or 82/81/31 in the reference compiler dump with eight
 *            MRT stores.
 *
 * PCO currently executes fp32 ALU, so this lowering emulates each live half
 * boundary with f2f16/f2f32.  Exact post-lowering conversion counts are:
 *
 *   D4: f2f16_rtne=43, f2f16_rtz=15, f2f32=58
 *   D5: f2f16_rtne=42, f2f16_rtz=15, f2f32=57
 *   D6: f2f16_rtne=5,  f2f16_rtz=4,  f2f32=9
 */

struct pvrgpu_terrain_d4_d5_graph {
   nir_function_impl *entrypoint;
   nir_intrinsic_instr *varying;
   nir_tex_instr *texture[5];
   nir_intrinsic_instr *store[4];
   nir_load_const_instr *zero;
   nir_load_const_instr *one;
   nir_load_const_instr *two;
   nir_load_const_instr *step;
   nir_load_const_instr *weight[3];
   nir_alu_instr *absolute;
   nir_alu_instr *scaled_multiply;
   nir_alu_instr *scaled_divide;
   nir_alu_instr *twice;
   nir_alu_instr *negative_twice;
   nir_alu_instr *negative_scaled;
   nir_alu_instr *term[5];
   nir_alu_instr *sum[4];
};

struct pvrgpu_terrain_d4_d6_lowered_counts {
   unsigned constants;
   unsigned textures;
   unsigned frag_stores;
   unsigned fabs;
   unsigned fadd;
   unsigned fdiv;
   unsigned fmul;
   unsigned fneg;
   unsigned vec2;
   unsigned vec4;
   unsigned f2f16_rtne;
   unsigned f2f16_rtz;
   unsigned f2f32;
};

static nir_scalar
pvrgpu_terrain_d4_d6_resolve_source(const nir_alu_src *source,
                                     unsigned component)
{
   if (!source || !source->src.ssa || component >= NIR_MAX_VEC_COMPONENTS)
      return (nir_scalar){ 0 };
   return nir_scalar_resolved(source->src.ssa,
                              source->swizzle[component]);
}

static bool
pvrgpu_terrain_d4_d6_scalar_is_def(nir_scalar scalar,
                                    const nir_def *def,
                                    unsigned component)
{
   return scalar.def && def && scalar.def == def &&
          scalar.comp == component;
}

static bool
pvrgpu_terrain_d4_d6_source_is_def_slice(const nir_alu_src *source,
                                          const nir_def *def,
                                          unsigned first_component,
                                          unsigned components)
{
   if (!source || !source->src.ssa || !def ||
       first_component + components > def->num_components)
      return false;
   for (unsigned component = 0; component < components; ++component) {
      if (!pvrgpu_terrain_d4_d6_scalar_is_def(
             pvrgpu_terrain_d4_d6_resolve_source(source, component),
             def,
             first_component + component))
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d4_d6_source_is_splat_def(const nir_alu_src *source,
                                          const nir_def *def,
                                          unsigned component,
                                          unsigned components)
{
   if (!source || !source->src.ssa || !def ||
       component >= def->num_components)
      return false;
   for (unsigned lane = 0; lane < components; ++lane) {
      if (!pvrgpu_terrain_d4_d6_scalar_is_def(
             pvrgpu_terrain_d4_d6_resolve_source(source, lane),
             def,
             component))
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d4_d6_binary_matches(const nir_alu_instr *alu,
                                     nir_op op,
                                     const nir_def *left,
                                     const nir_def *right,
                                     unsigned components)
{
   if (!alu || alu->op != op || alu->def.bit_size != 32 ||
       alu->def.num_components != components)
      return false;
   return (pvrgpu_terrain_d4_d6_source_is_def_slice(&alu->src[0],
                                                     left,
                                                     0,
                                                     components) &&
           pvrgpu_terrain_d4_d6_source_is_def_slice(&alu->src[1],
                                                     right,
                                                     0,
                                                     components)) ||
          (pvrgpu_terrain_d4_d6_source_is_def_slice(&alu->src[1],
                                                     left,
                                                     0,
                                                     components) &&
           pvrgpu_terrain_d4_d6_source_is_def_slice(&alu->src[0],
                                                     right,
                                                     0,
                                                     components));
}

static nir_alu_instr *
pvrgpu_terrain_d4_d6_find_binary(nir_function_impl *entrypoint,
                                  nir_op op,
                                  const nir_def *left,
                                  const nir_def *right,
                                  unsigned components)
{
   nir_alu_instr *found = NULL;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (!pvrgpu_terrain_d4_d6_binary_matches(alu,
                                                   op,
                                                   left,
                                                   right,
                                                   components))
            continue;
         if (found)
            return NULL;
         found = alu;
      }
   }
   return found;
}

static nir_alu_instr *
pvrgpu_terrain_d4_d6_find_vector_times_splat(
   nir_function_impl *entrypoint,
   const nir_def *vector,
   const nir_def *scalar,
   unsigned components)
{
   nir_alu_instr *found = NULL;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_fmul || alu->def.bit_size != 32 ||
             alu->def.num_components != components)
            continue;
         bool matches = false;
         for (unsigned vector_source = 0; vector_source < 2;
              ++vector_source) {
            const unsigned scalar_source = 1U - vector_source;
            if (pvrgpu_terrain_d4_d6_source_is_def_slice(
                   &alu->src[vector_source], vector, 0, components) &&
                pvrgpu_terrain_d4_d6_source_is_splat_def(
                   &alu->src[scalar_source], scalar, 0, components)) {
               matches = true;
               break;
            }
         }
         if (!matches)
            continue;
         if (found)
            return NULL;
         found = alu;
      }
   }
   return found;
}

static nir_alu_instr *
pvrgpu_terrain_d4_d6_find_fneg(nir_function_impl *entrypoint,
                                const nir_def *value)
{
   nir_alu_instr *found = NULL;
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_fneg || alu->def.bit_size != 32 ||
             alu->def.num_components != 1 ||
             !pvrgpu_terrain_d4_d6_source_is_def_slice(&alu->src[0],
                                                        value,
                                                        0,
                                                        1))
            continue;
         if (found)
            return NULL;
         found = alu;
      }
   }
   return found;
}

static bool
pvrgpu_terrain_d4_d5_texture_is_canonical(const nir_tex_instr *texture)
{
   if (!texture || texture->op != nir_texop_tex ||
       texture->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       texture->dest_type != nir_type_float32 ||
       texture->def.num_components != 4 || texture->def.bit_size != 32 ||
       texture->num_srcs != 1 || texture->coord_components != 2 ||
       texture->src[0].src_type != nir_tex_src_coord ||
       !texture->src[0].src.ssa || texture->is_array ||
       texture->is_shadow || texture->is_sparse ||
       texture->texture_non_uniform || texture->sampler_non_uniform ||
       texture->embedded_sampler || texture->offset_non_uniform ||
       texture->texture_index != 0 || texture->sampler_index != 0 ||
       texture->backend_flags != 0)
      return false;
   return true;
}

static bool
pvrgpu_terrain_d4_d5_coord_matches(const nir_tex_instr *texture,
                                    const nir_intrinsic_instr *varying,
                                    unsigned active_component,
                                    const nir_def *offset,
                                    const nir_def *zero)
{
   if (!texture || !varying || active_component >= 2 || !offset || !zero)
      return false;
   const int coord_index =
      nir_tex_instr_src_index(texture, nir_tex_src_coord);
   if (coord_index < 0 || !texture->src[coord_index].src.ssa ||
       nir_def_instr(texture->src[coord_index].src.ssa)->type !=
          nir_instr_type_alu)
      return false;
   nir_alu_instr *add = nir_instr_as_alu(
      nir_def_instr(texture->src[coord_index].src.ssa));
   if (add->op != nir_op_fadd || add->def.bit_size != 32 ||
       add->def.num_components != 2)
      return false;

   for (unsigned varying_source = 0; varying_source < 2;
        ++varying_source) {
      const unsigned delta_source = 1U - varying_source;
      if (!pvrgpu_terrain_d4_d6_source_is_def_slice(
             &add->src[varying_source], &varying->def, 0, 2))
         continue;
      bool delta_matches = true;
      for (unsigned component = 0; component < 2; ++component) {
         const nir_def *expected =
            component == active_component ? offset : zero;
         if (!pvrgpu_terrain_d4_d6_scalar_is_def(
                pvrgpu_terrain_d4_d6_resolve_source(
                   &add->src[delta_source], component),
                expected,
                0)) {
            delta_matches = false;
            break;
         }
      }
      if (delta_matches)
         return true;
   }
   return false;
}

static bool
pvrgpu_terrain_d4_d6_store_matches(const nir_intrinsic_instr *store,
                                    const nir_def *value,
                                    unsigned component)
{
   if (!store || store->intrinsic != nir_intrinsic_frag_store_pco ||
       !store->src[0].ssa || component >= value->num_components)
      return false;
   const nir_scalar actual =
      nir_scalar_resolved(store->src[0].ssa, 0);
   return pvrgpu_terrain_d4_d6_scalar_is_def(actual, value, component);
}

static bool
pvrgpu_terrain_d4_d5_collect_graph(
   nir_shader *nir,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_terrain_d4_d5_graph *graph,
   char *error,
   size_t error_size)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT || !graph ||
       (profile != PVRGPU_PCO_TERRAIN_D4 &&
        profile != PVRGPU_PCO_TERRAIN_D5))
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4/D5 mediump profile mismatch");

   memset(graph, 0, sizeof(*graph));
   graph->entrypoint = nir_shader_get_entrypoint(nir);
   if (!graph->entrypoint)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4/D5 mediump entrypoint is missing");

   const uint32_t step_bits =
      profile == PVRGPU_PCO_TERRAIN_D4 ? UINT32_C(0x3c4ccccd) :
                                        UINT32_C(0x3b7ffbce);
   const uint32_t step_bits_800x600 =
      profile == PVRGPU_PCO_TERRAIN_D4 ? UINT32_C(0x3aa3d70a) : step_bits;
   unsigned constants = 0;
   unsigned textures = 0;
   unsigned frag_stores = 0;
   unsigned fabs_count = 0;
   unsigned fadd_count = 0;
   unsigned fdiv_count = 0;
   unsigned fmul_count = 0;
   unsigned fneg_count = 0;
   unsigned vec2_count = 0;
   unsigned vec4_count = 0;

   nir_foreach_block (block, graph->entrypoint) {
      nir_foreach_instr (instr, block) {
         switch (instr->type) {
         case nir_instr_type_load_const: {
            nir_load_const_instr *constant = nir_instr_as_load_const(instr);
            if (constant->def.bit_size != 32 ||
                constant->def.num_components != 1)
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D4/D5 constant shape changed");
            ++constants;
            nir_load_const_instr **slot = NULL;
            switch (constant->value[0].u32) {
            case UINT32_C(0x00000000):
               slot = &graph->zero;
               break;
            case UINT32_C(0x3f800000):
               slot = &graph->one;
               break;
            case UINT32_C(0x40000000):
               slot = &graph->two;
               break;
            case UINT32_C(0x3e40214b):
               slot = &graph->weight[0];
               break;
            case UINT32_C(0x3e53037d):
               slot = &graph->weight[1];
               break;
            case UINT32_C(0x3e59b62c):
               slot = &graph->weight[2];
               break;
            default:
               if (constant->value[0].u32 == step_bits ||
                   constant->value[0].u32 == step_bits_800x600)
                  slot = &graph->step;
               break;
            }
            if (!slot || *slot)
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D4/D5 constant set changed");
            *slot = constant;
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               nir_variable *var = nir_intrinsic_get_var(intr, 0);
               if (graph->varying || !var ||
                   var->data.mode != nir_var_shader_in ||
                   var->data.location != VARYING_SLOT_VAR0 ||
                   intr->def.bit_size != 32 ||
                   intr->def.num_components != 2)
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain D4/D5 varying graph changed");
               graph->varying = intr;
            } else if (intr->intrinsic == nir_intrinsic_frag_store_pco) {
               const unsigned base = nir_intrinsic_base(intr);
               if (base >= ARRAY_SIZE(graph->store) ||
                   graph->store[base])
                  return pvrgpu_pco_fail(
                     error,
                     error_size,
                     "terrain D4/D5 fragment store graph changed");
               graph->store[base] = intr;
               ++frag_stores;
            }
            break;
         }
         case nir_instr_type_tex:
            if (textures >= ARRAY_SIZE(graph->texture) ||
                !pvrgpu_terrain_d4_d5_texture_is_canonical(
                   nir_instr_as_tex(instr)))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D4/D5 texture graph changed");
            graph->texture[textures++] = nir_instr_as_tex(instr);
            break;
         case nir_instr_type_alu:
            switch (nir_instr_as_alu(instr)->op) {
            case nir_op_fabs:
               ++fabs_count;
               graph->absolute = nir_instr_as_alu(instr);
               break;
            case nir_op_fadd:
               ++fadd_count;
               break;
            case nir_op_fdiv:
               ++fdiv_count;
               graph->scaled_divide = nir_instr_as_alu(instr);
               break;
            case nir_op_fmul:
               ++fmul_count;
               break;
            case nir_op_fneg:
               ++fneg_count;
               break;
            case nir_op_vec2:
               ++vec2_count;
               break;
            case nir_op_vec4:
               ++vec4_count;
               break;
            case nir_op_mov:
               /* pvrgpu_canonicalize_fragment_output() may add channels. */
               break;
            case nir_op_ffma:
            case nir_op_ffma_weak:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D4/D5 fused arithmetic changed");
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D4/D5 ALU graph changed");
            }
            break;
         default:
            break;
         }
      }
   }

   if (constants != 7U || textures != 5U || frag_stores != 4U ||
       fabs_count != 1U || fadd_count != 8U || fdiv_count != 1U ||
       fmul_count != 7U || fneg_count != 2U || vec2_count != 4U ||
       vec4_count != 1U || !graph->varying || !graph->zero ||
       !graph->one || !graph->two || !graph->step ||
       !graph->weight[0] || !graph->weight[1] || !graph->weight[2] ||
       !graph->absolute || !graph->scaled_divide) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D4/D5 raw graph signature changed "
         "(const=%u tex=%u store=%u abs=%u add=%u div=%u mul=%u neg=%u)",
         constants,
         textures,
         frag_stores,
         fabs_count,
         fadd_count,
         fdiv_count,
         fmul_count,
         fneg_count);
   }

   if (graph->absolute->def.bit_size != 32 ||
       graph->absolute->def.num_components != 1 ||
       !pvrgpu_terrain_d4_d6_source_is_def_slice(
          &graph->absolute->src[0], &graph->varying->def, 1, 1))
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4/D5 tilt graph changed");

   graph->scaled_multiply = pvrgpu_terrain_d4_d6_find_binary(
      graph->entrypoint,
      nir_op_fmul,
      &graph->step->def,
      &graph->absolute->def,
      1);
   if (!graph->scaled_multiply ||
       graph->scaled_divide->def.bit_size != 32 ||
       graph->scaled_divide->def.num_components != 1 ||
       !pvrgpu_terrain_d4_d6_source_is_def_slice(
          &graph->scaled_divide->src[0],
          &graph->scaled_multiply->def,
          0,
          1) ||
       !pvrgpu_terrain_d4_d6_source_is_def_slice(
          &graph->scaled_divide->src[1], &graph->one->def, 0, 1))
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4/D5 scaled-step graph changed");

   graph->twice = pvrgpu_terrain_d4_d6_find_binary(
      graph->entrypoint,
      nir_op_fmul,
      &graph->two->def,
      &graph->scaled_divide->def,
      1);
   graph->negative_twice = pvrgpu_terrain_d4_d6_find_fneg(
      graph->entrypoint, graph->twice ? &graph->twice->def : NULL);
   graph->negative_scaled = pvrgpu_terrain_d4_d6_find_fneg(
      graph->entrypoint, &graph->scaled_divide->def);
   if (!graph->twice || !graph->negative_twice ||
       !graph->negative_scaled)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4/D5 signed-offset graph changed");

   const unsigned active_component =
      profile == PVRGPU_PCO_TERRAIN_D4 ? 0U : 1U;
   const nir_def *offset[5] = {
      &graph->negative_twice->def,
      &graph->negative_scaled->def,
      NULL,
      &graph->scaled_divide->def,
      &graph->twice->def,
   };
   for (unsigned tap = 0; tap < ARRAY_SIZE(graph->texture); ++tap) {
      const int coord_index = nir_tex_instr_src_index(
         graph->texture[tap], nir_tex_src_coord);
      if ((tap == 2U &&
           (coord_index < 0 ||
            graph->texture[tap]->src[coord_index].src.ssa !=
               &graph->varying->def)) ||
          (tap != 2U &&
           !pvrgpu_terrain_d4_d5_coord_matches(
              graph->texture[tap],
              graph->varying,
              active_component,
              offset[tap],
              &graph->zero->def))) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D4/D5 tap %u coordinate changed",
                                tap);
      }
   }

   const unsigned weight_index[5] = { 0U, 1U, 2U, 1U, 0U };
   for (unsigned tap = 0; tap < ARRAY_SIZE(graph->term); ++tap) {
      graph->term[tap] =
         pvrgpu_terrain_d4_d6_find_vector_times_splat(
            graph->entrypoint,
            &graph->texture[tap]->def,
            &graph->weight[weight_index[tap]]->def,
            4);
      if (!graph->term[tap])
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D4/D5 tap %u weight changed",
                                tap);
   }

   graph->sum[0] = pvrgpu_terrain_d4_d6_find_binary(
      graph->entrypoint,
      nir_op_fadd,
      &graph->term[0]->def,
      &graph->term[1]->def,
      4);
   for (unsigned tap = 2; tap < ARRAY_SIZE(graph->term); ++tap) {
      graph->sum[tap - 1U] = pvrgpu_terrain_d4_d6_find_binary(
         graph->entrypoint,
         nir_op_fadd,
         graph->sum[tap - 2U] ? &graph->sum[tap - 2U]->def : NULL,
         &graph->term[tap]->def,
         4);
   }
   if (!graph->sum[0] || !graph->sum[1] ||
       !graph->sum[2] || !graph->sum[3])
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4/D5 accumulation order changed");

   for (unsigned component = 0; component < 3; ++component) {
      if (!pvrgpu_terrain_d4_d6_store_matches(
             graph->store[component], &graph->sum[3]->def, component))
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D4/D5 RGB output graph changed");
   }
   if (!pvrgpu_terrain_d4_d6_store_matches(
          graph->store[3], &graph->one->def, 0))
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4/D5 alpha graph changed");

   return true;
}

static bool
pvrgpu_terrain_d4_d6_validate_lowered_counts(
   nir_function_impl *entrypoint,
   const struct pvrgpu_terrain_d4_d6_lowered_counts *expected,
   const char *name,
   char *error,
   size_t error_size)
{
   struct pvrgpu_terrain_d4_d6_lowered_counts observed = { 0 };
   nir_foreach_block (block, entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_load_const) {
            ++observed.constants;
         } else if (instr->type == nir_instr_type_tex) {
            ++observed.textures;
         } else if (instr->type == nir_instr_type_intrinsic) {
            if (nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_frag_store_pco)
               ++observed.frag_stores;
         } else if (instr->type == nir_instr_type_alu) {
            switch (nir_instr_as_alu(instr)->op) {
            case nir_op_fabs:
               ++observed.fabs;
               break;
            case nir_op_fadd:
               ++observed.fadd;
               break;
            case nir_op_fdiv:
               ++observed.fdiv;
               break;
            case nir_op_fmul:
               ++observed.fmul;
               break;
            case nir_op_fneg:
               ++observed.fneg;
               break;
            case nir_op_vec2:
               ++observed.vec2;
               break;
            case nir_op_vec4:
               ++observed.vec4;
               break;
            case nir_op_f2f16_rtne:
               ++observed.f2f16_rtne;
               break;
            case nir_op_f2f16_rtz:
               ++observed.f2f16_rtz;
               break;
            case nir_op_f2f32:
               ++observed.f2f32;
               break;
            case nir_op_mov:
               break;
            case nir_op_ffma:
            case nir_op_ffma_weak:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s retained fused arithmetic",
                                      name);
            default:
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "%s lowered ALU graph changed",
                                      name);
            }
         }
      }
   }

   if (memcmp(&observed, expected, sizeof(observed)) != 0) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "%s lowered count changed "
         "(const=%u tex=%u store=%u abs=%u add=%u div=%u mul=%u neg=%u "
         "vec2=%u vec4=%u rtne=%u rtz=%u widen=%u)",
         name,
         observed.constants,
         observed.textures,
         observed.frag_stores,
         observed.fabs,
         observed.fadd,
         observed.fdiv,
         observed.fmul,
         observed.fneg,
         observed.vec2,
         observed.vec4,
         observed.f2f16_rtne,
         observed.f2f16_rtz,
         observed.f2f32);
   }
   return true;
}

static nir_def *
pvrgpu_terrain_d4_d6_round_texture_component_rtz(nir_builder *builder,
                                                  nir_tex_instr *texture,
                                                  unsigned component)
{
   return nir_f2f32(
      builder,
      nir_f2f16_rtz(builder, nir_channel(builder, &texture->def, component)));
}

static bool
pvrgpu_lower_terrain_d4_d5_fragment_mediump(
   nir_shader *nir,
   enum pvrgpu_pco_terrain_profile profile,
   char *error,
   size_t error_size)
{
   struct pvrgpu_terrain_d4_d5_graph graph;
   if (!pvrgpu_terrain_d4_d5_collect_graph(nir,
                                            profile,
                                            &graph,
                                            error,
                                            error_size))
      return false;

   const unsigned active_component =
      profile == PVRGPU_PCO_TERRAIN_D4 ? 0U : 1U;
   nir_builder common =
      nir_builder_at(nir_before_instr(&graph.texture[0]->instr));

   /* Six source arithmetic constants enter the mediump island through RTNE.
    * The new -2 literal below is already exactly representable in binary16;
    * it corresponds to Gallivm's separate half constant and intentionally is
    * not derived through fneg(+2), which would add an extra fneg absent from
    * the captured native operation sequence. */
   nir_def *one_half = pvrgpu_terrain_d1_round_half_rtne(
      &common, &graph.one->def);
   nir_def *two_half = pvrgpu_terrain_d1_round_half_rtne(
      &common, &graph.two->def);
   nir_def *step_half = pvrgpu_terrain_d1_round_half_rtne(
      &common, &graph.step->def);
   nir_def *weight_half[3];
   for (unsigned weight = 0; weight < ARRAY_SIZE(weight_half); ++weight) {
      weight_half[weight] = pvrgpu_terrain_d1_round_half_rtne(
         &common, &graph.weight[weight]->def);
   }

   nir_def *base_half = pvrgpu_terrain_d1_round_half_rtne(
      &common,
      nir_channel(&common, &graph.varying->def, active_component));
   nir_def *tilt_half =
      active_component == 1U
         ? base_half
         : pvrgpu_terrain_d1_round_half_rtne(
              &common, nir_channel(&common, &graph.varying->def, 1));
   nir_def *tilt = nir_fabs(&common, tilt_half);
   nir_def *scaled_product = pvrgpu_terrain_d1_round_half_rtne(
      &common, nir_fmul(&common, step_half, tilt));
   nir_def *scaled = pvrgpu_terrain_d1_round_half_rtne(
      &common, nir_fdiv(&common, scaled_product, one_half));
   nir_def *negative_scaled = nir_fneg(&common, scaled);
   nir_def *negative_twice = pvrgpu_terrain_d1_round_half_rtne(
      &common,
      nir_fmul(&common, nir_imm_float(&common, -2.0f), scaled));
   nir_def *positive_twice = pvrgpu_terrain_d1_round_half_rtne(
      &common, nir_fmul(&common, two_half, scaled));

   nir_def *offset[5] = {
      negative_twice,
      negative_scaled,
      NULL,
      scaled,
      positive_twice,
   };
   const unsigned weight_index[5] = { 0U, 1U, 2U, 1U, 0U };
   nir_def *accumulator[3] = { 0 };
   for (unsigned tap = 0; tap < ARRAY_SIZE(graph.texture); ++tap) {
      const int coord_index = nir_tex_instr_src_index(
         graph.texture[tap], nir_tex_src_coord);
      if (tap == 2U) {
         nir_src_rewrite(&graph.texture[tap]->src[coord_index].src,
                         &graph.varying->def);
      } else {
         nir_builder coord_builder =
            nir_builder_at(nir_before_instr(&graph.texture[tap]->instr));
         nir_def *active = pvrgpu_terrain_d1_round_half_rtne(
            &coord_builder,
            nir_fadd(&coord_builder, base_half, offset[tap]));
         nir_def *coord =
            active_component == 0U
               ? nir_vec2(&coord_builder,
                          active,
                          nir_channel(&coord_builder,
                                      &graph.varying->def,
                                      1))
               : nir_vec2(&coord_builder,
                          nir_channel(&coord_builder,
                                      &graph.varying->def,
                                      0),
                          active);
         nir_src_rewrite(&graph.texture[tap]->src[coord_index].src, coord);
      }

      nir_builder sample_builder =
         nir_builder_at(nir_after_instr(&graph.texture[tap]->instr));
      for (unsigned component = 0; component < 3; ++component) {
         nir_def *sample_half =
            pvrgpu_terrain_d4_d6_round_texture_component_rtz(
               &sample_builder, graph.texture[tap], component);
         nir_def *term = pvrgpu_terrain_d1_round_half_rtne(
            &sample_builder,
            nir_fmul(&sample_builder,
                     sample_half,
                     weight_half[weight_index[tap]]));
         accumulator[component] =
            tap == 0U
               ? term
               : pvrgpu_terrain_d1_round_half_rtne(
                    &sample_builder,
                    nir_fadd(&sample_builder,
                             accumulator[component],
                             term));
      }
   }

   for (unsigned component = 0; component < 3; ++component)
      nir_src_rewrite(&graph.store[component]->src[0],
                      accumulator[component]);

   /* Split source 1.0 by precision domain: denominator is mediump above,
    * while the declared highp literal alpha reaches the framebuffer raw. */
   nir_builder alpha_builder =
      nir_builder_at(nir_before_instr(&graph.store[3]->instr));
   nir_src_rewrite(&graph.store[3]->src[0],
                   nir_imm_float(&alpha_builder, 1.0f));

   nir_opt_dce(nir);
   const struct pvrgpu_terrain_d4_d6_lowered_counts expected = {
      .constants = 8,
      .textures = 5,
      .frag_stores = 4,
      .fabs = 1,
      .fadd = 16,
      .fdiv = 1,
      .fmul = 18,
      .fneg = 1,
      .vec2 = 4,
      .f2f16_rtne = profile == PVRGPU_PCO_TERRAIN_D4 ? 43U : 42U,
      .f2f16_rtz = 15,
      .f2f32 = profile == PVRGPU_PCO_TERRAIN_D4 ? 58U : 57U,
   };
   if (!pvrgpu_terrain_d4_d6_validate_lowered_counts(
          graph.entrypoint,
          &expected,
          profile == PVRGPU_PCO_TERRAIN_D4 ? "terrain D4" : "terrain D5",
          error,
          error_size))
      return false;

   nir_progress(true, graph.entrypoint, nir_metadata_control_flow);
   return true;
}

struct pvrgpu_terrain_d6_graph {
   nir_function_impl *entrypoint;
   nir_intrinsic_instr *varying;
   nir_intrinsic_instr *opacity;
   nir_tex_instr *texture;
   nir_alu_instr *multiply;
   nir_intrinsic_instr *store[4];
};

static bool
pvrgpu_terrain_d6_collect_graph(nir_shader *nir,
                                 struct pvrgpu_terrain_d6_graph *graph,
                                 char *error,
                                 size_t error_size)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT || !graph)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D6 mediump requires a fragment shader");
   memset(graph, 0, sizeof(*graph));
   graph->entrypoint = nir_shader_get_entrypoint(nir);
   if (!graph->entrypoint)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D6 mediump entrypoint is missing");

   /* Uniform-to-push lowering creates the byte-address immediate before the
    * load and leaves the old vec4-slot zero dead until normal optimization. */
   nir_opt_dce(nir);

   unsigned constants = 0;
   unsigned textures = 0;
   unsigned multiplies = 0;
   unsigned frag_stores = 0;
   nir_load_const_instr *zero = NULL;
   nir_foreach_block (block, graph->entrypoint) {
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_load_const) {
            nir_load_const_instr *constant = nir_instr_as_load_const(instr);
            ++constants;
            if (zero || constant->def.bit_size != 32 ||
                constant->def.num_components != 1 ||
                constant->value[0].u32 != 0U)
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D6 push address changed");
            zero = constant;
         } else if (instr->type == nir_instr_type_tex) {
            if (graph->texture ||
                !pvrgpu_terrain_d4_d5_texture_is_canonical(
                   nir_instr_as_tex(instr)))
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D6 texture graph changed");
            graph->texture = nir_instr_as_tex(instr);
            ++textures;
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               nir_variable *var = nir_intrinsic_get_var(intr, 0);
               if (graph->varying || !var ||
                   var->data.mode != nir_var_shader_in ||
                   var->data.location != VARYING_SLOT_VAR0 ||
                   intr->def.bit_size != 32 ||
                   intr->def.num_components != 2)
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain D6 varying graph changed");
               graph->varying = intr;
            } else if (intr->intrinsic ==
                       nir_intrinsic_load_push_constant) {
               if (graph->opacity || intr->def.bit_size != 32 ||
                   intr->def.num_components != 1 ||
                   nir_intrinsic_base(intr) != 0 ||
                   nir_intrinsic_range(intr) != 16 ||
                   !nir_src_is_const(intr->src[0]) ||
                   nir_src_as_uint(intr->src[0]) != 0U)
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain D6 opacity push graph changed");
               graph->opacity = intr;
            } else if (intr->intrinsic == nir_intrinsic_frag_store_pco) {
               const unsigned base = nir_intrinsic_base(intr);
               if (base >= ARRAY_SIZE(graph->store) ||
                   graph->store[base])
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain D6 output graph changed");
               graph->store[base] = intr;
               ++frag_stores;
            }
         } else if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_fmul) {
               if (graph->multiply)
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "terrain D6 multiply is duplicated");
               graph->multiply = alu;
               ++multiplies;
            } else if (alu->op != nir_op_mov) {
               return pvrgpu_pco_fail(error,
                                      error_size,
                                      "terrain D6 ALU graph changed");
            }
         }
      }
   }

   if (constants != 1U || textures != 1U || multiplies != 1U ||
       frag_stores != 4U || !zero || !graph->varying ||
       !graph->opacity || !graph->texture || !graph->multiply) {
      return pvrgpu_pco_fail(
         error,
         error_size,
         "terrain D6 raw signature changed "
         "(const=%u tex=%u mul=%u store=%u)",
         constants,
         textures,
         multiplies,
         frag_stores);
   }

   const int coord_index =
      nir_tex_instr_src_index(graph->texture, nir_tex_src_coord);
   if (coord_index < 0 ||
       graph->texture->src[coord_index].src.ssa != &graph->varying->def ||
       graph->multiply->def.bit_size != 32 ||
       graph->multiply->def.num_components != 4)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D6 sample/coordinate graph changed");

   bool multiply_matches = false;
   for (unsigned texture_source = 0; texture_source < 2;
        ++texture_source) {
      const unsigned opacity_source = 1U - texture_source;
      if (pvrgpu_terrain_d4_d6_source_is_def_slice(
             &graph->multiply->src[texture_source],
             &graph->texture->def,
             0,
             4) &&
          pvrgpu_terrain_d4_d6_source_is_splat_def(
             &graph->multiply->src[opacity_source],
             &graph->opacity->def,
             0,
             4)) {
         multiply_matches = true;
         break;
      }
   }
   if (!multiply_matches)
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D6 modulation graph changed");
   for (unsigned component = 0; component < 4; ++component) {
      if (!pvrgpu_terrain_d4_d6_store_matches(
             graph->store[component], &graph->multiply->def, component))
         return pvrgpu_pco_fail(error,
                                error_size,
                                "terrain D6 output component order changed");
   }
   return true;
}

static bool
pvrgpu_lower_terrain_d6_fragment_mediump(nir_shader *nir,
                                          char *error,
                                          size_t error_size)
{
   struct pvrgpu_terrain_d6_graph graph;
   if (!pvrgpu_terrain_d6_collect_graph(nir, &graph, error, error_size))
      return false;

   nir_builder opacity_builder =
      nir_builder_at(nir_after_instr(&graph.opacity->instr));
   nir_def *opacity_half = pvrgpu_terrain_d1_round_half_rtne(
      &opacity_builder, &graph.opacity->def);

   nir_builder sample_builder =
      nir_builder_at(nir_after_instr(&graph.texture->instr));
   for (unsigned component = 0; component < 4; ++component) {
      nir_def *sample_half =
         pvrgpu_terrain_d4_d6_round_texture_component_rtz(
            &sample_builder, graph.texture, component);
      nir_def *product = pvrgpu_terrain_d1_round_half_rtne(
         &sample_builder,
         nir_fmul(&sample_builder, sample_half, opacity_half));
      nir_src_rewrite(&graph.store[component]->src[0], product);
   }

   nir_opt_dce(nir);
   const struct pvrgpu_terrain_d4_d6_lowered_counts expected = {
      .constants = 1,
      .textures = 1,
      .frag_stores = 4,
      .fmul = 4,
      .f2f16_rtne = 5,
      .f2f16_rtz = 4,
      .f2f32 = 9,
   };
   if (!pvrgpu_terrain_d4_d6_validate_lowered_counts(
          graph.entrypoint,
          &expected,
          "terrain D6",
          error,
          error_size))
      return false;

   nir_progress(true, graph.entrypoint, nir_metadata_control_flow);
   return true;
}

static bool
pvrgpu_lower_terrain_d4_d6_fragment_mediump(
   nir_shader *nir,
   enum pvrgpu_pco_terrain_profile profile,
   char *error,
   size_t error_size)
{
   switch (profile) {
   case PVRGPU_PCO_TERRAIN_D4:
   case PVRGPU_PCO_TERRAIN_D5:
      return pvrgpu_lower_terrain_d4_d5_fragment_mediump(nir,
                                                          profile,
                                                          error,
                                                          error_size);
   case PVRGPU_PCO_TERRAIN_D6:
      return pvrgpu_lower_terrain_d6_fragment_mediump(nir,
                                                       error,
                                                       error_size);
   default:
      return pvrgpu_pco_fail(error,
                             error_size,
                             "terrain D4-D6 mediump profile mismatch");
   }
}

/* Terrain D7/D8 native-mediump lowering fragment.
 *
 * Integration point: place this after pvrgpu_terrain_d1_round_half_rtne()
 * and the D3 helpers in pvrgpu_pco.c, then add the following arm to the
 * mediump dispatch immediately before pco_preprocess_nir():
 *
 *   ((profile == PVRGPU_PCO_TERRAIN_D7 ||
 *     profile == PVRGPU_PCO_TERRAIN_D8) &&
 *    !pvrgpu_lower_terrain_d7d8_fragment_mediump(fs,
 *                                                 profile,
 *                                                 error,
 *                                                 error_size))
 *
 * The source-hash and canonical-signature checks in
 * pvrgpu_pco_compile_terrain() remain the outer trust boundary.  This pass
 * additionally matches every source ALU and all nine texture/data-flow edges
 * before replacing the widened fp32 graph.  Any compiler-shape drift fails
 * closed.
 */

#define PVRGPU_TERRAIN_D7D8_TAPS 9U
#define PVRGPU_TERRAIN_D7D8_COLOR_COMPONENTS 3U

struct pvrgpu_terrain_d7d8_graph {
   nir_function_impl *entrypoint;
   nir_intrinsic_instr *varying;
   nir_intrinsic_instr *stores[4];
   nir_tex_instr *textures[PVRGPU_TERRAIN_D7D8_TAPS];

   nir_alu_instr *fadds[17];
   nir_alu_instr *fmuls[13];
   nir_alu_instr *fnegs[5];
   nir_alu_instr *vec2s[8];
   nir_alu_instr *movs[4];
   nir_alu_instr *fabs;
   nir_alu_instr *fdiv;
   nir_alu_instr *vec4;

   nir_alu_instr *tilt_neg;
   nir_alu_instr *tilt_add;
   nir_alu_instr *scale_mul;
   nir_alu_instr *magnitudes[3];       /* 2*t, 3*t, 4*t. */
   nir_alu_instr *negative_offsets[4]; /* -4*t, -3*t, -2*t, -t. */
   nir_alu_instr *coord_adds[8];
   nir_alu_instr *offset_vecs[8];
   nir_alu_instr *products[PVRGPU_TERRAIN_D7D8_TAPS];
   nir_alu_instr *accumulator_adds[PVRGPU_TERRAIN_D7D8_TAPS - 1U];
   bool step_800x600;
};

struct pvrgpu_terrain_d7d8_lower_counts {
   unsigned textures;
   unsigned offset_coords;
   unsigned input_rtne;
   unsigned texture_rtz_lanes;
   unsigned add_rtne;
   unsigned mul_rtne;
   unsigned div_rtne;
   unsigned exact_negates;
   unsigned exact_absolutes;
   unsigned weighted_multiplies;
   unsigned accumulator_adds;
   unsigned widened_values;
   unsigned architectural_coord_exits;
   unsigned architectural_color_exits;
};

static nir_scalar
pvrgpu_terrain_d7d8_alu_scalar(const nir_alu_src *source, unsigned component)
{
   if (!source || !source->src.ssa || component >= NIR_MAX_VEC_COMPONENTS)
      return (nir_scalar){0};
   return nir_scalar_resolved(source->src.ssa, source->swizzle[component]);
}

static bool
pvrgpu_terrain_d7d8_alu_source_is_def(const nir_alu_src *source,
                                      const nir_def *def, unsigned components)
{
   if (!source || !source->src.ssa || !def || def->num_components < components)
      return false;

   for (unsigned component = 0; component < components; ++component) {
      const nir_scalar actual =
         pvrgpu_terrain_d7d8_alu_scalar(source, component);
      const nir_scalar expected = {
         .def = (nir_def *)def,
         .comp = component,
      };
      if (!nir_scalar_equal(actual, expected))
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d7d8_alu_source_is_splat_bits(const nir_alu_src *source,
                                             uint32_t bits, unsigned components)
{
   if (!source || !source->src.ssa)
      return false;
   for (unsigned component = 0; component < components; ++component) {
      const nir_scalar scalar =
         pvrgpu_terrain_d7d8_alu_scalar(source, component);
      if (!nir_scalar_is_const(scalar) || nir_scalar_as_uint(scalar) != bits)
         return false;
   }
   return true;
}

static bool
pvrgpu_terrain_d7d8_match_def_and_bits(const nir_alu_instr *alu,
                                       const nir_def *def, uint32_t bits,
                                       unsigned components)
{
   if (!alu)
      return false;
   return (pvrgpu_terrain_d7d8_alu_source_is_def(&alu->src[0], def,
                                                 components) &&
           pvrgpu_terrain_d7d8_alu_source_is_splat_bits(&alu->src[1], bits,
                                                        components)) ||
          (pvrgpu_terrain_d7d8_alu_source_is_def(&alu->src[1], def,
                                                 components) &&
           pvrgpu_terrain_d7d8_alu_source_is_splat_bits(&alu->src[0], bits,
                                                        components));
}

static bool
pvrgpu_terrain_d7d8_match_ordered_defs(const nir_alu_instr *alu,
                                       const nir_def *left,
                                       const nir_def *right,
                                       unsigned components)
{
   return alu &&
          pvrgpu_terrain_d7d8_alu_source_is_def(&alu->src[0], left,
                                                components) &&
          pvrgpu_terrain_d7d8_alu_source_is_def(&alu->src[1], right,
                                                components);
}

static nir_alu_instr *
pvrgpu_terrain_d7d8_scalar_alu(nir_scalar scalar, nir_op op,
                               unsigned components)
{
   if (!scalar.def || scalar.comp != 0 || scalar.def->bit_size != 32 ||
       scalar.def->num_components != components ||
       nir_def_instr(scalar.def)->type != nir_instr_type_alu)
      return NULL;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(scalar.def));
   return alu->op == op ? alu : NULL;
}

static bool
pvrgpu_terrain_d7d8_mark(nir_instr **recognized, unsigned *recognized_count,
                         nir_alu_instr *alu)
{
   return alu && pvrgpu_terrain_d1_instr_array_add(recognized, recognized_count,
                                                   50U, &alu->instr);
}

static bool
pvrgpu_terrain_d7d8_check_constant_set(const nir_shader *nir, bool horizontal,
                                       char *error, size_t error_size)
{
   const uint32_t step_800x600 =
      horizontal ? UINT32_C(0x3aa3d70a) : UINT32_C(0x3ada7f3d);
   const uint32_t expected[12] = {
      UINT32_C(0x00000000), /* 0 */
      horizontal ? UINT32_C(0x3c4ccccd) : UINT32_C(0x3c88893b),
      UINT32_C(0x3d5edbf9), /* K4 */
      UINT32_C(0x3db4195d), /* K3 */
      UINT32_C(0x3dfdc619), /* K2 */
      UINT32_C(0x3e1be059), /* K1 */
      UINT32_C(0x3e26f156), /* K0 */
      UINT32_C(0x3f000000), /* 0.5 */
      UINT32_C(0x3f800000), /* 1 */
      UINT32_C(0x40000000), /* 2 */
      UINT32_C(0x40400000), /* 3 */
      UINT32_C(0x40800000), /* 4 */
   };
   uint16_t seen = 0;
   unsigned constants = 0;

   nir_foreach_function_impl(impl, nir)
   {
      nir_foreach_block(block, impl)
      {
         nir_foreach_instr(instr, block)
         {
            if (instr->type != nir_instr_type_load_const)
               continue;
            nir_load_const_instr *constant = nir_instr_as_load_const(instr);
            if (constant->def.bit_size != 32 ||
                constant->def.num_components != 1) {
               return pvrgpu_pco_fail(error, error_size,
                                      "terrain D7/D8 constant shape changed");
            }
            unsigned match =
               constant->value[0].u32 == step_800x600 ? 1U :
                                                        ARRAY_SIZE(expected);
            for (unsigned i = 0;
                 match == ARRAY_SIZE(expected) && i < ARRAY_SIZE(expected);
                 ++i) {
               if (constant->value[0].u32 == expected[i]) {
                  match = i;
                  break;
               }
            }
            if (match == ARRAY_SIZE(expected) || (seen & BITFIELD_BIT(match))) {
               return pvrgpu_pco_fail(error, error_size,
                                      "terrain D7/D8 constant set changed");
            }
            seen |= BITFIELD_BIT(match);
            ++constants;
         }
      }
   }

   if (constants != ARRAY_SIZE(expected) ||
       seen != BITFIELD_MASK(ARRAY_SIZE(expected))) {
      return pvrgpu_pco_fail(error, error_size,
                             "terrain D7/D8 constant count changed (%u)",
                             constants);
   }
   return true;
}

static bool
pvrgpu_terrain_d7d8_collect_source(nir_shader *nir,
                                   enum pvrgpu_pco_terrain_profile profile,
                                   struct pvrgpu_terrain_d7d8_graph *graph,
                                   char *error, size_t error_size)
{
   const bool horizontal = profile == PVRGPU_PCO_TERRAIN_D7;
   unsigned functions = 0;
   unsigned blocks = 0;
   unsigned derefs = 0;
   unsigned load_derefs = 0;
   unsigned frag_stores = 0;
   unsigned textures = 0;
   unsigned fadds = 0;
   unsigned fmuls = 0;
   unsigned fnegs = 0;
   unsigned vec2s = 0;
   unsigned movs = 0;
   unsigned fabs = 0;
   unsigned fdiv = 0;
   unsigned vec4 = 0;
   uint8_t store_mask = 0;

   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT ||
       (profile != PVRGPU_PCO_TERRAIN_D7 && profile != PVRGPU_PCO_TERRAIN_D8)) {
      return pvrgpu_pco_fail(error, error_size,
                             "terrain D7/D8 mediump profile mismatch");
   }
   memset(graph, 0, sizeof(*graph));

   nir_foreach_function(function, nir)
   {
      if (!function->impl)
         continue;
      if (!function->is_entrypoint || ++functions != 1U) {
         return pvrgpu_pco_fail(error, error_size,
                                "terrain D7/D8 requires one entrypoint");
      }
      graph->entrypoint = function->impl;
      nir_foreach_block(block, function->impl)
      {
         ++blocks;
         nir_foreach_instr(instr, block)
         {
            switch (instr->type) {
            case nir_instr_type_load_const:
               break;
            case nir_instr_type_deref:
               if (nir_instr_as_deref(instr)->deref_type != nir_deref_type_var)
                  goto source_changed;
               ++derefs;
               break;
            case nir_instr_type_intrinsic: {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_deref) {
                  nir_variable *var = nir_intrinsic_get_var(intr, 0);
                  if (++load_derefs != 1U || graph->varying || !var ||
                      var->data.mode != nir_var_shader_in ||
                      var->data.location != VARYING_SLOT_VAR0 ||
                      intr->def.bit_size != 32 || intr->def.num_components != 2)
                     goto source_changed;
                  graph->varying = intr;
               } else if (intr->intrinsic == nir_intrinsic_frag_store_pco) {
                  const unsigned base = nir_intrinsic_base(intr);
                  if (base >= ARRAY_SIZE(graph->stores) ||
                      (store_mask & BITFIELD_BIT(base)) || !intr->src[0].ssa ||
                      intr->src[0].ssa->bit_size != 32 ||
                      intr->src[0].ssa->num_components != 1)
                     goto source_changed;
                  graph->stores[base] = intr;
                  store_mask |= BITFIELD_BIT(base);
                  ++frag_stores;
               } else {
                  goto source_changed;
               }
               break;
            }
            case nir_instr_type_tex: {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (textures >= ARRAY_SIZE(graph->textures) ||
                   tex->op != nir_texop_tex || tex->texture_index != 0U ||
                   tex->sampler_index != 0U ||
                   tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
                   tex->dest_type != nir_type_float32 ||
                   tex->def.bit_size != 32 || tex->def.num_components != 4 ||
                   tex->coord_components != 2 || tex->num_srcs != 1 ||
                   tex->src[0].src_type != nir_tex_src_coord ||
                   !tex->src[0].src.ssa || tex->is_array || tex->is_shadow ||
                   tex->is_sparse || tex->texture_non_uniform ||
                   tex->sampler_non_uniform || tex->embedded_sampler ||
                   tex->offset_non_uniform || tex->backend_flags != 0)
                  goto source_changed;
               graph->textures[textures++] = tex;
               break;
            }
            case nir_instr_type_alu: {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               switch (alu->op) {
               case nir_op_fadd:
                  if (fadds >= ARRAY_SIZE(graph->fadds))
                     goto source_changed;
                  graph->fadds[fadds++] = alu;
                  break;
               case nir_op_fmul:
                  if (fmuls >= ARRAY_SIZE(graph->fmuls))
                     goto source_changed;
                  graph->fmuls[fmuls++] = alu;
                  break;
               case nir_op_fneg:
                  if (fnegs >= ARRAY_SIZE(graph->fnegs))
                     goto source_changed;
                  graph->fnegs[fnegs++] = alu;
                  break;
               case nir_op_vec2:
                  if (vec2s >= ARRAY_SIZE(graph->vec2s))
                     goto source_changed;
                  graph->vec2s[vec2s++] = alu;
                  break;
               case nir_op_mov:
                  if (movs >= ARRAY_SIZE(graph->movs))
                     goto source_changed;
                  graph->movs[movs++] = alu;
                  break;
               case nir_op_fabs:
                  if (fabs++)
                     goto source_changed;
                  graph->fabs = alu;
                  break;
               case nir_op_fdiv:
                  if (fdiv++)
                     goto source_changed;
                  graph->fdiv = alu;
                  break;
               case nir_op_vec4:
                  if (vec4++)
                     goto source_changed;
                  graph->vec4 = alu;
                  break;
               case nir_op_ffma:
               case nir_op_ffma_weak:
               default:
                  goto source_changed;
               }
               break;
            }
            default:
               goto source_changed;
            }
         }
      }
   }

   if (functions != 1U || blocks != 1U || derefs != 2U || load_derefs != 1U ||
       frag_stores != 4U || store_mask != 0x0f ||
       textures != ARRAY_SIZE(graph->textures) ||
       fadds != ARRAY_SIZE(graph->fadds) || fmuls != ARRAY_SIZE(graph->fmuls) ||
       fnegs != ARRAY_SIZE(graph->fnegs) || vec2s != ARRAY_SIZE(graph->vec2s) ||
       movs != ARRAY_SIZE(graph->movs) || fabs != 1U || fdiv != 1U ||
       vec4 != 1U || !graph->entrypoint || !graph->varying || !graph->fabs ||
       !graph->fdiv || !graph->vec4) {
      return pvrgpu_pco_fail(
         error, error_size,
         "terrain D7/D8 post-canonical signature changed "
         "(block=%u deref=%u load=%u store=%u tex=%u add=%u mul=%u "
         "neg=%u vec2=%u mov=%u)",
         blocks, derefs, load_derefs, frag_stores, textures, fadds, fmuls,
         fnegs, vec2s, movs);
   }
   if (!pvrgpu_terrain_d7d8_check_constant_set(nir, horizontal, error,
                                               error_size))
      return false;
   return true;

source_changed:
   return pvrgpu_pco_fail(error, error_size,
                          "terrain D7/D8 source instruction graph changed");
}

static bool
pvrgpu_terrain_d7d8_match_source(nir_shader *nir,
                                 enum pvrgpu_pco_terrain_profile profile,
                                 struct pvrgpu_terrain_d7d8_graph *graph,
                                 char *error, size_t error_size)
{
   const bool horizontal = profile == PVRGPU_PCO_TERRAIN_D7;
   const unsigned active = horizontal ? 0U : 1U;
   const uint32_t source_step =
      horizontal ? UINT32_C(0x3c4ccccd) : UINT32_C(0x3c88893b);
   const uint32_t source_step_800x600 =
      horizontal ? UINT32_C(0x3aa3d70a) : UINT32_C(0x3ada7f3d);
   static const uint32_t source_weights[PVRGPU_TERRAIN_D7D8_TAPS] = {
      UINT32_C(0x3d5edbf9), UINT32_C(0x3db4195d), UINT32_C(0x3dfdc619),
      UINT32_C(0x3e1be059), UINT32_C(0x3e26f156), UINT32_C(0x3e1be059),
      UINT32_C(0x3dfdc619), UINT32_C(0x3db4195d), UINT32_C(0x3d5edbf9),
   };
   static const uint32_t magnitude_bits[3] = {
      UINT32_C(0x40000000),
      UINT32_C(0x40400000),
      UINT32_C(0x40800000),
   };
   static const unsigned noncenter_taps[8] = {
      0U, 1U, 2U, 3U, 5U, 6U, 7U, 8U,
   };
   nir_instr *recognized[50] = {0};
   unsigned recognized_count = 0;

   if (!pvrgpu_terrain_d7d8_collect_source(nir, profile, graph, error,
                                           error_size))
      return false;

   /* step = stepConstant * abs(0.5 - uv.y) / 0.5 */
   if (graph->fdiv->def.bit_size != 32 ||
       graph->fdiv->def.num_components != 1 ||
       graph->fabs->def.bit_size != 32 ||
       graph->fabs->def.num_components != 1 ||
       !pvrgpu_terrain_d7d8_alu_source_is_splat_bits(&graph->fdiv->src[1],
                                                     UINT32_C(0x3f000000), 1))
      goto graph_changed;
   graph->scale_mul = pvrgpu_terrain_d7d8_scalar_alu(
      pvrgpu_terrain_d7d8_alu_scalar(&graph->fdiv->src[0], 0), nir_op_fmul, 1);
   if (!graph->scale_mul)
      goto graph_changed;
   const bool source_step_matches =
      pvrgpu_terrain_d7d8_match_def_and_bits(
         graph->scale_mul, &graph->fabs->def, source_step, 1);
   const bool source_step_800x600_matches =
      pvrgpu_terrain_d7d8_match_def_and_bits(
         graph->scale_mul, &graph->fabs->def, source_step_800x600, 1);
   if (source_step_matches == source_step_800x600_matches)
      goto graph_changed;
   graph->step_800x600 = source_step_800x600_matches;
   nir_scalar abs_source =
      pvrgpu_terrain_d7d8_alu_scalar(&graph->fabs->src[0], 0);
   graph->tilt_add = pvrgpu_terrain_d7d8_scalar_alu(abs_source, nir_op_fadd, 1);
   if (!graph->tilt_add)
      goto graph_changed;
   for (unsigned source = 0; source < 2; ++source) {
      if (!pvrgpu_terrain_d7d8_alu_source_is_splat_bits(
             &graph->tilt_add->src[source], UINT32_C(0x3f000000), 1))
         continue;
      graph->tilt_neg = pvrgpu_terrain_d7d8_scalar_alu(
         pvrgpu_terrain_d7d8_alu_scalar(&graph->tilt_add->src[1U - source], 0),
         nir_op_fneg, 1);
   }
   const nir_scalar tilt_source =
      graph->tilt_neg
         ? pvrgpu_terrain_d7d8_alu_scalar(&graph->tilt_neg->src[0], 0)
         : (nir_scalar){0};
   const nir_scalar expected_tilt_source = {
      .def = &graph->varying->def,
      .comp = 1U,
   };
   if (!graph->tilt_neg || !nir_scalar_equal(tilt_source, expected_tilt_source))
      goto graph_changed;

   if (!pvrgpu_terrain_d7d8_mark(recognized, &recognized_count, graph->fdiv) ||
       !pvrgpu_terrain_d7d8_mark(recognized, &recognized_count,
                                 graph->scale_mul) ||
       !pvrgpu_terrain_d7d8_mark(recognized, &recognized_count, graph->fabs) ||
       !pvrgpu_terrain_d7d8_mark(recognized, &recognized_count,
                                 graph->tilt_add) ||
       !pvrgpu_terrain_d7d8_mark(recognized, &recognized_count,
                                 graph->tilt_neg))
      goto graph_changed;

   /* Match the shared positive 2*t, 3*t, 4*t nodes. */
   for (unsigned magnitude = 0; magnitude < 3; ++magnitude) {
      for (unsigned i = 0; i < ARRAY_SIZE(graph->fmuls); ++i) {
         if (!pvrgpu_terrain_d7d8_match_def_and_bits(
                graph->fmuls[i], &graph->fdiv->def, magnitude_bits[magnitude],
                1))
            continue;
         if (graph->fmuls[i]->def.bit_size != 32 ||
             graph->fmuls[i]->def.num_components != 1)
            goto graph_changed;
         if (graph->magnitudes[magnitude])
            goto graph_changed;
         graph->magnitudes[magnitude] = graph->fmuls[i];
      }
      if (!pvrgpu_terrain_d7d8_mark(recognized, &recognized_count,
                                    graph->magnitudes[magnitude]))
         goto graph_changed;
   }

   const nir_def *positive_offsets[4] = {
      &graph->magnitudes[2]->def,
      &graph->magnitudes[1]->def,
      &graph->magnitudes[0]->def,
      &graph->fdiv->def,
   };
   for (unsigned offset = 0; offset < 4; ++offset) {
      for (unsigned i = 0; i < ARRAY_SIZE(graph->fnegs); ++i) {
         nir_alu_instr *neg = graph->fnegs[i];
         if (!pvrgpu_terrain_d7d8_alu_source_is_def(
                &neg->src[0], positive_offsets[offset], 1))
            continue;
         if (neg->def.bit_size != 32 || neg->def.num_components != 1)
            goto graph_changed;
         if (graph->negative_offsets[offset])
            goto graph_changed;
         graph->negative_offsets[offset] = neg;
      }
      if (!pvrgpu_terrain_d7d8_mark(recognized, &recognized_count,
                                    graph->negative_offsets[offset]))
         goto graph_changed;
   }

   /* Center is the unmodified highp coordinate.  Every other coordinate is
    * one vec2 delta plus one vec2 fadd; only the active axis is offset. */
   if (graph->textures[4]->src[0].src.ssa != &graph->varying->def)
      goto graph_changed;
   const nir_def *expected_offsets[PVRGPU_TERRAIN_D7D8_TAPS] = {
      &graph->negative_offsets[0]->def,
      &graph->negative_offsets[1]->def,
      &graph->negative_offsets[2]->def,
      &graph->negative_offsets[3]->def,
      NULL,
      &graph->fdiv->def,
      &graph->magnitudes[0]->def,
      &graph->magnitudes[1]->def,
      &graph->magnitudes[2]->def,
   };
   for (unsigned slot = 0; slot < ARRAY_SIZE(noncenter_taps); ++slot) {
      const unsigned tap = noncenter_taps[slot];
      nir_def *coord_def = graph->textures[tap]->src[0].src.ssa;
      if (!coord_def || coord_def->bit_size != 32 ||
          coord_def->num_components != 2 ||
          nir_def_instr(coord_def)->type != nir_instr_type_alu)
         goto graph_changed;
      nir_alu_instr *coord = nir_instr_as_alu(nir_def_instr(coord_def));
      if (coord->op != nir_op_fadd)
         goto graph_changed;

      unsigned varying_source = 2U;
      for (unsigned source = 0; source < 2; ++source) {
         if (pvrgpu_terrain_d7d8_alu_source_is_def(&coord->src[source],
                                                   &graph->varying->def, 2)) {
            varying_source = source;
            break;
         }
      }
      if (varying_source == 2U)
         goto graph_changed;
      nir_def *delta_def = coord->src[1U - varying_source].src.ssa;
      if (!delta_def || delta_def->bit_size != 32 ||
          delta_def->num_components != 2 ||
          nir_def_instr(delta_def)->type != nir_instr_type_alu)
         goto graph_changed;
      nir_alu_instr *delta = nir_instr_as_alu(nir_def_instr(delta_def));
      if (delta->op != nir_op_vec2)
         goto graph_changed;

      const nir_alu_src *delta_source = &coord->src[1U - varying_source];
      const nir_scalar active_delta =
         pvrgpu_terrain_d7d8_alu_scalar(delta_source, active);
      const nir_scalar expected_delta = {
         .def = (nir_def *)expected_offsets[tap],
         .comp = 0,
      };
      const nir_scalar inactive_delta =
         pvrgpu_terrain_d7d8_alu_scalar(delta_source, 1U - active);
      if (!nir_scalar_equal(active_delta, expected_delta) ||
          !nir_scalar_is_const(inactive_delta) ||
          nir_scalar_as_uint(inactive_delta) != 0U)
         goto graph_changed;

      graph->coord_adds[slot] = coord;
      graph->offset_vecs[slot] = delta;
      if (!pvrgpu_terrain_d7d8_mark(recognized, &recognized_count, coord) ||
          !pvrgpu_terrain_d7d8_mark(recognized, &recognized_count, delta))
         goto graph_changed;
   }

   /* Each texture has exactly one vector product with the source fp32
    * coefficient, followed by an ordered, left-associated sum. */
   for (unsigned tap = 0; tap < PVRGPU_TERRAIN_D7D8_TAPS; ++tap) {
      for (unsigned i = 0; i < ARRAY_SIZE(graph->fmuls); ++i) {
         nir_alu_instr *mul = graph->fmuls[i];
         if (mul->def.bit_size != 32 || mul->def.num_components != 4 ||
             !pvrgpu_terrain_d7d8_match_def_and_bits(
                mul, &graph->textures[tap]->def, source_weights[tap], 4))
            continue;
         if (graph->products[tap])
            goto graph_changed;
         graph->products[tap] = mul;
      }
      if (!pvrgpu_terrain_d7d8_mark(recognized, &recognized_count,
                                    graph->products[tap]))
         goto graph_changed;
   }

   const nir_def *accumulator = &graph->products[0]->def;
   for (unsigned tap = 1; tap < PVRGPU_TERRAIN_D7D8_TAPS; ++tap) {
      nir_alu_instr *matched = NULL;
      for (unsigned i = 0; i < ARRAY_SIZE(graph->fadds); ++i) {
         nir_alu_instr *add = graph->fadds[i];
         if (add->def.bit_size != 32 || add->def.num_components != 4 ||
             !pvrgpu_terrain_d7d8_match_ordered_defs(
                add, accumulator, &graph->products[tap]->def, 4))
            continue;
         if (matched)
            goto graph_changed;
         matched = add;
      }
      graph->accumulator_adds[tap - 1U] = matched;
      if (!pvrgpu_terrain_d7d8_mark(recognized, &recognized_count, matched))
         goto graph_changed;
      accumulator = &matched->def;
   }

   if (graph->vec4->def.bit_size != 32 || graph->vec4->def.num_components != 4)
      goto graph_changed;
   for (unsigned component = 0;
        component < PVRGPU_TERRAIN_D7D8_COLOR_COMPONENTS; ++component) {
      const nir_scalar actual =
         pvrgpu_terrain_d7d8_alu_scalar(&graph->vec4->src[component], 0);
      const nir_scalar expected = {
         .def = (nir_def *)accumulator,
         .comp = component,
      };
      if (!nir_scalar_equal(actual, expected))
         goto graph_changed;
   }
   if (!pvrgpu_terrain_d7d8_alu_source_is_splat_bits(&graph->vec4->src[3],
                                                     UINT32_C(0x3f800000), 1) ||
       !pvrgpu_terrain_d7d8_mark(recognized, &recognized_count, graph->vec4))
      goto graph_changed;

   /* canonicalize_fragment_output() creates one scalar mov per component. */
   for (unsigned component = 0; component < 4; ++component) {
      nir_def *store_value = graph->stores[component]->src[0].ssa;
      if (!store_value ||
          nir_def_instr(store_value)->type != nir_instr_type_alu)
         goto graph_changed;
      nir_alu_instr *mov = nir_instr_as_alu(nir_def_instr(store_value));
      const nir_scalar actual = nir_scalar_resolved(store_value, 0);
      const nir_scalar expected =
         nir_scalar_resolved(&graph->vec4->def, component);
      if (mov->op != nir_op_mov || !nir_scalar_equal(actual, expected) ||
          !pvrgpu_terrain_d7d8_mark(recognized, &recognized_count, mov))
         goto graph_changed;
   }

   /* Exact coverage makes unexpected aliases, re-association, extra uses of
    * an ALU slot, and compiler-introduced operations fail closed. */
   for (unsigned i = 0; i < ARRAY_SIZE(graph->fadds); ++i) {
      if (!pvrgpu_terrain_d1_instr_array_contains(recognized, recognized_count,
                                                  &graph->fadds[i]->instr))
         goto graph_changed;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(graph->fmuls); ++i) {
      if (!pvrgpu_terrain_d1_instr_array_contains(recognized, recognized_count,
                                                  &graph->fmuls[i]->instr))
         goto graph_changed;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(graph->fnegs); ++i) {
      if (!pvrgpu_terrain_d1_instr_array_contains(recognized, recognized_count,
                                                  &graph->fnegs[i]->instr))
         goto graph_changed;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(graph->vec2s); ++i) {
      if (!pvrgpu_terrain_d1_instr_array_contains(recognized, recognized_count,
                                                  &graph->vec2s[i]->instr))
         goto graph_changed;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(graph->movs); ++i) {
      if (!pvrgpu_terrain_d1_instr_array_contains(recognized, recognized_count,
                                                  &graph->movs[i]->instr))
         goto graph_changed;
   }
   if (recognized_count != ARRAY_SIZE(recognized))
      goto graph_changed;
   return true;

graph_changed:
   return pvrgpu_pco_fail(error, error_size,
                          "terrain D7/D8 blur data-flow changed");
}

static nir_def *
pvrgpu_terrain_d7d8_input_rtne(nir_builder *builder, nir_def *value,
                               struct pvrgpu_terrain_d7d8_lower_counts *counts)
{
   ++counts->input_rtne;
   ++counts->widened_values;
   return pvrgpu_terrain_d1_round_half_rtne(builder, value);
}

static nir_def *
pvrgpu_terrain_d7d8_hadd(nir_builder *builder, nir_def *left, nir_def *right,
                         struct pvrgpu_terrain_d7d8_lower_counts *counts)
{
   ++counts->add_rtne;
   ++counts->widened_values;
   return pvrgpu_terrain_d1_round_half_rtne(builder,
                                            nir_fadd(builder, left, right));
}

static nir_def *
pvrgpu_terrain_d7d8_hmul(nir_builder *builder, nir_def *left, nir_def *right,
                         struct pvrgpu_terrain_d7d8_lower_counts *counts)
{
   ++counts->mul_rtne;
   ++counts->widened_values;
   return pvrgpu_terrain_d1_round_half_rtne(builder,
                                            nir_fmul(builder, left, right));
}

static nir_def *
pvrgpu_terrain_d7d8_hdiv(nir_builder *builder, nir_def *left, nir_def *right,
                         struct pvrgpu_terrain_d7d8_lower_counts *counts)
{
   ++counts->div_rtne;
   ++counts->widened_values;
   return pvrgpu_terrain_d1_round_half_rtne(builder,
                                            nir_fdiv(builder, left, right));
}

static nir_def *
pvrgpu_terrain_d7d8_hneg(nir_builder *builder, nir_def *value,
                         struct pvrgpu_terrain_d7d8_lower_counts *counts)
{
   /* Sign changes are exact for a widened binary16 value. */
   ++counts->exact_negates;
   return nir_fneg(builder, value);
}

static nir_def *
pvrgpu_terrain_d7d8_habs(nir_builder *builder, nir_def *value,
                         struct pvrgpu_terrain_d7d8_lower_counts *counts)
{
   /* Clearing the sign is exact for a widened binary16 value. */
   ++counts->exact_absolutes;
   return nir_fabs(builder, value);
}

static nir_def *
pvrgpu_terrain_d7d8_texture_rtz(nir_tex_instr *texture,
                                struct pvrgpu_terrain_d7d8_lower_counts *counts)
{
   nir_builder builder = nir_builder_at(nir_after_instr(&texture->instr));
   nir_def *components[4];
   for (unsigned component = 0; component < ARRAY_SIZE(components);
        ++component) {
      components[component] = nir_f2f32(
         &builder, nir_f2f16_rtz(&builder, nir_channel(&builder, &texture->def,
                                                       component)));
      ++counts->texture_rtz_lanes;
      ++counts->widened_values;
   }
   nir_def *replacement = nir_vec(&builder, components, 4);
   nir_def_rewrite_uses_after(&texture->def, replacement);
   return replacement;
}

static bool
pvrgpu_lower_terrain_d7d8_fragment_mediump(
   nir_shader *nir, enum pvrgpu_pco_terrain_profile profile, char *error,
   size_t error_size)
{
   const bool horizontal = profile == PVRGPU_PCO_TERRAIN_D7;
   const unsigned active = horizontal ? 0U : 1U;
   static const float half_weight_constants[5] = {
      0x1.bdcp-5f, /* 0x2af7, K4 */
      0x1.684p-4f, /* 0x2da1, K3 */
      0x1.fb8p-4f, /* 0x2fee, K2 */
      0x1.37cp-3f, /* 0x30df, K1 */
      0x1.4ep-3f,  /* 0x3138, K0 */
   };
   static const unsigned weight_index[PVRGPU_TERRAIN_D7D8_TAPS] = {
      0U, 1U, 2U, 3U, 4U, 3U, 2U, 1U, 0U,
   };
   struct pvrgpu_terrain_d7d8_graph graph;
   struct pvrgpu_terrain_d7d8_lower_counts counts = {0};

   if (!pvrgpu_terrain_d7d8_match_source(nir, profile, &graph, error,
                                         error_size))
      return false;

   nir_builder head = nir_builder_at(nir_after_instr(&graph.varying->instr));
   nir_def *original[2] = {
      nir_channel(&head, &graph.varying->def, 0),
      nir_channel(&head, &graph.varying->def, 1),
   };
   nir_def *active_half =
      pvrgpu_terrain_d7d8_input_rtne(&head, original[active], &counts);
   nir_def *tilt_half =
      active == 1U
         ? active_half
         : pvrgpu_terrain_d7d8_input_rtne(&head, original[1], &counts);

   /* These fp32 immediates exactly widen the captured binary16 constants. */
   nir_def *half = nir_imm_float(&head, 0.5f);         /* 0x3800 */
   nir_def *two = nir_imm_float(&head, 2.0f);          /* 0x4000 */
   nir_def *three = nir_imm_float(&head, 3.0f);        /* 0x4200 */
   nir_def *four = nir_imm_float(&head, 4.0f);         /* 0x4400 */
   nir_def *minus_two = nir_imm_float(&head, -2.0f);   /* 0xc000 */
   nir_def *minus_three = nir_imm_float(&head, -3.0f); /* 0xc200 */
   nir_def *minus_four = nir_imm_float(&head, -4.0f);  /* 0xc400 */
   const float step_constant_value =
      graph.step_800x600
         ? (horizontal ? 0x1.47cp-10f : 0x1.b50p-10f) /* 0x151f / 0x16d4 */
         : (horizontal ? 0x1.998p-7f : 0x1.110p-6f); /* 0x2266 / 0x2444 */
   nir_def *step_constant = nir_imm_float(&head, step_constant_value);
   nir_def *weight_constants[ARRAY_SIZE(half_weight_constants)];
   for (unsigned weight = 0; weight < ARRAY_SIZE(half_weight_constants);
        ++weight) {
      weight_constants[weight] =
         nir_imm_float(&head, half_weight_constants[weight]);
   }

   nir_def *tilt_delta = pvrgpu_terrain_d7d8_hadd(
      &head, half, pvrgpu_terrain_d7d8_hneg(&head, tilt_half, &counts),
      &counts);
   nir_def *distance = pvrgpu_terrain_d7d8_habs(&head, tilt_delta, &counts);
   nir_def *scaled =
      pvrgpu_terrain_d7d8_hmul(&head, step_constant, distance, &counts);
   nir_def *tap_step = pvrgpu_terrain_d7d8_hdiv(&head, scaled, half, &counts);
   nir_def *offset[PVRGPU_TERRAIN_D7D8_TAPS] = {
      pvrgpu_terrain_d7d8_hmul(&head, minus_four, tap_step, &counts),
      pvrgpu_terrain_d7d8_hmul(&head, minus_three, tap_step, &counts),
      pvrgpu_terrain_d7d8_hmul(&head, minus_two, tap_step, &counts),
      pvrgpu_terrain_d7d8_hneg(&head, tap_step, &counts),
      NULL,
      tap_step,
      pvrgpu_terrain_d7d8_hmul(&head, two, tap_step, &counts),
      pvrgpu_terrain_d7d8_hmul(&head, three, tap_step, &counts),
      pvrgpu_terrain_d7d8_hmul(&head, four, tap_step, &counts),
   };

   nir_def *accumulator[PVRGPU_TERRAIN_D7D8_COLOR_COMPONENTS] = {0};
   for (unsigned tap = 0; tap < PVRGPU_TERRAIN_D7D8_TAPS; ++tap) {
      if (tap != 4U) {
         nir_builder coord_builder =
            nir_builder_at(nir_before_instr(&graph.textures[tap]->instr));
         nir_def *active_coord = pvrgpu_terrain_d7d8_hadd(
            &coord_builder, active_half, offset[tap], &counts);
         nir_def *tap_coord =
            horizontal ? nir_vec2(&coord_builder, active_coord, original[1])
                       : nir_vec2(&coord_builder, original[0], active_coord);
         nir_src_rewrite(&graph.textures[tap]->src[0].src, tap_coord);
         ++counts.offset_coords;
         ++counts.architectural_coord_exits;
      }

      nir_def *sample =
         pvrgpu_terrain_d7d8_texture_rtz(graph.textures[tap], &counts);
      ++counts.textures;
      nir_builder sample_builder =
         nir_builder_at(nir_after_instr(nir_def_instr(sample)));
      for (unsigned component = 0;
           component < PVRGPU_TERRAIN_D7D8_COLOR_COMPONENTS; ++component) {
         nir_def *product = pvrgpu_terrain_d7d8_hmul(
            &sample_builder, nir_channel(&sample_builder, sample, component),
            weight_constants[weight_index[tap]], &counts);
         ++counts.weighted_multiplies;
         if (accumulator[component]) {
            product = pvrgpu_terrain_d7d8_hadd(
               &sample_builder, accumulator[component], product, &counts);
            ++counts.accumulator_adds;
         }
         accumulator[component] = product;
      }
   }

   nir_builder output_builder =
      nir_builder_at(nir_before_instr(&graph.stores[0]->instr));
   for (unsigned component = 0;
        component < PVRGPU_TERRAIN_D7D8_COLOR_COMPONENTS; ++component) {
      nir_src_rewrite(&graph.stores[component]->src[0], accumulator[component]);
      ++counts.architectural_color_exits;
   }
   nir_src_rewrite(&graph.stores[3]->src[0],
                   nir_imm_float(&output_builder, 1.0f));

   const unsigned expected_inputs = horizontal ? 2U : 1U;
   const unsigned expected_widens = expected_inputs + 68U + 36U;
   if (counts.textures != 9U || counts.offset_coords != 8U ||
       counts.input_rtne != expected_inputs ||
       counts.texture_rtz_lanes != 36U || counts.add_rtne != 33U ||
       counts.mul_rtne != 34U || counts.div_rtne != 1U ||
       counts.exact_negates != 2U || counts.exact_absolutes != 1U ||
       counts.weighted_multiplies != 27U || counts.accumulator_adds != 24U ||
       counts.widened_values != expected_widens ||
       counts.architectural_coord_exits != 8U ||
       counts.architectural_color_exits != 3U) {
      return pvrgpu_pco_fail(
         error, error_size,
         "terrain D7/D8 lowered count changed "
         "(tex=%u coord=%u in=%u rtz=%u add=%u mul=%u div=%u "
         "neg=%u abs=%u weighted=%u accum=%u widen=%u)",
         counts.textures, counts.offset_coords, counts.input_rtne,
         counts.texture_rtz_lanes, counts.add_rtne, counts.mul_rtne,
         counts.div_rtne, counts.exact_negates, counts.exact_absolutes,
         counts.weighted_multiplies, counts.accumulator_adds,
         counts.widened_values);
   }

   nir_opt_dce(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_progress(true, graph.entrypoint, nir_metadata_control_flow);
   return true;
}
bool pvrgpu_pco_compile_terrain(
   struct pvrgpu_pco_compiler *compiler,
   const nir_shader *vertex_nir,
   const nir_shader *fragment_nir,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));
   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir ||
       profile < PVRGPU_PCO_TERRAIN_D1 ||
       profile > PVRGPU_PCO_TERRAIN_D8) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "missing compiler or invalid terrain profile");
   }

   const struct pvrgpu_terrain_pco_desc *desc =
      &pvrgpu_terrain_profiles[profile];
   const bool fragment_source_matches =
      pvrgpu_source_hash_matches(fragment_nir,
                                 desc->fragment_source_hash) ||
      (desc->fragment_source_hash_800x600[0] != 0U &&
       pvrgpu_source_hash_matches(fragment_nir,
                                  desc->fragment_source_hash_800x600));
   if (!pvrgpu_source_hash_matches(vertex_nir, desc->vertex_source_hash) ||
       !fragment_source_matches ||
       !pvrgpu_validate_terrain_nir(vertex_nir,
                                    profile,
                                    MESA_SHADER_VERTEX,
                                    error,
                                    error_size) ||
       !pvrgpu_validate_terrain_nir(fragment_nir,
                                    profile,
                                    MESA_SHADER_FRAGMENT,
                                    error,
                                    error_size)) {
      if (error && error_size && error[0] == '\0') {
         snprintf(error,
                  error_size,
                  "%s NIR source signature mismatch",
                  desc->name);
      }
      return false;
   }

   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");
   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "failed to clone %s NIR",
                             desc->name);
   }

   nir_shader_compiler_options terrain_options = *pco_nir_options();
   terrain_options.float_mul_add32 &= ~nir_float_muladd_support_fuse;
   terrain_options.float_mul_add32 |=
      nir_float_muladd_support_prefers_split;
   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = &terrain_options;
   fs->options = &terrain_options;
   nir_lower_fragcolor(fs, 1);
   if (!pvrgpu_validate_lit_mesh_fragment_output(fs,
                                                 desc->name,
                                                 error,
                                                 error_size) ||
       !pvrgpu_canonicalize_fragment_output(fs,
                                            desc->name,
                                            error,
                                            error_size) ||
       !pvrgpu_lower_uniform_slots_to_push_constants(
          vs,
          desc->vertex_uniform_dwords,
          desc->vertex_uniform_loads,
          desc->name,
          error,
          error_size) ||
       (desc->fragment_uniform_loads != 0 &&
        !pvrgpu_lower_uniform_slots_to_push_constants(
           fs,
           desc->fragment_uniform_dwords,
           desc->fragment_uniform_loads,
           desc->name,
           error,
           error_size)) ||
       (desc->vertex_texture_count != 0 &&
        !pvrgpu_pack_terrain_texture_bindings(
           vs,
           desc->vertex_texture_count,
           desc->vertex_texture_count,
           desc->name,
           error,
           error_size)) ||
       (desc->fragment_texture_count != 0 &&
        !pvrgpu_pack_terrain_texture_bindings(
           fs,
           desc->fragment_texture_count,
           desc->fragment_texture_ops,
           desc->name,
           error,
           error_size))) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   if ((profile == PVRGPU_PCO_TERRAIN_D1 &&
        !pvrgpu_lower_terrain_d1_fragment_mediump(fs,
                                                   error,
                                                   error_size)) ||
       (profile == PVRGPU_PCO_TERRAIN_D2 &&
        !pvrgpu_lower_terrain_d2_fragment_mediump(fs,
                                                   error,
                                                   error_size)) ||
       (profile == PVRGPU_PCO_TERRAIN_D3 &&
        !pvrgpu_lower_terrain_d3_vertex_mediump(vs,
                                                 error,
                                                 error_size)) ||
       (profile == PVRGPU_PCO_TERRAIN_D3 &&
        !pvrgpu_lower_terrain_d3_fragment_mediump(fs,
                                                   error,
                                                   error_size)) ||
       ((profile == PVRGPU_PCO_TERRAIN_D4 ||
         profile == PVRGPU_PCO_TERRAIN_D5 ||
         profile == PVRGPU_PCO_TERRAIN_D6) &&
        !pvrgpu_lower_terrain_d4_d6_fragment_mediump(fs,
                                                      profile,
                                                      error,
                                                      error_size)) ||
       ((profile == PVRGPU_PCO_TERRAIN_D7 ||
         profile == PVRGPU_PCO_TERRAIN_D8) &&
        !pvrgpu_lower_terrain_d7d8_fragment_mediump(fs,
                                                     profile,
                                                     error,
                                                     error_size))) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   if (!pvrgpu_init_terrain_shader_data(profile,
                                        &vertex_data,
                                        &fragment_data,
                                        compile_mem_ctx,
                                        error,
                                        error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }
   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);
   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   const unsigned vertex_descriptor_dwords =
      desc->vertex_texture_count * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   const unsigned fragment_descriptor_dwords =
      desc->fragment_texture_count * PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   if (!pvrgpu_allocate_prefixed_push_constants(
          &vertex_data,
          vertex_descriptor_dwords,
          desc->vertex_uniform_dwords,
          desc->vertex_uniform_used_dwords,
          desc->name,
          "VS",
          error,
          error_size) ||
       !pvrgpu_allocate_prefixed_push_constants(
          &fragment_data,
          fragment_descriptor_dwords,
          desc->fragment_uniform_dwords,
          desc->fragment_uniform_used_dwords,
          desc->name,
          "FS",
          error,
          error_size)) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate %s NIR",
                             desc->name);
   }
   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);
   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment,
                              false,
                              &out->fragment,
                              error,
                              error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   out->position_output_start = 0;
   out->position_output_count = 4;
   out->fragment_position_start = 0;
   out->fragment_position_count = 4;
   out->varying_output_start = 4;
   out->varying_output_count = desc->varying_components;
   out->fragment_varying_start = 4;
   out->fragment_varying_count = desc->varying_components * 4U;
   if (fragment_descriptor_dwords != 0) {
      out->fragment_texture_descriptor_start = 0;
      out->fragment_texture_descriptor_count = fragment_descriptor_dwords;
      out->fragment_texture_descriptor_stride =
         PVRGPU_TEXTURE_DESCRIPTOR_DWORDS;
   }

   const unsigned expected_vertex_inputs =
      profile == PVRGPU_PCO_TERRAIN_D3 ? 16U : 4U;
   if (out->vertex.abi.vertex_inputs != expected_vertex_inputs ||
       out->vertex.abi.vertex_outputs != 4U + desc->varying_components ||
       out->vertex.abi.coefficients != 0 ||
       out->vertex.abi.shareds !=
          vertex_descriptor_dwords + desc->vertex_uniform_dwords ||
       out->vertex.abi.push_constant_start != vertex_descriptor_dwords ||
       out->vertex.abi.push_constant_count != desc->vertex_uniform_dwords ||
       out->fragment.abi.coefficients !=
          4U + desc->varying_components * 4U ||
       out->fragment.abi.shareds !=
          fragment_descriptor_dwords + desc->fragment_uniform_dwords ||
       out->fragment.abi.push_constant_start != fragment_descriptor_dwords ||
       out->fragment.abi.push_constant_count !=
          desc->fragment_uniform_dwords ||
       out->position_output_start != 0 || out->position_output_count != 4 ||
       out->varying_output_start != 4 ||
       out->varying_output_count != desc->varying_components ||
       out->fragment_position_start != 0 ||
       out->fragment_position_count != 4 ||
       out->fragment_varying_start != 4 ||
       out->fragment_varying_count != desc->varying_components * 4U ||
       out->fragment_texture_descriptor_start != 0 ||
       out->fragment_texture_descriptor_count !=
          fragment_descriptor_dwords ||
       out->fragment_texture_descriptor_stride !=
          (fragment_descriptor_dwords ?
             PVRGPU_TEXTURE_DESCRIPTOR_DWORDS : 0U)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "compiled %s PCO ABI changed",
                             desc->name);
   }

   ralloc_free(compile_mem_ctx);
   return true;
}

static bool
pvrgpu_ideas_float_vector_variable(const nir_variable *var,
                                   nir_variable_mode mode,
                                   int location,
                                   unsigned components)
{
   return var && var->data.mode == mode &&
          var->data.location == location && var->data.location_frac == 0 &&
          glsl_type_is_vector(var->type) &&
          glsl_get_base_type(var->type) == GLSL_TYPE_FLOAT &&
          glsl_get_bit_size(var->type) == 32 &&
          glsl_get_vector_elements(var->type) == components;
}

static bool
pvrgpu_ideas_matrix_uniform(const nir_variable *var,
                            int location,
                            unsigned columns)
{
   return var && var->data.mode == nir_var_uniform &&
          var->data.location == location && glsl_type_is_matrix(var->type) &&
          glsl_get_base_type(var->type) == GLSL_TYPE_FLOAT &&
          glsl_get_bit_size(var->type) == 32 &&
          glsl_get_vector_elements(var->type) == columns &&
          glsl_get_matrix_columns(var->type) == columns;
}

static bool
pvrgpu_validate_ideas_constant_fragment(const nir_shader *nir,
                                        enum pvrgpu_pco_ideas_profile profile,
                                        char *error,
                                        size_t error_size)
{
   const uint32_t expected =
      profile == PVRGPU_PCO_IDEAS_WHITE ? UINT32_C(0x3f800000) : 0;
   unsigned constants = 0;
   unsigned stores = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_load_const) {
               const nir_load_const_instr *constant =
                  nir_instr_as_load_const(instr);
               if (constants++ || constant->def.bit_size != 32 ||
                   constant->def.num_components != 4 ||
                   constant->value[0].u32 != expected ||
                   constant->value[1].u32 != expected ||
                   constant->value[2].u32 != expected ||
                   constant->value[3].u32 != UINT32_C(0x3f800000)) {
                  return pvrgpu_pco_fail(error,
                                         error_size,
                                         "ideas constant fragment changed");
               }
            } else if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_instr *intr =
                  nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_deref)
                  stores++;
            }
         }
      }
   }
   if (constants != 1 || stores != 1) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "ideas constant fragment graph changed");
   }
   return true;
}

static bool
pvrgpu_validate_ideas_variables(const nir_shader *nir,
                                enum pvrgpu_pco_ideas_profile profile,
                                char *error,
                                size_t error_size)
{
   unsigned inputs = 0;
   unsigned outputs = 0;
   unsigned uniforms = 0;
   unsigned input_mask = 0;
   unsigned output_mask = 0;
   unsigned uniform_mask = 0;
   const bool lighting = profile == PVRGPU_PCO_IDEAS_LIGHTING;

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in) {
      ++inputs;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         if (var->data.location < VERT_ATTRIB_GENERIC0 ||
             var->data.location >
                (lighting ? VERT_ATTRIB_GENERIC1 : VERT_ATTRIB_GENERIC0) ||
             !pvrgpu_ideas_float_vector_variable(var,
                                                  nir_var_shader_in,
                                                  var->data.location,
                                                  4)) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas VS input ABI changed");
         }
         const unsigned bit = var->data.location - VERT_ATTRIB_GENERIC0;
         if (input_mask & BITFIELD_BIT(bit))
            return pvrgpu_pco_fail(error, error_size, "ideas duplicate input");
         input_mask |= BITFIELD_BIT(bit);
      } else {
         unsigned components = 0;
         unsigned bit = 0;
         switch (var->data.location) {
         case VARYING_SLOT_VAR0:
            components = 4;
            bit = 0;
            break;
         case VARYING_SLOT_VAR1:
            components = 4;
            bit = 1;
            break;
         case VARYING_SLOT_VAR2:
            components = 2;
            bit = 2;
            break;
         default:
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas FS varying location changed");
         }
         if (!lighting || (input_mask & BITFIELD_BIT(bit)) ||
             !pvrgpu_ideas_float_vector_variable(var,
                                                  nir_var_shader_in,
                                                  var->data.location,
                                                  components) ||
             var->data.interpolation != INTERP_MODE_SMOOTH) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas FS varying ABI changed");
         }
         input_mask |= BITFIELD_BIT(bit);
      }
   }

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
      ++outputs;
      if (nir->info.stage == MESA_SHADER_FRAGMENT) {
         if (!pvrgpu_ideas_float_vector_variable(var,
                                                  nir_var_shader_out,
                                                  FRAG_RESULT_COLOR,
                                                  4) &&
             !pvrgpu_ideas_float_vector_variable(var,
                                                  nir_var_shader_out,
                                                  FRAG_RESULT_DATA0,
                                                  4)) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas FS output ABI changed");
         }
         output_mask |= 1;
         continue;
      }
      unsigned components = 0;
      unsigned bit = 0;
      switch (var->data.location) {
      case VARYING_SLOT_POS:
         components = 4;
         bit = 0;
         break;
      case VARYING_SLOT_VAR0:
         components = 4;
         bit = 1;
         break;
      case VARYING_SLOT_VAR1:
         components = 4;
         bit = 2;
         break;
      case VARYING_SLOT_VAR2:
         components = 2;
         bit = 3;
         break;
      default:
         return pvrgpu_pco_fail(error,
                                error_size,
                                "ideas VS output location changed");
      }
      if ((!lighting && bit != 0) || (output_mask & BITFIELD_BIT(bit)) ||
          !pvrgpu_ideas_float_vector_variable(var,
                                               nir_var_shader_out,
                                               var->data.location,
                                               components)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "ideas VS output ABI changed");
      }
      output_mask |= BITFIELD_BIT(bit);
   }

   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      ++uniforms;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         unsigned bit = UINT_MAX;
         if (pvrgpu_ideas_matrix_uniform(var, 0, 4))
            bit = 0;
         else if (pvrgpu_ideas_matrix_uniform(var, 1, 4))
            bit = 1;
         else if (lighting && pvrgpu_ideas_matrix_uniform(var, 2, 3))
            bit = 2;
         if (bit == UINT_MAX || (uniform_mask & BITFIELD_BIT(bit)))
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas VS uniform ABI changed");
         uniform_mask |= BITFIELD_BIT(bit);
      } else if (profile == PVRGPU_PCO_IDEAS_LOGO) {
         if (uniforms != 1 ||
             !pvrgpu_ideas_float_vector_variable(var,
                                                  nir_var_uniform,
                                                  2,
                                                  4)) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas logo FS uniform ABI changed");
         }
      } else if (lighting) {
         if (var->data.location < 3 || var->data.location > 5 ||
             !pvrgpu_ideas_float_vector_variable(var,
                                                  nir_var_uniform,
                                                  var->data.location,
                                                  4)) {
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas lighting FS uniform ABI changed");
         }
         const unsigned bit = var->data.location - 3;
         if (uniform_mask & BITFIELD_BIT(bit))
            return pvrgpu_pco_fail(error,
                                   error_size,
                                   "ideas lighting FS duplicate uniform");
         uniform_mask |= BITFIELD_BIT(bit);
      } else {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "ideas constant FS unexpectedly has uniforms");
      }
   }

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      const unsigned expected_inputs = lighting ? 2 : 1;
      const unsigned expected_outputs = lighting ? 4 : 1;
      const unsigned expected_uniforms = lighting ? 3 : 2;
      if (inputs != expected_inputs ||
          input_mask != (lighting ? 0x3u : 0x1u) ||
          outputs != expected_outputs ||
          output_mask != (lighting ? 0xfu : 0x1u) ||
          uniforms != expected_uniforms ||
          uniform_mask != (lighting ? 0x7u : 0x3u)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "ideas VS variable signature changed");
      }
   } else {
      const unsigned expected_inputs = lighting ? 3 : 0;
      const unsigned expected_uniforms =
         lighting ? 3 : (profile == PVRGPU_PCO_IDEAS_LOGO ? 1 : 0);
      const unsigned expected_uniform_mask = lighting ? 0x7u : 0;
      if (inputs != expected_inputs ||
          input_mask != (lighting ? 0x7u : 0) || outputs != 1 ||
          output_mask != 1 || uniforms != expected_uniforms ||
          (lighting && uniform_mask != expected_uniform_mask)) {
         return pvrgpu_pco_fail(error,
                                error_size,
                                "ideas FS variable signature changed");
      }
   }
   return true;
}

static bool
pvrgpu_validate_ideas_nir(const nir_shader *nir,
                          mesa_shader_stage stage,
                          enum pvrgpu_pco_ideas_profile profile,
                          char *error,
                          size_t error_size)
{
   if (!nir || nir->info.stage != stage ||
       profile < PVRGPU_PCO_IDEAS_LOGO ||
       profile > PVRGPU_PCO_IDEAS_BLACK) {
      return pvrgpu_pco_fail(error,
                             error_size,
                             "ideas compiler arguments are invalid");
   }
   const struct pvrgpu_ideas_pco_desc *desc =
      &pvrgpu_ideas_profiles[profile];
   const uint32_t *hash = stage == MESA_SHADER_VERTEX ?
                             desc->vertex_source_hash :
                             desc->fragment_source_hash;
   if (!pvrgpu_source_hash_matches(nir, hash) ||
       !pvrgpu_validate_ideas_variables(nir, profile, error, error_size)) {
      if (error && error_size && error[0] == '\0')
         snprintf(error, error_size, "%s NIR signature mismatch", desc->name);
      return false;
   }
   if (stage == MESA_SHADER_FRAGMENT &&
       (profile == PVRGPU_PCO_IDEAS_WHITE ||
        profile == PVRGPU_PCO_IDEAS_BLACK) &&
       !pvrgpu_validate_ideas_constant_fragment(nir,
                                                profile,
                                                error,
                                                error_size))
      return false;
   return true;
}

static void
pvrgpu_init_ideas_shader_data(enum pvrgpu_pco_ideas_profile profile,
                              pco_data *vertex_data,
                              pco_data *fragment_data)
{
   vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC0] =
      PIPE_FORMAT_R32G32B32A32_FLOAT;
   vertex_data->vs.attribs[VERT_ATTRIB_GENERIC0] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->common.vtxins = 4;
   if (profile == PVRGPU_PCO_IDEAS_LIGHTING) {
      vertex_data->vs.attrib_formats[VERT_ATTRIB_GENERIC1] =
         PIPE_FORMAT_R32G32B32A32_FLOAT;
      vertex_data->vs.attribs[VERT_ATTRIB_GENERIC1] = (pco_range){
         .start = 4,
         .count = 4,
      };
      vertex_data->common.vtxins = 8;
   }
   vertex_data->vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   vertex_data->vs.vtxouts = 4;

   if (profile == PVRGPU_PCO_IDEAS_LIGHTING) {
      vertex_data->vs.varyings[VARYING_SLOT_VAR0] = (pco_range){4, 4};
      vertex_data->vs.varyings[VARYING_SLOT_VAR1] = (pco_range){8, 4};
      vertex_data->vs.varyings[VARYING_SLOT_VAR2] = (pco_range){12, 2};
      vertex_data->vs.vtxouts = 14;
      vertex_data->vs.f32_smooth = 10;

      fragment_data->fs.uses.w = true;
      fragment_data->fs.varyings[VARYING_SLOT_POS] = (pco_range){0, 4};
      fragment_data->fs.varyings[VARYING_SLOT_VAR0] = (pco_range){4, 16};
      fragment_data->fs.varyings[VARYING_SLOT_VAR1] = (pco_range){20, 16};
      fragment_data->fs.varyings[VARYING_SLOT_VAR2] = (pco_range){36, 8};
      fragment_data->common.coeffs = 44;
   }
   fragment_data->fs.z_replicate = ~0U;
   fragment_data->fs.rasterization_samples = 1;
}

bool pvrgpu_pco_compile_ideas(
   struct pvrgpu_pco_compiler *compiler,
   const nir_shader *vertex_nir,
   const nir_shader *fragment_nir,
   enum pvrgpu_pco_ideas_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size)
{
   if (error && error_size)
      error[0] = '\0';
   if (!out)
      return pvrgpu_pco_fail(error, error_size, "missing PCO output object");
   memset(out, 0, sizeof(*out));
   if (!compiler || !compiler->pco || !vertex_nir || !fragment_nir ||
       profile < PVRGPU_PCO_IDEAS_LOGO ||
       profile > PVRGPU_PCO_IDEAS_BLACK ||
       !pvrgpu_validate_ideas_nir(vertex_nir,
                                  MESA_SHADER_VERTEX,
                                  profile,
                                  error,
                                  error_size) ||
       !pvrgpu_validate_ideas_nir(fragment_nir,
                                  MESA_SHADER_FRAGMENT,
                                  profile,
                                  error,
                                  error_size)) {
      if (error && error_size && error[0] == '\0')
         snprintf(error, error_size, "missing compiler or ideas NIR stage");
      return false;
   }

   const struct pvrgpu_ideas_pco_desc *desc =
      &pvrgpu_ideas_profiles[profile];
   void *compile_mem_ctx = ralloc_context(compiler->mem_ctx);
   if (!compile_mem_ctx)
      return pvrgpu_pco_fail(error, error_size, "out of memory cloning NIR");
   nir_shader *vs = nir_shader_clone(compile_mem_ctx, vertex_nir);
   nir_shader *fs = nir_shader_clone(compile_mem_ctx, fragment_nir);
   if (!vs || !fs) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "failed to clone %s NIR",
                             desc->name);
   }
   vs->info.internal = true;
   fs->info.internal = true;
   vs->options = pco_nir_options();
   fs->options = pco_nir_options();
   nir_lower_fragcolor(fs, 1);

   const unsigned vs_loads =
      profile == PVRGPU_PCO_IDEAS_LIGHTING ? 11 : 8;
   const unsigned fs_loads =
      profile == PVRGPU_PCO_IDEAS_LIGHTING ? 3 :
      (profile == PVRGPU_PCO_IDEAS_LOGO ? 1 : 0);
   if (!pvrgpu_validate_lit_mesh_fragment_output(fs,
                                                 desc->name,
                                                 error,
                                                 error_size) ||
       !pvrgpu_canonicalize_fragment_output(fs,
                                            desc->name,
                                            error,
                                            error_size) ||
       !pvrgpu_lower_uniform_slots_to_push_constants(
          vs,
          desc->vertex_shared_dwords,
          vs_loads,
          desc->name,
          error,
          error_size) ||
       (fs_loads != 0 &&
        !pvrgpu_lower_uniform_slots_to_push_constants(
           fs,
           desc->fragment_shared_dwords,
           fs_loads,
           desc->name,
           error,
           error_size))) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   pvrgpu_init_ideas_shader_data(profile, &vertex_data, &fragment_data);
   pco_preprocess_nir(compiler->pco, vs);
   pco_preprocess_nir(compiler->pco, fs);
   pco_link_nir(compiler->pco, vs, fs, &vertex_data, &fragment_data);
   pco_rev_link_nir(compiler->pco, vs, fs);
   pco_lower_nir(compiler->pco, vs, &vertex_data);
   pco_lower_nir(compiler->pco, fs, &fragment_data);
   pco_postprocess_nir(compiler->pco, vs, &vertex_data);
   pco_postprocess_nir(compiler->pco, fs, &fragment_data);

   const unsigned vs_used =
      profile == PVRGPU_PCO_IDEAS_LIGHTING ? 43 : 32;
   if (!pvrgpu_allocate_push_constants(&vertex_data,
                                       desc->vertex_shared_dwords,
                                       vs_used,
                                       desc->name,
                                       "VS",
                                       error,
                                       error_size) ||
       (desc->fragment_shared_dwords != 0 &&
        !pvrgpu_allocate_push_constants(&fragment_data,
                                        desc->fragment_shared_dwords,
                                        desc->fragment_shared_dwords,
                                        desc->name,
                                        "FS",
                                        error,
                                        error_size))) {
      ralloc_free(compile_mem_ctx);
      return false;
   }

   pco_shader *vertex =
      pco_trans_nir(compiler->pco, vs, &vertex_data, compile_mem_ctx);
   pco_shader *fragment =
      pco_trans_nir(compiler->pco, fs, &fragment_data, compile_mem_ctx);
   if (!vertex || !fragment) {
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "PCO failed to translate %s NIR",
                             desc->name);
   }
   pco_process_ir(compiler->pco, vertex);
   pco_process_ir(compiler->pco, fragment);
   pco_encode_ir(compiler->pco, vertex);
   pco_encode_ir(compiler->pco, fragment);
   if (!pvrgpu_copy_pco_stage(vertex, true, &out->vertex, error, error_size) ||
       !pvrgpu_copy_pco_stage(fragment,
                              false,
                              &out->fragment,
                              error,
                              error_size)) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return false;
   }

   out->position_output_start =
      vertex_data.vs.varyings[VARYING_SLOT_POS].start;
   out->position_output_count =
      vertex_data.vs.varyings[VARYING_SLOT_POS].count;
   if (profile == PVRGPU_PCO_IDEAS_LIGHTING) {
      out->fragment_position_start =
         fragment_data.fs.varyings[VARYING_SLOT_POS].start;
      out->fragment_position_count =
         fragment_data.fs.varyings[VARYING_SLOT_POS].count;
      out->varying_output_start =
         vertex_data.vs.varyings[VARYING_SLOT_VAR0].start;
      out->varying_output_count = 10;
      out->fragment_varying_start =
         fragment_data.fs.varyings[VARYING_SLOT_VAR0].start;
      out->fragment_varying_count = 40;
   }

   const unsigned expected_vtxins = desc->attribute_count * 4;
   const unsigned expected_vtxouts = 4 + desc->varying_components;
   const unsigned expected_coeffs = desc->varying_components ? 44 : 0;
   if (out->vertex.abi.vertex_inputs != expected_vtxins ||
       out->vertex.abi.vertex_outputs != expected_vtxouts ||
       out->vertex.abi.coefficients != 0 ||
       out->vertex.abi.shareds != desc->vertex_shared_dwords ||
       out->vertex.abi.push_constant_start != 0 ||
       out->vertex.abi.push_constant_count != desc->vertex_shared_dwords ||
       out->fragment.abi.coefficients != expected_coeffs ||
       out->fragment.abi.shareds != desc->fragment_shared_dwords ||
       out->fragment.abi.push_constant_start != 0 ||
       out->fragment.abi.push_constant_count !=
          desc->fragment_shared_dwords ||
       out->position_output_start != 0 || out->position_output_count != 4 ||
       (desc->varying_components == 0 &&
        (out->fragment_position_count != 0 ||
         out->varying_output_count != 0 ||
         out->fragment_varying_count != 0)) ||
       (desc->varying_components != 0 &&
        (out->fragment_position_start != 0 ||
         out->fragment_position_count != 4 ||
         out->varying_output_start != 4 ||
         out->varying_output_count != 10 ||
         out->fragment_varying_start != 4 ||
         out->fragment_varying_count != 40))) {
      pvrgpu_pco_graphics_binary_finish(out);
      ralloc_free(compile_mem_ctx);
      return pvrgpu_pco_fail(error,
                             error_size,
                             "compiled %s PCO ABI changed",
                             desc->name);
   }

   ralloc_free(compile_mem_ctx);
   return true;
}
