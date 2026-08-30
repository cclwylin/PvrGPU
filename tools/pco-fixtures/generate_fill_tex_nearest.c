/*
 * Development-time generator for the exact GLBench fill_tex_nearest PCO
 * vertex and fragment fixtures.
 *
 * This utility constructs NIR matching the upstream GLBench shaders and runs
 * the pinned public Mesa PowerVR PCO backend.  It never constructs a local
 * instruction byte stream.  The vertex shader reads two R32G32_FLOAT
 * attributes, applies the public scale push-constant ABI to position.xy,
 * exports gl_Position and the linked texture-coordinate varying.  The
 * fragment shader perspective-interpolates v1.xy, executes a real combined
 * image-sampler texture operation, and exports its four returned components.
 *
 * Generated files are immutable runtime fixtures; PvrGPU does not link Mesa.
 *
 * Provenance:
 *   GLBench commit e99bc684272bffd68b06c998e272531c9c84330f
 *   Mesa 26.2.1 commit da14d65e4499e66468094be52bff9ea0915a695e
 *   public target gx6250
 */

#include "common/pvr_device_info.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "nir/nir_builder_opcodes.h"
#include "pco/pco.h"
#include "pco/pco_data.h"
#include "util/ralloc.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static nir_shader *build_vertex_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                   pco_nir_options(),
                                                   "glbench_fill_tex_nearest_vs");
   nir_variable *position_in = nir_variable_create(b.shader,
                                                    nir_var_shader_in,
                                                    glsl_vec4_type(),
                                                    "position");
   position_in->data.location = VERT_ATTRIB_GENERIC0;

   nir_variable *texcoord_in = nir_variable_create(b.shader,
                                                    nir_var_shader_in,
                                                    glsl_vec4_type(),
                                                    "texcoord");
   texcoord_in->data.location = VERT_ATTRIB_GENERIC1;

   nir_variable *position_out = nir_variable_create(b.shader,
                                                     nir_var_shader_out,
                                                     glsl_vec4_type(),
                                                     "gl_Position");
   position_out->data.location = VARYING_SLOT_POS;

   nir_variable *varying_out = nir_variable_create(b.shader,
                                                    nir_var_shader_out,
                                                    glsl_vec4_type(),
                                                    "v1");
   varying_out->data.location = VARYING_SLOT_VAR0;
   varying_out->data.interpolation = INTERP_MODE_NONE;

   /* Keep source-level variables through pco_link_nir so Mesa, rather than
    * this generator, removes the unused z/w texture-coordinate channels.
    */
   nir_def *position = nir_load_var(&b, position_in);
   nir_def *texcoord = nir_load_var(&b, texcoord_in);

   /* GLBench declares a scalar uniform.  The pinned PCO Vulkan-facing ABI
    * makes that scalar one push-constant dword; the generated instruction
    * remains a real shared-register FMUL input rather than specializing 1.0.
    */
   nir_def *scale = nir_load_push_constant(&b,
                                           1,
                                           32,
                                           nir_imm_int(&b, 0),
                                           .base = 0,
                                           .range = sizeof(float));
   nir_def *scaled_position = nir_vec4(&b,
                                       nir_fmul(&b,
                                                nir_channel(&b, position, 0),
                                                scale),
                                       nir_fmul(&b,
                                                nir_channel(&b, position, 1),
                                                scale),
                                       nir_channel(&b, position, 2),
                                       nir_channel(&b, position, 3));

   nir_store_var(&b, position_out, scaled_position, 0xf);
   nir_store_var(&b, varying_out, texcoord, 0xf);
   nir_jump(&b, nir_jump_return);
   return b.shader;
}

static nir_shader *build_fragment_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                   pco_nir_options(),
                                                   "glbench_fill_tex_nearest_fs");
   nir_variable *varying = nir_variable_create(b.shader,
                                                nir_var_shader_in,
                                                glsl_vec4_type(),
                                                "v1");
   varying->data.location = VARYING_SLOT_VAR0;
   varying->data.interpolation = INTERP_MODE_SMOOTH;

   const struct glsl_type *sampler_type =
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D,
                        false,
                        false,
                        GLSL_TYPE_FLOAT);
   nir_variable *sampler = nir_variable_create(b.shader,
                                                nir_var_uniform,
                                                sampler_type,
                                                "texture");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 0;

   nir_def *texcoord = nir_channels(&b, nir_load_var(&b, varying), 0x3);
   nir_deref_instr *sampler_deref = nir_build_deref_var(&b, sampler);
   nir_def *color = nir_tex(&b,
                            texcoord,
                            .texture_deref = sampler_deref,
                            .sampler_deref = sampler_deref);
   for (unsigned component = 0; component < 4; ++component) {
      nir_frag_store_pco(&b,
                         nir_channel(&b, color, component),
                         .base = component);
   }
   nir_jump(&b, nir_jump_return);
   return b.shader;
}

static unsigned location_dwords(nir_shader *nir,
                                nir_variable_mode mode,
                                unsigned location)
{
   unsigned dwords = 0;
   nir_foreach_variable_with_modes (var, nir, mode) {
      if (var->data.location == location)
         dwords += glsl_count_dword_slots(var->type, false);
   }
   return dwords;
}

static void init_descriptor_binding(pco_data *data)
{
   pco_descriptor_set_data *set = &data->common.desc_sets[0];
   set->binding_count = 1;
   set->bindings = rzalloc_array(NULL, pco_binding_data, 1);
   set->bindings[0].is_img_smp = true;
}

static void init_vertex_attributes(pco_data *data)
{
   /* PVI (Packed Vertex Input) lowering consumes this linked driver ABI, so
    * it must be present before pco_lower_nir rather than only at translation.
    */
   data->vs.attrib_formats[VERT_ATTRIB_GENERIC0] =
      PIPE_FORMAT_R32G32_FLOAT;
   data->vs.attrib_formats[VERT_ATTRIB_GENERIC1] =
      PIPE_FORMAT_R32G32_FLOAT;
   data->vs.attribs[VERT_ATTRIB_GENERIC0] = (pco_range){
      .start = 0,
      .count = 2,
   };
   data->vs.attribs[VERT_ATTRIB_GENERIC1] = (pco_range){
      .start = 2,
      .count = 2,
   };
   data->common.vtxins = 4;
}

static bool allocate_linked_io(pco_data *vertex_data,
                               pco_data *fragment_data,
                               nir_shader *vertex_nir,
                               nir_shader *fragment_nir)
{
   const unsigned position_dwords =
      location_dwords(vertex_nir, nir_var_shader_out, VARYING_SLOT_POS);
   const unsigned vertex_varying_dwords =
      location_dwords(vertex_nir, nir_var_shader_out, VARYING_SLOT_VAR0);
   const unsigned fragment_varying_dwords =
      location_dwords(fragment_nir, nir_var_shader_in, VARYING_SLOT_VAR0);
   if (position_dwords != 4 || vertex_varying_dwords != 2 ||
       fragment_varying_dwords != 2) {
      fprintf(stderr,
              "unexpected linked ABI: position=%u VS texcoord=%u "
              "FS texcoord=%u dwords\n",
              position_dwords,
              vertex_varying_dwords,
              fragment_varying_dwords);
      return false;
   }
   vertex_data->vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = position_dwords,
   };
   vertex_data->vs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = position_dwords,
      .count = vertex_varying_dwords,
   };
   vertex_data->vs.f32_smooth = vertex_varying_dwords;
   vertex_data->vs.vtxouts = position_dwords + vertex_varying_dwords;

   /* Smooth interpolation requires the position-W coefficient set. Driver
    * shader-data gathering normally establishes this before lower_nir.
    */
   fragment_data->fs.uses.w = true;
   fragment_data->fs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = ROGUE_USC_COEFFICIENT_SET_SIZE,
   };
   fragment_data->fs.varyings[VARYING_SLOT_VAR0] = (pco_range){
      .start = ROGUE_USC_COEFFICIENT_SET_SIZE,
      .count = fragment_varying_dwords * ROGUE_USC_COEFFICIENT_SET_SIZE,
   };
   fragment_data->common.coeffs =
      (1 + fragment_varying_dwords) * ROGUE_USC_COEFFICIENT_SET_SIZE;
   return true;
}

static bool allocate_vertex_shared_data(pco_data *data)
{
   if (data->common.push_consts.used != 1) {
      fprintf(stderr,
              "unexpected VS push-constant ABI: %u dwords\n",
              data->common.push_consts.used);
      return false;
   }
   data->common.push_consts.range = (pco_range){
      .start = 0,
      .count = 1,
   };
   data->common.shareds = 1;
   return true;
}

static bool allocate_fragment_shared_data(pco_data *data)
{
   pco_descriptor_set_data *set = &data->common.desc_sets[0];
   pco_binding_data *binding = &set->bindings[0];
   const unsigned combined_descriptor_dwords =
      ROGUE_NUM_TEXSTATE_DWORDS + PCO_IMAGE_META_COUNT +
      ROGUE_NUM_TEXSTATE_DWORDS + PCO_SAMPLER_META_COUNT +
      ROGUE_NUM_TEXSTATE_DWORDS;
   if (!set->used || !binding->used || !binding->is_img_smp) {
      fprintf(stderr, "PCO did not mark combined sampler set=0 binding=0 used\n");
      return false;
   }
   set->range = (pco_range){
      .start = 0,
      .count = combined_descriptor_dwords,
   };
   binding->range = (pco_range){
      .start = 0,
      .count = combined_descriptor_dwords,
      .stride = combined_descriptor_dwords,
   };
   data->common.shareds = combined_descriptor_dwords;
   return true;
}

static pco_shader *translate_shader(pco_ctx *ctx,
                                    nir_shader *nir,
                                    pco_data *data)
{
   pco_shader *shader = pco_trans_nir(ctx, nir, data, NULL);
   if (!shader)
      return NULL;
   pco_process_ir(ctx, shader);
   pco_encode_ir(ctx, shader);
   return shader;
}

static int write_binary(const char *path, pco_shader *shader)
{
   FILE *file = fopen(path, "wb");
   if (!file) {
      fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno));
      return 1;
   }
   const unsigned size = pco_shader_binary_size(shader);
   const size_t written =
      fwrite(pco_shader_binary_data(shader), 1, size, file);
   const int close_result = fclose(file);
   if (written != size || close_result != 0) {
      fprintf(stderr, "failed to write complete PCO binary: %s\n", path);
      return 1;
   }
   fprintf(stderr, "wrote %u bytes of Mesa PCO binary to %s\n", size, path);
   return 0;
}

int main(int argc, char **argv)
{
   if (argc != 3) {
      fprintf(stderr, "Usage: %s VERTEX.bin FRAGMENT.bin\n", argv[0]);
      return 2;
   }

   void *mem_ctx = ralloc_context(NULL);
   struct pvr_device_info dev_info;
   struct pvr_device_runtime_info runtime_info = { 0 };
   if (!pvr_device_info_init_public_name(&dev_info, "gx6250")) {
      fprintf(stderr, "Mesa does not recognize the public gx6250 target\n");
      ralloc_free(mem_ctx);
      return 1;
   }
   pco_ctx *ctx = pco_ctx_create(&dev_info, &runtime_info, mem_ctx);
   nir_shader *vertex_nir = build_vertex_shader();
   nir_shader *fragment_nir = build_fragment_shader();
   pco_data vertex_data = { 0 };
   pco_data fragment_data = { 0 };
   init_vertex_attributes(&vertex_data);
   init_descriptor_binding(&fragment_data);

   pco_preprocess_nir(ctx, vertex_nir);
   pco_preprocess_nir(ctx, fragment_nir);
   pco_link_nir(ctx,
                vertex_nir,
                fragment_nir,
                &vertex_data,
                &fragment_data);
   pco_rev_link_nir(ctx, vertex_nir, fragment_nir);
   if (!allocate_linked_io(&vertex_data,
                           &fragment_data,
                           vertex_nir,
                           fragment_nir)) {
      ralloc_free(fragment_data.common.desc_sets[0].bindings);
      ralloc_free(vertex_nir);
      ralloc_free(fragment_nir);
      ralloc_free(mem_ctx);
      return 1;
   }
   pco_lower_nir(ctx, vertex_nir, &vertex_data);
   pco_lower_nir(ctx, fragment_nir, &fragment_data);
   pco_postprocess_nir(ctx, vertex_nir, &vertex_data);
   pco_postprocess_nir(ctx, fragment_nir, &fragment_data);
   if (!allocate_vertex_shared_data(&vertex_data) ||
       !allocate_fragment_shared_data(&fragment_data)) {
      ralloc_free(fragment_data.common.desc_sets[0].bindings);
      ralloc_free(vertex_nir);
      ralloc_free(fragment_nir);
      ralloc_free(mem_ctx);
      return 1;
   }

   pco_shader *vertex = translate_shader(ctx, vertex_nir, &vertex_data);
   pco_shader *fragment = translate_shader(ctx, fragment_nir, &fragment_data);
   if (!vertex || !fragment) {
      fprintf(stderr, "PCO fill_tex_nearest translation failed\n");
      ralloc_free(vertex);
      ralloc_free(fragment);
      ralloc_free(fragment_data.common.desc_sets[0].bindings);
      ralloc_free(vertex_nir);
      ralloc_free(fragment_nir);
      ralloc_free(mem_ctx);
      return 1;
   }

   pco_print_shader(vertex, stdout, "fill_tex_nearest vertex");
   pco_print_binary(vertex, stdout, "fill_tex_nearest vertex");
   pco_print_shader(fragment, stdout, "fill_tex_nearest fragment");
   pco_print_binary(fragment, stdout, "fill_tex_nearest fragment");
   const int result = write_binary(argv[1], vertex) ||
                      write_binary(argv[2], fragment);
   ralloc_free(vertex);
   ralloc_free(fragment);
   ralloc_free(fragment_data.common.desc_sets[0].bindings);
   ralloc_free(vertex_nir);
   ralloc_free(fragment_nir);
   ralloc_free(mem_ctx);
   return result;
}
