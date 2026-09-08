/* SPDX-License-Identifier: MIT */
/* Reuse the binary/ABI serializer, not any execution or expected-answer code. */
int compute_buffer_compiler_main(int argc, char **argv);
#define main compute_buffer_compiler_main
#include "pvrgpu_compute_compiler_test.c"
#undef main

static nir_shader *image_shader(unsigned kind, bool dereference_images)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      pco_nir_options(), "native_image2d_r32ui");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = kind == 35 ? 33 : 4;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *global = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, global, 0);
   nir_def *y = nir_channel(&b, global, 1);
   nir_def *coords = nir_vec4(&b, x, y, nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_def *slot = nir_imm_int(&b, 1);
   nir_def *sample = nir_undef(&b, 1, 32);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *size = nir_image_size(&b, 2, 32, slot, zero,
      .image_dim = GLSL_SAMPLER_DIM_2D, .format = PIPE_FORMAT_R32_UINT);
   nir_def *linear = nir_iadd(&b, x, nir_imul(&b, y, nir_channel(&b, size, 0)));
   nir_def *offset = nir_imul_imm(&b, linear, 4);
   if (kind == 32 || kind == 36) {
      if (kind == 36)
         coords = nir_vec4(&b, nir_iadd_imm(&b, x, -1), y, zero, zero);
      nir_def *value = nir_image_load(&b, 4, 32, slot, coords, sample, zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .format = PIPE_FORMAT_R32_UINT,
         .dest_type = nir_type_uint32);
      nir_store_ssbo(&b, value, zero, nir_imul_imm(&b, linear, 16),
                     .write_mask = 15, .align_mul = 4);
   } else {
      nir_def *value = kind == 37 ?
         nir_iadd(&b, linear, nir_load_uniform(&b, 1, 32, zero, .range = 1,
                                              .dest_type = nir_type_uint32)) :
         nir_load_ssbo(&b, 1, 32, zero, offset, .align_mul = 4);
      if (kind == 34) {
         nir_def *old = nir_image_atomic(&b, 32, slot, coords, sample, value,
            .image_dim = GLSL_SAMPLER_DIM_2D, .format = PIPE_FORMAT_R32_UINT,
            .atomic_op = nir_atomic_op_iadd);
         nir_store_ssbo(&b, old, zero, offset, .write_mask = 1, .align_mul = 4);
      } else {
         nir_image_store(&b, slot, coords, sample, nir_vec4(&b, value, zero, zero, zero), zero,
            .image_dim = GLSL_SAMPLER_DIM_2D, .format = PIPE_FORMAT_R32_UINT,
            .src_type = nir_type_uint32);
         if (kind == 35 || kind == 38) {
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP, .memory_modes = nir_var_image,
               .memory_semantics = NIR_MEMORY_ACQ_REL);
            nir_def *read = nir_image_load(&b, 4, 32,
               kind == 38 ? nir_imm_int(&b, 3) : slot, coords, sample, zero,
               .image_dim = GLSL_SAMPLER_DIM_2D, .format = PIPE_FORMAT_R32_UINT,
               .dest_type = nir_type_uint32);
            nir_store_ssbo(&b, nir_channel(&b, read, 0), zero, offset,
                           .write_mask = 1, .align_mul = 4);
         }
      }
   }
   if (dereference_images) {
      nir_variable *images = nir_variable_create(b.shader, nir_var_image,
         glsl_array_type(glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT), 4, 0),
         "gallium_images");
      images->data.driver_location = 0;
      images->data.binding = 19; /* Deliberately different from Gallium slots. */
      images->data.image.format = PIPE_FORMAT_R32_UINT;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_image_load: intr->intrinsic = nir_intrinsic_image_deref_load; break;
            case nir_intrinsic_image_store: intr->intrinsic = nir_intrinsic_image_deref_store; break;
            case nir_intrinsic_image_atomic: intr->intrinsic = nir_intrinsic_image_deref_atomic; break;
            case nir_intrinsic_image_size: intr->intrinsic = nir_intrinsic_image_deref_size; break;
            default: continue;
            }
            b.cursor = nir_before_instr(instr);
            nir_deref_instr *element = nir_build_deref_array(&b,
               nir_build_deref_var(&b, images), intr->src[0].ssa);
            nir_src_rewrite(&intr->src[0], &element->def);
            nir_intrinsic_set_format(intr, PIPE_FORMAT_NONE);
         }
      }
   }
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_images = kind == 38 ? 4 : 2;
   b.shader->info.num_ssbos = kind == 37 ? 0 : 1;
   return b.shader;
}

int main(int argc, char **argv)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   check(compiler != NULL, error);
   for (unsigned deref = 0; deref < 2; ++deref)
   for (unsigned kind = 32; kind <= 38; ++kind) {
      nir_shader *nir = image_shader(kind, deref != 0);
      struct pvrgpu_pco_compute_binary native = {0};
      check(pvrgpu_pco_compile_compute(compiler, nir, kind == 37 ? 4 : 0,
                                       &native, error, sizeof(error)), error);
      const struct pvrgpu_pco_compute_abi *a = &native.abi;
      check(a->image_descriptor_count == (kind == 38 ? 4 : 2) &&
            a->image_used_mask == (kind == 38 ? 10 : 2),
            "image slot namespace preserved independently of SSBOs");
      check(a->image_descriptor_start == (kind == 37 ? 0 : 4) &&
            a->stage.push_constant_start == (kind == 37 ? 16 : kind == 35 ? 24 : kind == 38 ? 36 : 20),
            "image/shared/CB0 descriptor spans");
      save_binary(argc > 1 ? argv[1] : NULL, kind, &native);
      if (argc > 1) {
         char path[4096];
         snprintf(path, sizeof(path), "%s/compute-%u.abi", argv[1], kind);
         FILE *file = fopen(path, "a");
         check(file != NULL, "image ABI file");
         fprintf(file, "images=%u,%u image_used=0x%x image_read=0x%x image_write=0x%x\n",
            a->image_descriptor_start, a->image_descriptor_count,
            a->image_used_mask, a->image_read_mask, a->image_write_mask);
         check(fclose(file) == 0, "image ABI close");
      }
      pvrgpu_pco_compute_binary_finish(&native);
      ralloc_free(nir);
   }
   for (unsigned bad = 0; bad < 9; ++bad) {
      nir_shader *nir = image_shader(32, bad < 4);
      if (bad < 4) {
         nir_foreach_variable_with_modes(var, nir, nir_var_image) {
            if (bad == 0) var->data.bindless = true;
            if (bad == 1) var->data.driver_location = 32;
            if (bad == 2) var->data.image.format = PIPE_FORMAT_R32_FLOAT;
            if (bad == 3) var->data.driver_location = 31; // array element 1 exceeds slot 31
         }
      } else {
         nir_foreach_function_impl(impl, nir) {
            nir_builder b = nir_builder_create(impl);
            nir_foreach_block(block, impl) {
               nir_foreach_instr_safe(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic) continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic != nir_intrinsic_image_load) continue;
                  b.cursor = nir_before_instr(instr);
                  if (bad == 4) nir_intrinsic_set_range_base(intr, 1);
                  if (bad == 5) nir_intrinsic_set_image_array(intr, true);
                  if (bad == 6) nir_src_rewrite(&intr->src[3], nir_imm_int(&b, 1));
                  if (bad == 7) nir_src_rewrite(&intr->src[0], nir_load_local_invocation_index(&b));
                  if (bad == 8) nir_intrinsic_set_format(intr, PIPE_FORMAT_NONE);
               }
            }
         }
      }
      struct pvrgpu_pco_compute_binary native = {0};
      error[0] = 0;
      check(!pvrgpu_pco_compile_compute(compiler, nir, 0, &native, error, sizeof(error)) &&
            error[0] && !native.data && !native.size,
            "unsupported image binding/format/LOD must fail closed");
      ralloc_free(nir);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   puts("native image compiler: PASS 14 indexed/dereferenced image2D R32UI programs and 9 rejected inputs");
   return 0;
}
