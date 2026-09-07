/* SPDX-License-Identifier: MIT */
/* Compile actual NIR through the standalone compute compiler. Expected ABI
 * metadata is checked here; shader answers are never evaluated on the CPU. */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(bool ok, const char *message)
{
   if (!ok) { fprintf(stderr, "%s\n", message); abort(); }
}

static nir_shader *make_shader(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   pco_nir_options(), "native_compute");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = 2;
   b.shader->info.workgroup_size[2] = 5;
   if (kind == 7) {
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
   }
   if (!kind) return b.shader;
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_def *offset = nir_imul_imm(&b, id, 4);
   nir_def *value = id;
   nir_variable *counter = NULL;
   if (kind == 8 || kind == 9) {
      /* The loop bound depends on a dispatch system value. Keep actual native
       * loop/break control, not an unrolled sequence with the same answer. */
      nir_def *limit = kind == 8 ? nir_channel(&b, nir_load_num_workgroups(&b), 0)
                                 : nir_iadd_imm(&b, nir_umod_imm(&b, id, 3), 1);
      counter = nir_local_variable_create(b.impl, glsl_uint_type(), "iteration");
      nir_store_var(&b, counter, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      nir_def *iteration = nir_load_var(&b, counter);
      nir_push_if(&b, nir_uge(&b, iteration, limit));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      offset = nir_iadd(&b, offset, nir_imul_imm(&b, iteration, 120));
   }
   if (kind == 1 || kind == 5 || kind == 7 || kind == 8 || kind == 9) {
      value = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 0), offset,
                            .align_mul = 4, .align_offset = 0,
                            .access = kind == 5 ? ACCESS_COHERENT : 0);
   } else if (kind == 2) {
      value = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 2), offset,
                           .align_mul = 4, .align_offset = 0,
                           .range_base = 0, .range = 4096);
   } else if (kind == 3) {
      nir_def *global = nir_load_global_invocation_id(&b, 32);
      nir_def *groups = nir_load_num_workgroups(&b);
      nir_def *local = nir_load_local_invocation_id(&b);
      value = nir_iadd(&b, nir_iadd(&b, global, groups), local);
      offset = nir_imul_imm(&b, id, 16);
   } else if (kind == 4) {
      value = nir_iadd(&b, id,
         nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 1), .base = 0,
                           .range = 1, .dest_type = nir_type_uint32));
   } else if (kind == 6) {
      value = nir_get_ssbo_size(&b, 32, nir_imm_int(&b, 0));
   } else if (kind == 10 || kind == 11) {
      nir_def *addend = kind == 10 ? nir_imm_int(&b, 1) :
         nir_iadd(&b, id, nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 1),
                      .base = 0, .range = 1, .dest_type = nir_type_uint32));
      value = nir_ssbo_atomic(&b, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                              addend, .atomic_op = nir_atomic_op_iadd,
                              .access = ACCESS_COHERENT);
   } else if (kind == 12) {
      nir_def *addend = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), offset,
                                       .align_mul = 4, .align_offset = 0);
      /* Preserve the atomic side effect even when GLSL ignores oldValue. */
      nir_ssbo_atomic(&b, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0), addend,
                       .atomic_op = nir_atomic_op_iadd, .access = ACCESS_COHERENT);
   } else if (kind == 22 || kind == 23) {
      /* PCO's real memory vectorizer combines adjacent scalar counter reads
       * into vec8/vec16. The native LD/ST path supports 1..16 DWORD bursts. */
      const unsigned width = kind == 22 ? 8 : 16;
      offset = nir_imul_imm(&b, id, 4 * width);
      value = nir_load_ssbo(&b, width, 32, nir_imm_int(&b, 0), offset,
                            .align_mul = 4, .align_offset = 0);
   } else if (kind >= 13) {
      static const nir_atomic_op operations[] = {
         nir_atomic_op_xchg, nir_atomic_op_umin, nir_atomic_op_imin,
         nir_atomic_op_umax, nir_atomic_op_imax, nir_atomic_op_iand,
         nir_atomic_op_ior, nir_atomic_op_ixor,
      };
      nir_def *operand = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), offset,
                                       .align_mul = 4, .align_offset = 0);
      if (kind == 21) {
         /* PCO lowers this into genuine usclib mutex + per-instance LD/ST,
          * not a made-up DMA atomic operation. Preserve compare/value order. */
         value = nir_ssbo_atomic_swap(&b, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                    operand, nir_iadd_imm(&b, operand, 13),
                    .atomic_op = nir_atomic_op_cmpxchg, .access = ACCESS_COHERENT);
      } else {
         check(kind - 13 < sizeof(operations) / sizeof(operations[0]), "atomic fixture kind");
         value = nir_ssbo_atomic(&b, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  operand, .atomic_op = operations[kind - 13], .access = ACCESS_COHERENT);
      }
   }
   if (kind != 12) nir_store_ssbo(&b, value, nir_imm_int(&b, 1), offset,
                  .write_mask = (1u << value->num_components) - 1,
                  .align_mul = 4, .align_offset = 0,
                  .access = kind == 5 ? ACCESS_COHERENT : 0);
   if (kind == 8 || kind == 9) {
      nir_store_var(&b, counter, nir_iadd_imm(&b, nir_load_var(&b, counter), 1), 1);
      nir_pop_loop(&b, NULL);
   }
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_ubos = kind == 2 ? 3 : 0;
   /* Deliberately smaller than the actual highest binding: capture must use
    * the final intrinsic extent, as lowered atomic buffers can append slots. */
   b.shader->info.num_ssbos = 1;
   return b.shader;
}

static void save_binary(const char *dir, unsigned kind,
                        const struct pvrgpu_pco_compute_binary *binary)
{
   if (!dir) return;
   char path[4096];
   check(snprintf(path, sizeof(path), "%s/compute-%u.bin", dir, kind) < (int)sizeof(path),
         "fixture path too long");
   FILE *file = fopen(path, "wb");
   check(file && fwrite(binary->data, 1, binary->size, file) == binary->size,
         "writing compute binary");
   check(fclose(file) == 0, "closing compute binary");
   snprintf(path, sizeof(path), "%s/compute-%u.abi", dir, kind);
   file = fopen(path, "w");
   check(file != NULL, "opening compute ABI");
   const struct pvrgpu_pco_compute_abi *a = &binary->abi;
   fprintf(file, "temps=%u inputs=%u coefficients=%u shareds=%u entry=%u\n"
      "local_size=%u,%u,%u local_index=%u,%u workgroup_id=%u,%u num_workgroups=%u,%u\n"
      "ubo=%u,%u ssbo=%u,%u push=%u,%u ubo_used=0x%x ssbo_used=0x%x read=0x%x write=0x%x\n",
      a->stage.temps, a->stage.vertex_inputs, a->stage.coefficients, a->stage.shareds,
      a->stage.entry_offset, a->local_size[0], a->local_size[1], a->local_size[2],
      a->local_invocation_index_start, a->local_invocation_index_count,
      a->workgroup_id_start, a->workgroup_id_count,
      a->num_workgroups_start, a->num_workgroups_count,
      a->stage.uniform_buffer_descriptor_start, a->stage.uniform_buffer_descriptor_count,
      a->storage_buffer_descriptor_start, a->storage_buffer_descriptor_count,
      a->stage.push_constant_start, a->stage.push_constant_count,
      a->uniform_buffer_used_mask, a->storage_buffer_used_mask,
      a->storage_buffer_read_mask, a->storage_buffer_write_mask);
   check(fclose(file) == 0, "closing compute ABI");
}

int main(int argc, char **argv)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   check(compiler != NULL, error);
   for (unsigned kind = 0; kind < 24; ++kind) {
      nir_shader *nir = make_shader(kind);
      struct pvrgpu_pco_compute_binary binary = {0};
      const bool push = kind == 4 || kind == 11;
      check(pvrgpu_pco_compile_compute(compiler, nir, push ? 8 : 0,
                                        &binary, error, sizeof(error)), error);
      check(binary.data && binary.size && !nir->info.internal, "owned native CS binary");
      const struct pvrgpu_pco_compute_abi *a = &binary.abi;
      check(a->stage.vertex_outputs == 0 && a->shared_memory_bytes == 0 &&
               a->scratch_bytes == 0, "compute has no graphics output or hidden scratch");
      check(a->local_size[0] == (kind == 7 ? 32u : 3u) &&
               a->local_size[1] == (kind == 7 ? 1u : 2u) &&
               a->local_size[2] == (kind == 7 ? 1u : 5u),
            "static local size preserved");
      check(a->stage.uniform_buffer_descriptor_count == (kind == 2 ? 3u : 0u) &&
               a->storage_buffer_descriptor_count == (kind ? 2u : 0u), "buffer slot extents");
      unsigned prefix = kind ? (kind == 2 ? 20 : 8) : 0;
      check(a->stage.push_constant_start == prefix &&
               a->stage.push_constant_count == (push ? 8u : 0u) &&
               a->stage.shareds == prefix + (push ? 8u : 0u), "canonical descriptor/CB0 layout");
      check(a->storage_buffer_write_mask == (kind >= 22 ? 2u : kind == 12 ? 1u : kind >= 10 ? 3u : kind ? 2u : 0u) &&
               a->storage_buffer_read_mask == (kind >= 22 ? 1u : kind >= 12 ? 3u : (kind == 1 || kind == 5 || kind >= 7) ? 1u : 0u),
            "actual SSBO read/write masks");
      check(a->uniform_buffer_used_mask == (kind == 2 ? 4u : 0u), "actual UBO mask");
      if (kind == 3)
         check(a->local_invocation_index_count == 1 && a->workgroup_id_count == 3 &&
                  a->num_workgroups_count == 3 && !(a->workgroup_id_start & 1) &&
                  !(a->num_workgroups_start & 1), "native VTXIN/COEFF system-value ranges");
      save_binary(argc > 1 ? argv[1] : NULL, kind, &binary);
      pvrgpu_pco_compute_binary_finish(&binary);
      check(!binary.data && !binary.size, "finish clears ownership");
      ralloc_free(nir);
   }
   for (unsigned bad = 0; bad < 6; ++bad) {
      nir_shader *nir = make_shader(1);
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
      if (bad == 0) nir->info.workgroup_size[0] = 0;
      if (bad == 1) nir->info.workgroup_size_variable = true;
      if (bad == 2) nir->info.shared_size = 4;
      if (bad == 3) {
         nir->info.workgroup_size[0] = 32;
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                                  .memory_scope = SCOPE_WORKGROUP,
                                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                                  .memory_modes = nir_var_mem_ssbo);
      }
      if (bad == 4) nir_ssbo_atomic(&b, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                     nir_imm_float(&b, 1), .atomic_op = nir_atomic_op_fadd);
      if (bad == 5) nir->info.num_images = 1;
      struct pvrgpu_pco_compute_binary binary = {0};
      check(!pvrgpu_pco_compile_compute(compiler, nir, 0, &binary, error, sizeof(error)) &&
               !binary.data && !binary.size && error[0], "unsupported CS fails closed");
      ralloc_free(nir);
   }
   nir_shader *push = make_shader(4);
   struct pvrgpu_pco_compute_binary binary = {0};
   check(!pvrgpu_pco_compile_compute(compiler, push, 4, &binary, error, sizeof(error)),
         "CB0 load past bound suffix rejected");
   ralloc_free(push);
   for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
      nir_shader *nir = make_shader(1);
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic) continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_load_ssbo) continue;
               nir_builder b = nir_builder_at(nir_before_instr(instr));
               nir_def *binding = dynamic ? nir_load_local_invocation_index(&b)
                                          : nir_iadd(&b, nir_imm_int(&b, 3), nir_imm_int(&b, -3));
               nir_src_rewrite(&intr->src[0], binding);
            }
         }
      }
      bool ok = pvrgpu_pco_compile_compute(compiler, nir, 0, &binary, error, sizeof(error));
      check(ok == !dynamic, "constant binding SSA folds before strict validation; dynamic remains rejected");
      pvrgpu_pco_compute_binary_finish(&binary);
      ralloc_free(nir);
   }
   for (unsigned bad = 0; bad < 2; ++bad) {
      nir_shader *nir = make_shader(10);
      if (bad == 0) {
         nir_foreach_function_impl(impl, nir) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr_safe(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic) continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic != nir_intrinsic_ssbo_atomic) continue;
                  nir_builder b = nir_builder_at(nir_before_instr(instr));
                  nir_src_rewrite(&intr->src[0], nir_load_local_invocation_index(&b));
               }
            }
         }
      } else {
         nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
         nir_ssbo_atomic(&b, 64, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                          nir_imm_int64(&b, 1), .atomic_op = nir_atomic_op_iadd);
      }
      check(!pvrgpu_pco_compile_compute(compiler, nir, 0, &binary, error, sizeof(error)),
            "dynamic atomic binding and 64-bit atomicAdd fail closed");
      check(!binary.data && !binary.size, "rejected atomic has no owned output");
      ralloc_free(nir);
   }
   for (unsigned fence = 0; fence < 3; ++fence) {
      nir_shader *nir = make_shader(1);
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
      const mesa_scope scope = fence == 0 ? SCOPE_WORKGROUP :
                               fence == 1 ? SCOPE_SUBGROUP : SCOPE_NONE;
      /* More than one task is legal for a memory-only fence, not for a WG
       * execution barrier. These lower through PCO, not a shader answer path. */
      if (fence == 2) nir->info.workgroup_size[0] = 32;
      nir_barrier(&b, .execution_scope = scope, .memory_scope = SCOPE_DEVICE,
                      .memory_semantics = NIR_MEMORY_ACQ_REL,
                      .memory_modes = nir_var_mem_ssbo);
      check(pvrgpu_pco_compile_compute(compiler, nir, 0, &binary, error, sizeof(error)),
            "native single-task barrier or memory-only fence is supported");
      pvrgpu_pco_compute_binary_finish(&binary);
      ralloc_free(nir);
   }
   /* Legal CAS does not require the application to mark its SSBO coherent.
    * Normalize only the owned NIR clone before PCO's usclib assertion. */
   nir_shader *cas = make_shader(21);
   nir_intrinsic_instr *cas_intr = NULL;
   nir_foreach_function_impl(impl, cas) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_ssbo_atomic_swap)
               cas_intr = nir_instr_as_intrinsic(instr);
         }
      }
   }
   check(cas_intr != NULL, "CAS qualifier test intrinsic exists");
   nir_intrinsic_set_access(cas_intr, 0);
   check(pvrgpu_pco_compile_compute(compiler, cas, 0, &binary, error, sizeof(error)),
         "CAS without coherent qualifier must compile into coherent usclib");
   check(nir_intrinsic_access(cas_intr) == 0, "CAS normalization mutated caller NIR");
   pvrgpu_pco_compute_binary_finish(&binary);
   ralloc_free(cas);
   for (unsigned kind = 22; kind <= 23; ++kind) {
      nir_shader *nir = make_shader(kind);
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic) continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_ssbo)
                  nir_intrinsic_set_write_mask(intr,
                     1u | (1u << (intr->src[0].ssa->num_components - 1)));
            }
         }
      }
      check(!pvrgpu_pco_compile_compute(compiler, nir, 0, &binary, error, sizeof(error)) &&
               !binary.data && !binary.size && strstr(error, "full component write mask"),
            "sparse vec8/vec16 store must not overwrite unselected components");
      ralloc_free(nir);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   puts("native compute compiler tests: PASS (29 programs, 12 fail-closed inputs)");
   return 0;
}
