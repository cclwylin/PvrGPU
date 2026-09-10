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

static nir_shader *make_shared_shader(unsigned kind)
{
   const unsigned sizes[] = {1, 30, 33, 64, 96, 1024, 64};
   const unsigned count = sizes[kind - 24];
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      pco_nir_options(), "native_workgroup_shared");
   b.shader->info.workgroup_size[0] = count;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_def *group = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *output = nir_imul_imm(&b,
      nir_iadd(&b, nir_imul_imm(&b, group, count), id), 4);
   nir_variable *array = nir_variable_create(b.shader, nir_var_mem_shared,
      glsl_array_type(glsl_uint_type(), count, 0), "workgroup_values");
   nir_deref_instr *base = nir_build_deref_var(&b, array);
   nir_def *reverse = nir_isub(&b, nir_imm_int(&b, count - 1), id);
   nir_store_deref(&b, nir_build_deref_array(&b, base, reverse),
      nir_iadd(&b, nir_imul(&b, id, id), nir_imul_imm(&b, group, count)), 1);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
      .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
      .memory_modes = nir_var_mem_shared);
   nir_def *value = nir_load_deref(&b, nir_build_deref_array(&b, base, id));
   if (kind == 30) {
      /* Real shared atomic results, followed by a second reusable barrier. */
      nir_deref_instr *cell = nir_build_deref_array(&b, base, nir_imm_int(&b, 0));
      value = nir_deref_atomic(&b, 32, &cell->def, nir_imm_int(&b, 1),
                               .atomic_op = nir_atomic_op_iadd);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
         .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
         .memory_modes = nir_var_mem_shared);
      value = nir_iadd(&b, value, nir_load_deref(&b, cell));
   }
   nir_store_ssbo(&b, value, nir_imm_int(&b, 0), output,
                   .write_mask = 1, .align_mul = 4);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_ssbos = 1;
   return b.shader;
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
   fprintf(file, "workgroup_shared=%u,%u,%u\n", a->shared_memory_bytes,
      a->shared_memory_descriptor_start, a->shared_memory_descriptor_count);
   check(fclose(file) == 0, "closing compute ABI");
}

/* The native execution fixture checks every memory transaction, including
 * holes between components and padding around each invocation's record. */
static unsigned test_masked_stores(struct pvrgpu_pco_compiler *compiler,
                                   const char *dir)
{
   FILE *manifest = NULL;
   if (dir) {
      char path[4096];
      check(snprintf(path, sizeof(path), "%s/masked-stores.txt", dir) < (int)sizeof(path),
            "masked fixture manifest path");
      manifest = fopen(path, "w");
      check(manifest != NULL, "opening masked fixture manifest");
   }
   const unsigned widths[] = {1, 2, 3, 4, 8, 16};
   unsigned cases = 0;
   for (unsigned w = 0; w < ARRAY_SIZE(widths); ++w) {
      const unsigned width = widths[w], full = BITFIELD_MASK(width);
      const unsigned wide_masks[] = {0, 1, 1u << (width - 1), full,
         full & 0x5555, full & 0xaaaa, full & ~3u, 1u | (1u << (width - 1)), 6};
      const unsigned mask_count = width <= 4 ? full + 1 : ARRAY_SIZE(wide_masks);
      for (unsigned m = 0; m < mask_count; ++m) {
         const unsigned mask = width <= 4 ? m : wide_masks[m];
         for (unsigned branch = 0; branch < 2; ++branch) {
            nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
               pco_nir_options(), "native_masked_ssbo_store");
            b.shader->info.workgroup_size[0] = 37;
            b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
            nir_def *local = nir_load_local_invocation_index(&b);
            nir_def *group = nir_channel(&b, nir_load_workgroup_id(&b), 0);
            nir_def *index = nir_iadd(&b, nir_imul_imm(&b, group, 37), local);
            const unsigned stride = branch ? ALIGN_POT(width + 1, 16) : ALIGN_POT(width, 4) + 4;
            const unsigned align = branch ? 64 : 16, align_offset = branch ? 4 : 0;
            nir_def *offset = nir_iadd_imm(&b, nir_imul_imm(&b, index, stride * 4), branch ? 4 : 16);
            nir_def *components[16];
            for (unsigned c = 0; c < width; ++c)
               components[c] = nir_iadd_imm(&b, nir_imul_imm(&b, index, 256), 0x12340000 + c);
            nir_def *value = nir_vec(&b, components, width);
            if (branch)
               nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, local, 1), 0));
            const unsigned access = branch ? ACCESS_COHERENT | ACCESS_VOLATILE : ACCESS_RESTRICT;
            nir_intrinsic_instr *store = nir_store_ssbo(&b, value,
               nir_imm_int(&b, branch ? 3 : 0), offset,
               .write_mask = mask, .align_mul = align, .align_offset = align_offset, .access = access);
            /* The builder treats zero as its default full mask. Explicitly
             * construct the legal zero-write no-op after that defaulting. */
            nir_intrinsic_set_write_mask(store, mask);
            if (branch)
               nir_pop_if(&b, NULL);
            nir_shader_gather_info(b.shader, b.impl);
            b.shader->info.num_ssbos = branch ? 4 : 1;
            char error[512] = {0};
            struct pvrgpu_pco_compute_binary binary = {0};
            if (!pvrgpu_pco_compile_compute(compiler, b.shader, 0,
                                             &binary, error, sizeof(error))) {
               fprintf(stderr, "masked store width=%u mask=0x%x branch=%u: %s\n",
                       width, mask, branch, error);
               abort();
            }
            check(binary.abi.storage_buffer_read_mask == 0,
                  "masked store invented a destination load");
            check(binary.abi.storage_buffer_write_mask == (mask ? 1u << (branch ? 3 : 0) : 0),
                  "masked store lost selected buffer or retained a zero-mask binding");
            check(nir_intrinsic_write_mask(store) == mask &&
                     nir_intrinsic_access(store) == access &&
                     nir_intrinsic_align_mul(store) == align &&
                     nir_intrinsic_align_offset(store) == align_offset &&
                     store->src[0].ssa->num_components == width,
                  "masked-store lowering mutated caller NIR");
            const unsigned kind = 400 + cases++;
            save_binary(dir, kind, &binary);
            if (manifest)
               fprintf(manifest, "%u %u %u %u\n", kind, width, mask, branch);
            pvrgpu_pco_compute_binary_finish(&binary);
            ralloc_free(b.shader);
         }
      }
   }
   if (manifest)
      check(fclose(manifest) == 0, "closing masked fixture manifest");
   return cases;
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
      if (bad == 2) nir->info.shared_size = 32 * 1024 + 4;
      if (bad == 3) {
         nir->info.workgroup_size[0] = 32;
         nir_barrier(&b, .execution_scope = SCOPE_DEVICE,
                                  .memory_scope = SCOPE_WORKGROUP,
                                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                                  .memory_modes = nir_var_mem_ssbo);
      }
      if (bad == 4) nir_ssbo_atomic(&b, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                     nir_imm_float(&b, 1), .atomic_op = nir_atomic_op_fadd);
      if (bad == 5) nir->info.num_images = 33;
      struct pvrgpu_pco_compute_binary binary = {0};
      check(!pvrgpu_pco_compile_compute(compiler, nir, 0, &binary, error, sizeof(error)) &&
               !binary.data && !binary.size && error[0], "unsupported CS fails closed");
      ralloc_free(nir);
   }
   for (unsigned kind = 24; kind <= 30; ++kind) {
      nir_shader *nir = make_shared_shader(kind);
      struct pvrgpu_pco_compute_binary native = {0};
      check(pvrgpu_pco_compile_compute(compiler, nir, 0, &native,
                                       error, sizeof(error)), error);
      check(native.data && native.size && native.abi.shared_memory_bytes &&
         native.abi.shared_memory_bytes <= 32768 &&
         native.abi.shared_memory_descriptor_start == 4 &&
         native.abi.shared_memory_descriptor_count == 4 &&
         native.abi.stage.push_constant_start == 8 &&
         native.abi.stage.shareds == 8 && !native.abi.stage.vertex_outputs &&
         native.abi.storage_buffer_descriptor_count == 1 &&
         native.abi.storage_buffer_used_mask == 1,
         "shared memory has a private descriptor, not an extra user SSBO");
      save_binary(argc > 1 ? argv[1] : NULL, kind, &native);
      pvrgpu_pco_compute_binary_finish(&native);
      ralloc_free(nir);
   }
   nir_shader *push = make_shader(4);
   struct pvrgpu_pco_compute_binary binary = {0};
   check(!pvrgpu_pco_compile_compute(compiler, push, 4, &binary, error, sizeof(error)),
         "CB0 load past bound suffix rejected");
   ralloc_free(push);
   for (unsigned components = 1; components <= 4; ++components) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
         pco_nir_options(), "native_dynamic_cb0");
      b.shader->info.workgroup_size[0] = 8;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *index = nir_load_local_invocation_index(&b);
      nir_def *value = nir_load_uniform(&b, components, 32, index,
         .base = 1, .range = 8, .dest_type = nir_type_uint32);
      nir_store_ssbo(&b, value, nir_imm_int(&b, 0), nir_imul_imm(&b, index, 16),
         .write_mask = (1u << components) - 1, .align_mul = 4);
      nir_shader_gather_info(b.shader, b.impl);
      b.shader->info.num_ssbos = 1;
      check(pvrgpu_pco_compile_compute(compiler, b.shader, 36, &binary, error, sizeof(error)), error);
      check(binary.abi.stage.push_constant_count == 36 &&
            binary.abi.stage.push_constant_start == 4,
            "invocation-indexed vec4-slot CB0 uses native bounded shared-register loads");
      pvrgpu_pco_compute_binary_finish(&binary);
      check(!pvrgpu_pco_compile_compute(compiler, b.shader, 32, &binary, error, sizeof(error)),
            "dynamic CB0 extent beyond capture remains fail-closed");
      ralloc_free(b.shader);
   }
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
   for (unsigned shape = 0; shape < 4; ++shape) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
         pco_nir_options(), "native_compute_texture_fifo");
      b.shader->info.workgroup_size[0] = 4;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *id = nir_load_local_invocation_index(&b);
      nir_def *coords = nir_load_uniform(&b, 4, 32, id, .range = 4,
                                         .dest_type = nir_type_float32);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
      tex->op = nir_texop_tex;
      tex->sampler_dim = shape == 1 ? GLSL_SAMPLER_DIM_3D :
         shape == 2 ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
      tex->is_array = shape == 3;
      tex->coord_components = shape ? 3 : 2;
      tex->dest_type = nir_type_float32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
         nir_trim_vector(&b, coords, tex->coord_components));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_ssbo(&b, &tex->def, nir_imm_int(&b, 0), nir_imul_imm(&b, id, 16),
                     .write_mask = 15, .align_mul = 16);
      nir_shader_gather_info(b.shader, b.impl);
      b.shader->info.num_textures = b.shader->info.num_ssbos = 1;
      check(pvrgpu_pco_compile_compute(compiler, b.shader, 16, &binary, error, sizeof(error)), error);
      check(binary.abi.sampled_texture_count == 1 &&
            binary.abi.stage.uniform_buffer_descriptor_start == 20 &&
            binary.abi.storage_buffer_descriptor_start == 20 &&
            binary.abi.stage.push_constant_start == 24 && binary.abi.stage.shareds == 40,
            "compute SMP descriptors have an independent native set before buffers and CB0");
      save_binary(argc > 1 ? argv[1] : NULL, 300 + shape, &binary);
      pvrgpu_pco_compute_binary_finish(&binary);
      ralloc_free(b.shader);
   }
   for (unsigned ubos = 0; ubos < 2; ++ubos) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
         pco_nir_options(), "native_large_cb0_dma");
      b.shader->info.workgroup_size[0] = 128;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *id = nir_load_local_invocation_index(&b);
      nir_def *value = nir_load_uniform(&b, 4, 32, nir_umod_imm(&b, id, 121),
         .range = 121, .dest_type = nir_type_uint32);
      if (ubos) value = nir_iadd(&b, value, nir_load_ubo(&b, 4, 32,
         nir_imm_int(&b, 2), nir_imm_int(&b, 0), .align_mul = 16, .range = 16));
      nir_store_ssbo(&b, value, nir_imm_int(&b, 0), nir_imul_imm(&b, id, 16),
         .write_mask = 15, .align_mul = 16);
      nir_shader_gather_info(b.shader, b.impl);
      b.shader->info.num_ssbos = 1;
      b.shader->info.num_ubos = ubos ? 3 : 0;
      check(pvrgpu_pco_compile_compute(compiler, b.shader, 484, &binary, error, sizeof(error)), error);
      check(binary.abi.cb0_uniform_buffer_slot == (ubos ? 4u : 1u) &&
            binary.abi.stage.uniform_buffer_descriptor_count == (ubos ? 4u : 1u) &&
            binary.abi.stage.push_constant_count == 0 &&
            binary.abi.stage.shareds == (ubos ? 20u : 8u),
            "large CB0 maps to its own native UBO DMA descriptor without register spills");
      pvrgpu_pco_compute_binary_finish(&binary);
      check(!pvrgpu_pco_compile_compute(compiler, b.shader, 480, &binary, error, sizeof(error)),
            "large dynamic CB0 out-of-range extent remains fail-closed");
      ralloc_free(b.shader);
   }
   /* Compute has no graphics-style CB0 prefix proof before the shared large
    * CB0 DMA lowering, so that helper must reject a signed negative base and
    * every unchecked base/offset/range sum itself.  Base -1 with direct
    * offset 1 previously widened to UINT64_MAX and wrapped the last slot to
    * zero, passing the span check with a negative byte address. */
   for (unsigned bad = 0; bad < 3; ++bad) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
         pco_nir_options(), "native_large_cb0_dma_invalid");
      b.shader->info.workgroup_size[0] = 128;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *id = nir_load_local_invocation_index(&b);
      nir_def *index = bad == 2 ? nir_umod_imm(&b, id, 121)
                     : nir_imm_int(&b, bad == 0 ? 1 : UINT32_MAX);
      const int base = bad == 0 ? -1 : bad == 1 ? 1 : INT32_MAX;
      nir_def *value = nir_load_uniform(&b, 4, 32, index,
         .base = base, .range = bad == 2 ? 121 : 1, .dest_type = nir_type_uint32);
      nir_store_ssbo(&b, value, nir_imm_int(&b, 0), nir_imul_imm(&b, id, 16),
         .write_mask = 15, .align_mul = 16);
      nir_shader_gather_info(b.shader, b.impl);
      b.shader->info.num_ssbos = 1;
      b.shader->info.num_ubos = 0;
      error[0] = '\0';
      check(!pvrgpu_pco_compile_compute(compiler, b.shader, 484, &binary, error, sizeof(error)) &&
               !binary.data && !binary.size && strstr(error, "CB0 DMA") != NULL,
            bad == 0 ? "large CB0 DMA negative base is rejected by the shared helper"
            : bad == 1 ? "large CB0 DMA direct offset past the 32-bit byte address is rejected"
                       : "large CB0 DMA indirect base/range sum past the span is rejected");
      ralloc_free(b.shader);
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
      check(pvrgpu_pco_compile_compute(compiler, nir, 0, &binary, error, sizeof(error)),
            "sparse vec8/vec16 stores lower to selected native component writes");
      check(binary.abi.storage_buffer_read_mask == 1 &&
               binary.abi.storage_buffer_write_mask == 2,
            "sparse stores do not invent destination reads");
      pvrgpu_pco_compute_binary_finish(&binary);
      ralloc_free(nir);
   }
   const unsigned masked_cases = test_masked_stores(compiler, argc > 1 ? argv[1] : NULL);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("native compute compiler tests: PASS (including %u masked-store programs)\n", masked_cases);
   return 0;
}
