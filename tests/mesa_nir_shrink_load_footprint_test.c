/*
 * Exercise the selected real Mesa NIR pass, not a reimplementation. Every
 * resulting load must stay in its original resource byte interval, including
 * rounded but unused lanes. Stores preserve the original selected values.
 * Run with script/run_mesa_nir_shrink_load_unit.sh (isolated ASan/UBSan).
 */
#define nir_opt_shrink_vectors pvrgpu_test_nir_opt_shrink_vectors
#ifndef PVRGPU_NIR_SHRINK_SOURCE
#define PVRGPU_NIR_SHRINK_SOURCE "nir_opt_shrink_vectors.c"
#endif
#include PVRGPU_NIR_SHRINK_SOURCE
#undef nir_opt_shrink_vectors

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned cases_run, checks, pipeline_cases;
static const nir_shader_compiler_options options = {0};

static void check(bool condition, const char *why)
{
   ++checks;
   if (!condition) {
      fprintf(stderr, "FAIL case %u: %s\n", cases_run, why);
      exit(1);
   }
}

static uint64_t memory_word(unsigned byte, unsigned bits)
{
   const uint64_t value = UINT64_C(0xd38173ea4c926fb5) ^
                          (UINT64_C(0x9e3779b97f4a7c15) * (byte + 1));
   return bits == 64 ? value : value & BITFIELD64_MASK(bits);
}

static uint64_t eval(nir_def *def, unsigned component, unsigned depth)
{
   check(depth < 64 && component < def->num_components, "bounded SSA evaluator");
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return nir_const_value_as_uint(nir_instr_as_load_const(instr)->value[component], def->bit_size);
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
      check(load->intrinsic == nir_intrinsic_load_ubo || load->intrinsic == nir_intrinsic_load_ssbo ||
            load->intrinsic == nir_intrinsic_load_global, "only real memory load values");
      nir_src *offset = nir_get_io_offset_src(load);
      check(offset != NULL, "memory offset source");
      return memory_word(eval(offset->ssa, 0, depth + 1) + component * (def->bit_size / 8), def->bit_size);
   }
   check(instr->type == nir_instr_type_alu, "constant/load/ALU SSA expression");
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op == nir_op_mov)
      return eval(alu->src[0].src.ssa, alu->src[0].swizzle[component], depth + 1);
   if (strncmp(nir_op_infos[alu->op].name, "vec", 3) == 0)
      return eval(alu->src[component].src.ssa, alu->src[component].swizzle[0], depth + 1);
   check(alu->op == nir_op_iadd, "supported address arithmetic");
   const uint64_t value = eval(alu->src[0].src.ssa, alu->src[0].swizzle[component], depth + 1) +
                          eval(alu->src[1].src.ssa, alu->src[1].swizzle[component], depth + 1);
   return def->bit_size == 64 ? value : value & BITFIELD64_MASK(def->bit_size);
}

static nir_def *load(nir_builder *b, nir_intrinsic_op op, unsigned components,
                     unsigned bits, unsigned offset)
{
   nir_def *base = nir_imm_int(b, 1);
   nir_def *address = nir_imm_intN_t(b, offset, op == nir_intrinsic_load_global ? 64 : 32);
   nir_intrinsic_instr *intr = nir_intrinsic_instr_create(b->shader, op);
   nir_def_init(&intr->instr, &intr->def, components, bits);
   intr->num_components = components;
   if (op == nir_intrinsic_load_global) intr->src[0] = nir_src_for_ssa(address);
   else { intr->src[0] = nir_src_for_ssa(base); intr->src[1] = nir_src_for_ssa(address); }
   nir_intrinsic_set_align(intr, 64, offset % 64);
   nir_intrinsic_set_access(intr, 0);
   if (nir_intrinsic_has_range_base(intr)) {
      nir_intrinsic_set_range_base(intr, offset);
      nir_intrinsic_set_range(intr, components * bits / 8);
   }
   nir_builder_instr_insert(b, &intr->instr);
   return &intr->def;
}

static nir_intrinsic_instr *use(nir_builder *b, nir_def *value, unsigned component, unsigned output)
{
   nir_def *scalar = nir_channel(b, value, component);
   nir_store_ssbo(b, scalar, nir_imm_int(b, 3), nir_imm_int(b, output * 8),
                  .align_mul = 8, .write_mask = 1);
   return nir_instr_as_intrinsic(nir_block_last_instr(nir_cursor_current_block(b->cursor)));
}

static unsigned verify_loads(nir_builder *b, nir_intrinsic_op op, unsigned first_byte, unsigned end_byte)
{
   unsigned loads = 0;
   nir_foreach_block(block, b->impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != op) continue;
         ++loads;
         const uint64_t address = eval(nir_get_io_offset_src(intr)->ssa, 0, 0);
         const unsigned bytes = intr->num_components * intr->def.bit_size / 8;
         if (!(address >= first_byte && address + bytes <= end_byte)) {
            fprintf(stderr, "load op=%u: [0x%" PRIx64 ",0x%" PRIx64 ") escaped [%u,%u)\n",
                    op, address, address + bytes, first_byte, end_byte);
            nir_print_shader(b->shader, stderr);
         }
         check(address >= first_byte && address + bytes <= end_byte, "all fetched bytes stay within original binding");
         check(intr->num_components == intr->def.num_components, "intrinsic/SSA width agreement");
         const unsigned alignment = nir_intrinsic_align_mul(intr);
         check(alignment && util_is_power_of_two_nonzero(alignment) &&
               nir_intrinsic_align_offset(intr) == address % alignment,
               "alignment follows actual shifted address");
         if (nir_intrinsic_has_range_base(intr)) {
            const uint64_t range_base = nir_intrinsic_range_base(intr);
            const uint64_t range_end = range_base + nir_intrinsic_range(intr);
            check(range_base <= address && range_end >= address + bytes &&
                  range_base >= first_byte && range_end <= end_byte,
                  "range metadata remains sound and inside original resource");
         }
         /* Validate even unused lanes, with no padded allocation or ignored OOB. */
         for (unsigned c = 0; c < intr->num_components; ++c)
            check(eval(&intr->def, c, 0) == memory_word(address + c * intr->def.bit_size / 8, intr->def.bit_size),
                  "real fetched component payload");
      }
   }
   return loads;
}

static void run_mask(nir_intrinsic_op op, unsigned count, unsigned bits, unsigned offset, unsigned mask)
{
   ++cases_run;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "shrink footprint matrix");
   nir_def *value = load(&b, op, count, bits, offset);
   nir_intrinsic_instr *stores[16] = {0};
   for (unsigned c = 0; c < count; ++c) if (mask & (1u << c)) stores[c] = use(&b, value, c, c);
   pvrgpu_test_nir_opt_shrink_vectors(b.shader, true);
   check(verify_loads(&b, op, offset, offset + count * bits / 8) == 1, "single load retained");
   for (unsigned c = 0; c < count; ++c) if (stores[c])
      check(eval(stores[c]->src[0].ssa, 0, 0) == memory_word(offset + c * bits / 8, bits), "selected output exact after reswizzle");
   check(!pvrgpu_test_nir_opt_shrink_vectors(b.shader, true), "shrink reaches fixed point");
   nir_validate_shader(b.shader, "footprint matrix");
   ralloc_free(b.shader);
}

/* Same no-hole/valid-width contract as PCO; use the actual Mesa vectorizer. */
static bool vectorize(unsigned align_mul, unsigned align_offset, unsigned bits, unsigned components,
                       int64_t hole, nir_intrinsic_instr *low, nir_intrinsic_instr *high, void *data)
{
   (void)align_mul; (void)align_offset; (void)low; (void)high; (void)data;
   return bits <= 32 && hole <= 0 && nir_num_components_valid(components);
}

static void run_pipeline(void)
{
   ++cases_run; ++pipeline_cases;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "adjacent vec4 UBO w components");
   nir_def *low = load(&b, nir_intrinsic_load_ubo, 4, 32, 0);
   nir_def *high = load(&b, nir_intrinsic_load_ubo, 4, 32, 16);
   nir_intrinsic_instr *lo_store = use(&b, low, 3, 0);
   nir_intrinsic_instr *hi_store = use(&b, high, 3, 1);
   nir_load_store_vectorize_options opts = {.modes = nir_var_mem_ubo, .callback = vectorize};
   /* PCO runs CSE before memory vectorization; equal block constants need
    * that same SSA identity before the generic vectorizer can merge them. */
   nir_opt_cse(b.shader);
   check(nir_opt_load_store_vectorize(b.shader, &opts), "actual vectorizer merges adjacent vec4 loads");
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   check(verify_loads(&b, nir_intrinsic_load_ubo, 0, 32) == 1, "one merged UBO load");
   pvrgpu_test_nir_opt_shrink_vectors(b.shader, true);
   check(verify_loads(&b, nir_intrinsic_load_ubo, 0, 32) == 1, "merged load remains in exact 32-byte binding");
   check(eval(lo_store->src[0].ssa, 0, 0) == memory_word(12, 32), "first original w preserved");
   check(eval(hi_store->src[0].ssa, 0, 0) == memory_word(28, 32), "second original w preserved");
   check(!pvrgpu_test_nir_opt_shrink_vectors(b.shader, true), "pipeline fixed point");
   nir_validate_shader(b.shader, "vectorize then shrink");
   ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const nir_intrinsic_op ops[] = {nir_intrinsic_load_ubo, nir_intrinsic_load_ssbo, nir_intrinsic_load_global};
   const unsigned widths[] = {2, 3, 4, 5, 8, 16};
   const unsigned bits[] = {8, 16, 32, 64};
   const unsigned offsets[] = {0, 12, 64};
   /* Run the real vectorize->shrink regression first: unpatched code must fail. */
   run_pipeline();
   for (unsigned o = 0; o < ARRAY_SIZE(ops); ++o)
      for (unsigned w = 0; w < ARRAY_SIZE(widths); ++w)
         for (unsigned z = 0; z < ARRAY_SIZE(bits); ++z)
            for (unsigned a = 0; a < ARRAY_SIZE(offsets); ++a) {
               const unsigned n = widths[w];
               if (n <= 8) {
                  for (unsigned mask = 1; mask < (1u << n); ++mask)
                     run_mask(ops[o], n, bits[z], offsets[a], mask);
               } else {
                  for (unsigned first = 0; first < n; ++first)
                     for (unsigned last = first; last < n; ++last) {
                        const unsigned endpoints = (1u << first) | (1u << last);
                        const unsigned span = BITFIELD_RANGE(first, last - first + 1);
                        run_mask(ops[o], n, bits[z], offsets[a], endpoints);
                        run_mask(ops[o], n, bits[z], offsets[a], span);
                        run_mask(ops[o], n, bits[z], offsets[a], (span & 0xaaaa) | endpoints);
                     }
               }
            }
   glsl_type_singleton_decref();
   printf("NIR shrink load footprint PASS: %u cases, %u checks, %u actual vectorize pipeline\n", cases_run, checks, pipeline_cases);
   return 0;
}
