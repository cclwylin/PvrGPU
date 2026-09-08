/*
 * Exercise the actual Mesa PCO parallel-copy emitter with physical IR refs.
 * The oracle applies every requested copy to the original register state,
 * then compares all registers against sequential execution of emitted MBYP,
 * MBYP2 and MOVS1 instructions. MBYP2 reads both sources before writing.
 *
 * Run via script/run_mesa_pco_register_bank_unit.sh. This translation unit
 * intentionally includes the configured external Mesa allocator to reach its
 * static helper; the allocator itself is not duplicated in this repository.
 */
#define pco_ra test_unused_pco_ra
#ifndef PVRGPU_PCO_RA_SOURCE
#define PVRGPU_PCO_RA_SOURCE "pco_ra.c"
#endif
#include PVRGPU_PCO_RA_SOURCE
#undef pco_ra

#include <math.h>
#include <string.h>

enum { BANKS = 16, REGS = 512, MAX_COPIES = 16 };
static unsigned cases_run, instructions_run, failed;
static uint32_t rng = 0x34862179;

static uint32_t random_u32(void)
{
   rng ^= rng << 13;
   rng ^= rng >> 17;
   rng ^= rng << 5;
   return rng;
}

static unsigned reg_key(pco_ref ref)
{
   assert(pco_ref_is_reg(ref));
   assert(ref.reg_class < BANKS && ref.val < REGS);
   return ref.reg_class * REGS + ref.val;
}

static uint32_t float_bits(float x)
{
   uint32_t value;
   memcpy(&value, &x, sizeof(value));
   return value;
}

static uint32_t read_ref(const uint32_t *state, pco_ref ref)
{
   uint32_t value = pco_ref_is_imm(ref) ? ref.val : state[reg_key(ref)];
   float x;
   memcpy(&x, &value, sizeof(x));
   if (ref.abs)
      x = fabsf(x);
   if (ref.neg)
      x = -x;
   /* MBYP/MBYP2 permit abs and neg source modifiers only. */
   assert(!ref.flr && !ref.oneminus && !ref.clamp && ref.elem == 0);
   return float_bits(x);
}

static pco_ref temp(unsigned n) { return pco_ref_hwreg(n, PCO_REG_CLASS_TEMP); }
static pco_ref vtxin(unsigned n) { return pco_ref_hwreg(n, PCO_REG_CLASS_VTXIN); }
static pco_copy copy(pco_ref dest, pco_ref src)
{
   return (pco_copy){ .dest = dest, .src = src };
}

static bool run_case(const char *name, pco_copy *input, unsigned count)
{
   uint32_t initial[BANKS * REGS], expected[BANKS * REGS], actual[BANKS * REGS];
   for (unsigned i = 0; i < ARRAY_SIZE(initial); ++i)
      initial[i] = float_bits((i & 1 ? -1.0f : 1.0f) * (i + 0.25f));
   memcpy(expected, initial, sizeof(initial));
   memcpy(actual, initial, sizeof(initial));
   for (unsigned i = 0; i < count; ++i)
      expected[reg_key(input[i].dest)] = read_ref(initial, input[i].src);

   pco_shader *shader = rzalloc_size(NULL, sizeof(*shader));
   list_inithead(&shader->funcs);
   pco_func *func = pco_func_create(shader, PCO_FUNC_TYPE_ENTRYPOINT, 0);
   pco_block *block = pco_block_create(func);
   pco_builder builder = pco_builder_create(func, pco_cursor_before_block(block));
   struct util_dynarray copies;
   util_dynarray_init(&copies, shader);
   for (unsigned i = 0; i < count; ++i)
      util_dynarray_append(&copies, input[i]);
   const enum pco_exec_cnd conditions[] = {
      PCO_EXEC_CND_E1_ZX, PCO_EXEC_CND_E1_Z1, PCO_EXEC_CND_EX_ZX,
      PCO_EXEC_CND_E1_Z0,
   };
   enum pco_exec_cnd condition = conditions[cases_run % ARRAY_SIZE(conditions)];
   emit_copies(&builder, &copies, condition);

   unsigned emitted = 0;
   pco_foreach_instr_in_block(instr, block) {
      assert(pco_instr_get_exec_cnd(instr) == condition);
      assert(instr->op == PCO_OP_MBYP || instr->op == PCO_OP_MOVS1 ||
             instr->op == PCO_OP_MBYP2);
      unsigned n = instr->op == PCO_OP_MBYP2 ? 2 : 1;
      assert(instr->num_dests == n && instr->num_srcs == n);
      uint32_t values[2];
      for (unsigned i = 0; i < n; ++i)
         values[i] = read_ref(actual, instr->src[i]);
      for (unsigned i = 0; i < n; ++i)
         actual[reg_key(instr->dest[i])] = values[i];
      ++emitted;
   }
   assert(emitted <= count);
   bool ok = memcmp(expected, actual, sizeof(actual)) == 0;
   if (!ok && failed < 8) {
      fprintf(stderr, "FAIL %s (%u copies, %u instructions)\n", name, count, emitted);
      for (unsigned i = 0; i < count; ++i)
         fprintf(stderr, "  bank%u[%u] <- type%u bank%u[%u] abs=%u neg=%u\n",
                 input[i].dest.reg_class, input[i].dest.val,
                 input[i].src.type, input[i].src.reg_class, input[i].src.val,
                 input[i].src.abs, input[i].src.neg);
      for (unsigned i = 0; i < ARRAY_SIZE(actual); ++i)
         if (actual[i] != expected[i])
            fprintf(stderr, "  bank%u[%u] actual=%08x expected=%08x\n",
                    i / REGS, i % REGS, actual[i], expected[i]);
   }
   ++cases_run;
   failed += !ok;
   instructions_run += emitted;
   ralloc_free(shader);
   return ok;
}

#define CASE(name, ...) do { \
   pco_copy input[] = { __VA_ARGS__ }; \
   run_case(name, input, ARRAY_SIZE(input)); \
} while (0)

int main(void)
{
   CASE("vtxin-only", copy(vtxin(8), vtxin(10)), copy(vtxin(9), vtxin(11)));
   CASE("bank-distinct-indices", copy(temp(0), vtxin(0)), copy(vtxin(1), temp(1)));
   CASE("temp-cycle", copy(temp(0), temp(1)), copy(temp(1), temp(0)));
   CASE("vtxin-cycle", copy(vtxin(10), vtxin(11)), copy(vtxin(11), vtxin(10)));
   CASE("cross-bank-cycle", copy(temp(0), vtxin(0)), copy(vtxin(0), temp(0)));
   CASE("three-cycle", copy(temp(0), temp(1)), copy(temp(1), vtxin(0)),
        copy(vtxin(0), temp(0)));
   CASE("repeated-sources", copy(temp(0), temp(1)), copy(temp(2), temp(1)),
        copy(temp(1), temp(3)), copy(vtxin(0), temp(1)));
   CASE("cycle-repeated-sources", copy(temp(0), temp(1)), copy(temp(1), temp(0)),
        copy(vtxin(0), temp(0)), copy(vtxin(1), temp(1)));
   CASE("immediate-and-constant", copy(temp(0), pco_ref_imm32(float_bits(2.5f))),
        copy(vtxin(0), pco_fone), copy(vtxin(1), pco_fnegone));
   CASE("special-source", ((pco_copy){.dest = temp(1),
        .src = pco_ref_hwreg(PCO_SR_LOCAL_ADDR_INST_NUM, PCO_REG_CLASS_SPEC), .s1 = true}));
   CASE("negative-dependency", copy(temp(0), temp(1)), copy(temp(1), pco_ref_neg(temp(0))));
   CASE("negative-cycle-first", copy(temp(0), pco_ref_neg(temp(1))), copy(temp(1), temp(0)));
   CASE("abs-negative-cycle", copy(vtxin(0), pco_ref_abs(temp(0))),
        copy(temp(0), pco_ref_neg(vtxin(0))));
   CASE("negative-repeated-source", copy(temp(0), temp(1)),
        copy(temp(1), pco_ref_neg(temp(0))), copy(vtxin(0), pco_ref_abs(temp(0))));
   CASE("modified-self-and-reader", copy(temp(0), pco_ref_neg(temp(0))),
        copy(temp(1), temp(0)));
   CASE("modified-self-after-cycle", copy(temp(0), pco_ref_neg(temp(1))),
        copy(temp(1), pco_ref_neg(temp(0))));

   for (unsigned iteration = 0; iteration < 20000; ++iteration) {
      pco_copy input[MAX_COPIES];
      unsigned destinations[MAX_COPIES];
      for (unsigned i = 0; i < MAX_COPIES; ++i)
         destinations[i] = i;
      for (unsigned i = MAX_COPIES - 1; i > 0; --i) {
         unsigned j = random_u32() % (i + 1);
         unsigned swap = destinations[i];
         destinations[i] = destinations[j];
         destinations[j] = swap;
      }
      unsigned count = random_u32() % MAX_COPIES + 1;
      for (unsigned i = 0; i < count; ++i) {
         unsigned d = destinations[i], s = random_u32() % 20;
         pco_ref dest = d < 8 ? temp(d) : vtxin(d - 8);
         pco_ref source = s < 8 ? temp(s) : s < 16 ? vtxin(s - 8) :
                          s < 18 ? pco_ref_imm32(float_bits(s + 0.75f)) :
                                   pco_ref_hwreg(s, PCO_REG_CLASS_CONST);
         unsigned mods = random_u32() % 4;
         if (mods & 1)
            source = pco_ref_abs(source);
         if (mods & 2)
            source = pco_ref_neg(source);
         input[i] = copy(dest, source);
      }
      run_case("randomized-physical-parallel-assignment", input, count);
   }
   printf("%u cases, %u emitted instructions, %u failures\n",
          cases_run, instructions_run, failed);
   return failed != 0;
}
