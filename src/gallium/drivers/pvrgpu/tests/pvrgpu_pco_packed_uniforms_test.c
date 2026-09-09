/* SPDX-License-Identifier: MIT */
/* Real NIR -> native PCO and direct preflight tests for driver-only CB0 maps.
 * No model/shader execution. Optional PVRGPU_PACKED_FIXTURE_DIR saves genuine
 * bytes, stable ABI and private map for independently checked ISS fixtures. */
#include "../pvrgpu_pco.c"
#include <stdio.h>

static unsigned checks;
static void check(bool value, const char *message)
{
   ++checks;
   if (!value) { fprintf(stderr, "packed_uniforms FAIL: %s\n", message); exit(1); }
}
static nir_shader *vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, pco_nir_options(), "packed_cb0_vertex");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, glsl_vec4_type(), "position");
   in->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "position_out");
   out->data.location = VARYING_SLOT_POS;
   nir_store_var(&b, out, nir_load_var(&b, in), 15);
   nir_jump(&b, nir_jump_return); nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
static nir_def *uniform(nir_builder *b, unsigned components, unsigned base, nir_def *index, unsigned range)
{
   return nir_load_uniform(b, components, 32, index, .base = base, .range = range, .dest_type = nir_type_uint32);
}
static nir_def *sample(nir_builder *b, unsigned slot)
{
   nir_variable *sampler = nir_variable_create(b->shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT), "image");
   sampler->data.binding = slot;
   nir_tex_instr *t = nir_tex_instr_create(b->shader, 2);
   t->op = nir_texop_txl; t->sampler_dim = GLSL_SAMPLER_DIM_2D; t->coord_components = 2;
   t->texture_index = t->sampler_index = slot; t->dest_type = nir_type_float32;
   t->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_imm_vec2(b, .25, .75));
   t->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(b, 0));
   nir_def_init(&t->instr, &t->def, 4, 32); nir_builder_instr_insert(b, &t->instr);
   return &t->def;
}
/* mode0: all n sparse scalars live, source[4*i], XOR reduction.
 * mode1: index source[0], vector[40..43], overlapping vec2[40..41],
 *        dynamic scalar source[80+4*index] for all four possible indices.
 *        Output before texture XOR: [selected, v.x, v.y^v.z, v.w^v.y].
 * mode2: source[400] proves source extent is not the physical map capacity. */
static nir_shader *fragment(unsigned mode, unsigned n, unsigned textures)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pco_nir_options(), "packed_cb0_fragment");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_uvec4_type(), "result");
   out->data.location = FRAG_RESULT_DATA0;
   nir_def *value;
   if (mode == 1) {
      nir_def *index = uniform(&b, 1, 0, nir_imm_int(&b, 0), 1);
      nir_def *v = uniform(&b, 4, 10, nir_imm_int(&b, 0), 1);
      nir_def *duplicate = uniform(&b, 2, 10, nir_imm_int(&b, 0), 1);
      nir_def *selected = uniform(&b, 1, 20, index, 4);
      value = nir_vec4(&b, selected, nir_channel(&b, duplicate, 0),
         nir_ixor(&b, nir_channel(&b, v, 1), nir_channel(&b, v, 2)),
         nir_ixor(&b, nir_channel(&b, v, 3), nir_channel(&b, duplicate, 1)));
   } else {
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned i = 0; i < n; ++i)
         sum = nir_ixor(&b, sum, uniform(&b, 1, mode == 2 ? 100 : i, nir_imm_int(&b, 0), 1));
      value = nir_vec4(&b, sum, sum, sum, sum);
   }
   for (unsigned i = 0; i < textures; ++i)
      value = nir_ixor(&b, value, sample(&b, i));
   nir_store_var(&b, out, value, 15);
   nir_jump(&b, nir_jump_return); nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
static void save(unsigned id, const char *name, const void *data, size_t size)
{
   const char *dir = getenv("PVRGPU_PACKED_FIXTURE_DIR");
   if (!dir) return;
   char path[4096];
   check(snprintf(path, sizeof(path), "%s/packed-%u.%s", dir, id, name) < sizeof(path), "artifact path");
   FILE *f = fopen(path, "wb"); check(f != NULL, "artifact open");
   check(fwrite(data, 1, size, f) == size && fclose(f) == 0, "artifact save");
}
static void compile_case(struct pvrgpu_pco_compiler *compiler, unsigned id,
                         unsigned mode, unsigned n, unsigned bound,
                         unsigned textures, unsigned expected_count, bool packed, bool success)
{
   nir_shader *vs = vertex(), *fs = fragment(mode, n, textures);
   char *before = nir_shader_as_str(fs, NULL), *vb = nir_shader_as_str(vs, NULL);
   const struct shader_info info = fs->info;
   struct pvrgpu_pco_graphics_binary binary = {0};
   enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   char error[2048] = {0};
   bool ok = pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format, false, false,
      1, 0, bound, 1, textures, &binary, error, sizeof(error));
   check(ok == success, error[0] ? error : "compile result");
   char *after = nir_shader_as_str(fs, NULL), *va = nir_shader_as_str(vs, NULL);
   check(!strcmp(before, after) && !strcmp(vb, va) && !memcmp(&info, &fs->info, sizeof(info)), "caller NIR unchanged");
   if (success) {
      const struct pvrgpu_pco_uniform_word_map *m = &binary.fragment.cb0_word_map;
      const struct pvrgpu_pco_stage_abi *a = &binary.fragment.abi;
      check(a->shareds == textures * 20 + expected_count && a->shareds <= 256, "exact physical shared count");
      check(a->push_constant_start == textures * 20 && a->push_constant_count == expected_count, "exact push suffix");
      check(m->count == (packed ? expected_count : 0) && m->source_dwords == (packed ? bound : 0), "map count/source extent");
      if (packed) {
         const unsigned dynamic_words[] = {0,40,41,42,43,80,84,88,92};
         for (unsigned i = 0; i < m->count; ++i)
            check(m->source_words[i] == (mode == 1 ? dynamic_words[i] : mode == 2 ? 400 : i * 4), "sorted complete source-word union");
      }
      for (unsigned i = m->count; i < 256; ++i) check(m->source_words[i] == 0, "canonical map tail");
      check(binary.vertex.cb0_word_map.count == 0 && binary.vertex.cb0_word_map.source_dwords == 0, "VS legacy map");
      save(id, "vs.bin", binary.vertex.data, binary.vertex.size);
      save(id, "fs.bin", binary.fragment.data, binary.fragment.size);
      save(id, "vs.abi", &binary.vertex.abi, sizeof(binary.vertex.abi));
      save(id, "fs.abi", &binary.fragment.abi, sizeof(binary.fragment.abi));
      save(id, "map", m, sizeof(*m));
      printf("packed-%u compiled VS=%zu FS=%zu TEMP=%u SH=%u CF=%u push=%u/%u map=%u original=%u\n", id,
         binary.vertex.size, binary.fragment.size, a->temps,a->shareds,a->coefficients,a->push_constant_start,a->push_constant_count,m->count,m->source_dwords);
   } else {
      check(error[0] && !binary.vertex.data && !binary.fragment.data && !binary.vertex.cb0_word_map.count && !binary.fragment.cb0_word_map.count,
         "failure leaves empty bytes/maps and diagnostic");
      printf("packed-%u rejected %s\n", id, error);
   }
   pvrgpu_pco_graphics_binary_finish(&binary);
   const struct pvrgpu_pco_graphics_binary zero = {0};
   check(!memcmp(&binary, &zero, sizeof(binary)), "finish clears bytes and maps");
   ralloc_free(before); ralloc_free(after); ralloc_free(vb); ralloc_free(va); ralloc_free(vs); ralloc_free(fs);
}
static void direct_preflight(void)
{
   nir_shader *fs = fragment(1, 0, 12);
   struct pvrgpu_pco_uniform_word_map map = {0}; char error[512];
   check(pvrgpu_color_uniform_word_map(fs, 128, 9, &map, error, sizeof(error)), "exact nine-word overlapping/dynamic proof");
   check(map.count == 9 && map.source_words[8] == 92, "every dynamic candidate retained");
   struct pvrgpu_pco_uniform_word_map sentinel; memset(&sentinel, 0x5a, sizeof(sentinel));
   const struct pvrgpu_pco_uniform_word_map original = sentinel;
   check(!pvrgpu_color_uniform_word_map(fs, 128, 8, &sentinel, error, sizeof(error)), "one word over budget rejects");
   check(!memcmp(&sentinel, &original, sizeof(sentinel)), "failed collection publishes no partial map");
   check(!pvrgpu_color_uniform_word_map(fs, 92, 256, &sentinel, error, sizeof(error)), "full indirect range bound checked");
   check(!memcmp(&sentinel, &original, sizeof(sentinel)), "out of bound preflight is atomic");
   ralloc_free(fs);
}
static void unsupported_ubo_dereference(struct pvrgpu_pco_compiler *compiler)
{
   for (unsigned fragment_stage = 0; fragment_stage < 2; ++fragment_stage) {
      for (unsigned pressure = 0; pressure < 2; ++pressure) {
         nir_shader *vs = vertex(), *fs = fragment(0, 1, pressure ? 12 : 0);
         nir_shader *s = fragment_stage ? fs : vs;
         nir_function_impl *impl = nir_shader_get_entrypoint(s);
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         nir_variable *ubo = nir_variable_create(s, nir_var_mem_ubo, glsl_vec4_type(), "unlowered_block");
         nir_def *value = nir_load_var(&b, ubo);
         nir_variable *out = NULL;
         nir_foreach_shader_out_variable(v, s) { out = v; break; }
         check(out != NULL, "negative output exists");
         /* Replace the real export, so this UBO dereference is not dead. */
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic && nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_deref)
                  nir_src_rewrite(&nir_instr_as_intrinsic(instr)->src[1], value);
            }
         }
         s->info.num_ubos = 1;
         char *before = nir_shader_as_str(s, NULL), error[512];
         struct pvrgpu_pco_graphics_binary result = {0};
         const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         check(!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format, false, false, 1,
            pressure ? 96 : 4, pressure ? 128 : 4, 1, pressure ? 12 : 0, &result, error, sizeof(error)),
            "unlowered UBO rejected before PCO regardless of pressure");
         check(strstr(error, "requires lowered UBO accesses") != NULL, "specific unlowered UBO diagnostic");
         check(!result.vertex.data && !result.fragment.data && !result.vertex.cb0_word_map.count && !result.fragment.cb0_word_map.count,
            "unlowered UBO rejection publishes no executable or map");
         char *after = nir_shader_as_str(s, NULL); check(!strcmp(before, after), "unlowered UBO caller unchanged");
         ralloc_free(before); ralloc_free(after); ralloc_free(vs); ralloc_free(fs);
      }
   }
}
int main(void)
{
   char error[1024];
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   check(compiler != NULL, error); direct_preflight(); unsupported_ubo_dereference(compiler);
   compile_case(compiler, 900, 1, 0, 128, 12, 9, true, true);
   compile_case(compiler, 901, 0, 15, 64, 12, 15, true, true); /* SH255 */
   compile_case(compiler, 902, 0, 16, 64, 12, 16, true, true); /* SH256 */
   compile_case(compiler, 903, 0, 17, 68, 12, 0, false, false); /* SH257 */
   compile_case(compiler, 904, 0, 4, 16, 12, 16, false, true); /* legacy */
   compile_case(compiler, 905, 2, 1, 512, 12, 1, true, true); /* high source extent */
   compile_case(compiler, 906, 1, 0, 92, 12, 0, false, false); /* dynamic OOB */
   pvrgpu_pco_compiler_destroy(compiler);
   printf("packed_uniforms: PASS %u checks\n", checks);
   return 0;
}
