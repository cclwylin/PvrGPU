"""Clone-local fragment-output specialization: source contract + real Mesa NIR.

The real-NIR test is opt-in with PVRGPU_MESA_PVRGPU_BUILD_DIR pointing at an
already built Mesa tree. It reads its compiler flags and static libraries, but
never invokes a Mesa build or edits that tree. PVRGPU_NIR_TEST_OUT optionally
retains generated sources, executable and logs outside this repository.
PVRGPU_NIR_TEST_SANITIZE=1 instruments the fixture and private NIR core/validator
objects; existing Mesa support archives are not rebuilt or instrumented.
This is an IR/helper regression, not a substitute for live rendering tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

try:
    from .test_pvrgpu_vertex_system_values import function_source
except ImportError:  # Direct execution and unittest discovery from tests/.
    from test_pvrgpu_vertex_system_values import function_source


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/gallium/drivers/pvrgpu/pvrgpu_pco.c"
HELPER = "pvrgpu_prune_unbound_fragment_outputs"


FIXTURE = r'''
#include "nir_builder.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool pvrgpu_pco_fail(char *error, size_t size, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   vsnprintf(error, size, format, args);
   va_end(args);
   return false;
}

/* PRODUCTION_HELPER */

static unsigned checks, cases;
static const nir_shader_compiler_options options = {0};

static void check(bool condition, const char *message)
{
   ++checks;
   if (!condition) {
      fprintf(stderr, "FAIL case %u: %s\n", cases, message);
      exit(1);
   }
}

static nir_variable *output(nir_builder *b, const char *name, unsigned location,
                            const struct glsl_type *type)
{
   nir_variable *var = nir_variable_create(b->shader, nir_var_shader_out, type, name);
   var->data.location = location;
   var->data.driver_location = b->shader->num_outputs++;
   return var;
}

static nir_variable *named(nir_shader *shader, const char *name)
{
   nir_foreach_variable_in_shader(var, shader)
      if (var->name && strcmp(var->name, name) == 0)
         return var;
   return NULL;
}

static unsigned instructions(nir_shader *shader, int intrinsic)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (intrinsic < 0 || (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == (nir_intrinsic_op)intrinsic))
               ++count;
         }
      }
   }
   return count;
}

static nir_shader *make_matrix(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options,
                                                  "output specialization matrix");
   nir_variable *colors[8];
   for (unsigned slot = 0; slot < 8; ++slot) {
      char name[16];
      snprintf(name, sizeof(name), "color%u", slot);
      colors[slot] = output(&b, name, FRAG_RESULT_DATA0 + slot, glsl_vec4_type());
      const float value = (slot + 1) * .125f;
      nir_store_var(&b, colors[slot], nir_imm_vec4(&b, value, value, value, value), 15);
   }
   /* The ignored output is read back and feeds a bound output, depth and a
    * memory side effect. Deleting its store would change those observations. */
   nir_def *read_back = nir_load_var(&b, colors[7]);
   nir_store_var(&b, colors[0], read_back, 15);
   nir_variable *depth = output(&b, "depth", FRAG_RESULT_DEPTH, glsl_float_type());
   nir_store_var(&b, depth, nir_channel(&b, read_back, 0), 1);
   nir_demote_if(&b, nir_feq_imm(&b, nir_channel(&b, read_back, 1), .5f));
   nir_def *integer = nir_f2u32(&b, nir_channel(&b, read_back, 2));
   nir_store_ssbo(&b, integer, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
                  .write_mask = 1, .align_mul = 4);
   nir_variable_create(b.shader, nir_var_uniform,
                       glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
                       "sampler_binding");
   nir_variable_create(b.shader, nir_var_image,
                       glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
                       "image_binding");
   nir_image_atomic(&b, 32, nir_imm_int(&b, 0), nir_imm_ivec4(&b, 0, 0, 0, 0),
                    nir_imm_int(&b, 0), integer, .image_dim = GLSL_SAMPLER_DIM_2D,
                    .format = PIPE_FORMAT_R32_UINT, .atomic_op = nir_atomic_op_iadd);
   b.shader->info.num_ubos = 2;
   b.shader->info.num_ssbos = 3;
   nir_shader_gather_info(b.shader, b.impl);
   nir_validate_shader(b.shader, "original output matrix");
   return b.shader;
}

static void verify_resources_and_effects(nir_shader *shader)
{
   check(shader->info.num_textures == 1 && shader->info.num_images == 1 &&
         shader->info.num_ubos == 2 && shader->info.num_ssbos == 3,
         "all original descriptor counts are preserved");
   check(shader->info.fs.uses_discard && shader->info.writes_memory,
         "discard and memory-effect metadata preserved");
   check(instructions(shader, nir_intrinsic_demote_if) == 1 &&
         instructions(shader, nir_intrinsic_store_ssbo) == 1 &&
         instructions(shader, nir_intrinsic_image_atomic) == 1,
         "discard, SSBO store and image atomic instructions retained");
   check(named(shader, "depth")->data.mode == nir_var_shader_out,
         "depth remains an output");
}

static void run_matrix(void)
{
   nir_shader *original = make_matrix();
   char *before = nir_shader_as_str(original, NULL);
   const unsigned original_instructions = instructions(original, -1);
   const uint64_t depth_bit = BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   /* Rebinding is always a fresh clone of the original, never the previously
    * specialized shader. Also cover the full eight-slot helper domain. */
   const unsigned counts[] = {1, 4, 1, 2, 3, 5, 6, 7, 8};
   for (unsigned index = 0; index < ARRAY_SIZE(counts); ++index) {
      ++cases;
      const unsigned count = counts[index];
      nir_shader *clone = nir_shader_clone(NULL, original);
      char error[256] = {0};
      check(pvrgpu_prune_unbound_fragment_outputs(clone, count, error, sizeof(error)), error);
      check(clone->info.outputs_written == (depth_bit | BITFIELD64_RANGE(FRAG_RESULT_DATA0, count)),
            "exact retained color/depth output mask");
      check(instructions(clone, -1) == original_instructions,
            "helper neither deletes nor fabricates any instruction");
      for (unsigned slot = 0; slot < 8; ++slot) {
         char name[16];
         snprintf(name, sizeof(name), "color%u", slot);
         check(named(clone, name)->data.mode ==
               (slot < count ? nir_var_shader_out : nir_var_shader_temp),
               "only whole ignored variables become temporary");
      }
      verify_resources_and_effects(clone);
      nir_validate_shader(clone, "specialized output matrix");
      char *after = nir_shader_as_str(original, NULL);
      check(strcmp(before, after) == 0, "original NIR survives clone specialization and rebind");
      ralloc_free(after);
      ralloc_free(clone);
   }
   /* Real generic NIR temp/SSA cleanup must retain the value read back from
    * color7 after its external export is removed. This is not a hand-written
    * shader evaluator or a substitute shader. */
   ++cases;
   nir_shader *clone = nir_shader_clone(NULL, original);
   char error[256] = {0};
   check(pvrgpu_prune_unbound_fragment_outputs(clone, 1, error, sizeof(error)), error);
   nir_lower_global_vars_to_local(clone);
   nir_lower_vars_to_ssa(clone);
   nir_opt_constant_folding(clone);
   nir_opt_copy_prop(clone);
   nir_opt_dce(clone);
   bool found_depth = false;
   nir_foreach_function_impl(impl, clone) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref) continue;
            nir_variable *var = nir_intrinsic_get_var(intr, 0);
            if (var && var->data.mode == nir_var_shader_out &&
                var->data.location == FRAG_RESULT_DEPTH) {
               check(nir_src_is_const(intr->src[1]) &&
                     nir_src_as_float(intr->src[1]) == 1.f,
                     "depth still reads the original ignored-output value after real SSA cleanup");
               found_depth = true;
            }
         }
      }
   }
   check(found_depth && instructions(clone, nir_intrinsic_store_ssbo) == 1 &&
         instructions(clone, nir_intrinsic_image_atomic) == 1,
         "real cleanup preserves depth and observable memory effects");
   nir_validate_shader(clone, "local-read cleanup");
   ralloc_free(clone);
   ralloc_free(before);
   ralloc_free(original);
}

static void run_array(unsigned location, unsigned length, unsigned targets,
                       bool dynamic, bool success)
{
   ++cases;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options, "array extent");
   nir_variable *array = output(&b, "array", location,
                                glsl_array_type(glsl_vec4_type(), length, 0));
   const unsigned constant_index = location + length > FRAG_RESULT_DATA0 + 8 ? 0 : length - 1;
   nir_def *index = dynamic ? nir_iand_imm(&b, nir_load_sample_id(&b), 1) : nir_imm_int(&b, constant_index);
   nir_deref_instr *element = nir_build_deref_array(&b, nir_build_deref_var(&b, array), index);
   nir_store_deref(&b, element, nir_imm_vec4(&b, 1, 2, 3, 4), 15);
   nir_shader_gather_info(b.shader, b.impl);
   char error[256] = {0};
   check(pvrgpu_prune_unbound_fragment_outputs(b.shader, targets, error, sizeof(error)) == success,
         "whole-unbound array allowed; mixed active/unbound array rejected");
   if (success) {
      check(array->data.mode == nir_var_shader_temp && b.shader->info.outputs_written == 0,
            "whole ignored array preserves local storage, including dynamic indexing");
      nir_validate_shader(b.shader, "whole ignored array");
   } else {
      check(array->data.mode == nir_var_shader_out && error[0],
            "preflight failure leaves original variable mode unchanged");
   }
   ralloc_free(b.shader);
}

static void run_rejections(void)
{
   for (unsigned kind = 0; kind < 4; ++kind) {
      ++cases;
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options, "fail closed");
      nir_variable *var = output(&b, "unbound", FRAG_RESULT_DATA7, glsl_vec4_type());
      if (kind == 0) var->data.index = 1;
      nir_def *value = nir_imm_vec4(&b, 1, 2, 3, 4);
      nir_deref_instr *deref = nir_build_deref_var(&b, var);
      if (kind == 2)
         deref = nir_build_deref_cast(&b, &deref->def, nir_var_shader_out, glsl_vec4_type(), 0);
      nir_store_deref(&b, deref, value, 15);
      if (kind == 1)
         nir_store_output(&b, value, nir_imm_int(&b, 0), .base = 0, .write_mask = 15,
                          .src_type = nir_type_float32,
                          .io_semantics = {.location = FRAG_RESULT_DATA7, .num_slots = 1});
      nir_shader_gather_info(b.shader, b.impl);
      if (kind == 3)
         b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA6);
      char error[256] = {0};
      check(!pvrgpu_prune_unbound_fragment_outputs(b.shader, 1, error, sizeof(error)),
            "dual-source, mixed lowered, aliased, or undeclared output is rejected");
      check(var->data.mode == nir_var_shader_out && error[0], "rejection is before mutation");
      ralloc_free(b.shader);
   }
   ++cases;
   nir_shader *shader = make_matrix();
   char error[256] = {0};
   check(!pvrgpu_prune_unbound_fragment_outputs(shader, 0, error, sizeof(error)), "zero transport count rejected");
   check(!pvrgpu_prune_unbound_fragment_outputs(shader, 9, error, sizeof(error)), "oversize transport count rejected");
   ralloc_free(shader);
}

static void run_legacy_noop(void)
{
   ++cases;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options, "legacy untouched");
   nir_variable *legacy = output(&b, "legacy", FRAG_RESULT_COLOR, glsl_vec4_type());
   nir_store_var(&b, legacy, nir_imm_vec4(&b, 1, 2, 3, 4), 15);
   nir_shader_gather_info(b.shader, b.impl);
   char *before = nir_shader_as_str(b.shader, NULL);
   char error[256] = {0};
   check(pvrgpu_prune_unbound_fragment_outputs(b.shader, 1, error, sizeof(error)), error);
   char *after = nir_shader_as_str(b.shader, NULL);
   check(strcmp(before, after) == 0, "legacy FragColor has exactly unchanged NIR");
   ralloc_free(before); ralloc_free(after); ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   run_matrix();
   run_array(FRAG_RESULT_DATA5, 3, 4, false, true);
   run_array(FRAG_RESULT_DATA5, 3, 4, true, true);
   run_array(FRAG_RESULT_DATA0, 8, 1, false, false);
   run_array(FRAG_RESULT_DATA0, 8, 4, true, false);
   run_array(FRAG_RESULT_DATA7, 2, 1, false, false);
   run_rejections();
   run_legacy_noop();
   glsl_type_singleton_decref();
   printf("real NIR fragment output pruning: %u cases, %u checks PASS\n", cases, checks);
   return 0;
}
'''


class FragmentOutputPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.helper = function_source(cls.source, HELPER)

    def test_both_paths_specialize_only_the_clone_before_validation(self) -> None:
        for name in ("pvrgpu_pco_compile_color_triangle", "pvrgpu_compile_extended_geometry_pipeline"):
            body = function_source(self.source, name)
            with self.subTest(path=name):
                calls = re.findall(HELPER + r"\([^;{}]*?\)", body)
                self.assertEqual(len(calls), 1)
                self.assertRegex(calls[0], r"\(fs,\s*render_target_count,")
                self.assertLess(body.index("nir_shader_clone"), body.index(HELPER))
                self.assertLess(body.index(HELPER), body.index("pvrgpu_validate_color_primitive_nir"))
                self.assertLess(body.index(HELPER), body.index("pvrgpu_pco_preprocess_nir"))

    def test_helper_does_not_delete_producers_or_change_resource_layout(self) -> None:
        for forbidden in ("nir_instr_remove", "nir_def_replace", "nir_opt_dce", "nir_remove_dead_variables"):
            self.assertNotIn(forbidden, self.helper)
        for field in ("num_textures", "num_images", "num_ubos", "num_ssbos"):
            self.assertRegex(self.helper, r"fs->info\." + field + r"\s*!=")
        self.assertIn("nir_var_shader_temp", self.helper)
        self.assertIn("nir_fixup_deref_modes(fs)", self.helper)

    def test_real_mesa_nir(self) -> None:
        value = os.environ.get("PVRGPU_MESA_PVRGPU_BUILD_DIR")
        if not value:
            self.skipTest("set PVRGPU_MESA_PVRGPU_BUILD_DIR for the real-NIR fixture")
        build = Path(value).resolve()
        self.assertFalse(build == ROOT or ROOT in build.parents, "use an external Mesa build")
        database = json.loads((build / "compile_commands.json").read_text())
        entry = next(item for item in database if Path(item["file"]).name == "nir_opt_shrink_vectors.c")
        command = entry.get("arguments") or shlex.split(entry["command"])
        compiler, arguments = command[0], iter(command[1:])
        flags = []
        for arg in arguments:
            if arg in ("-o", "-MF", "-MQ", "-MT"):
                next(arguments)
            elif arg not in ("-c", "-MD", "-MMD", "-DNDEBUG", entry["file"]):
                flags.append(arg)
        flags += ["-O0", "-g", "-UNDEBUG"]
        sanitizer_flags = (["-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            "-fno-omit-frame-pointer"]
                           if os.environ.get("PVRGPU_NIR_TEST_SANITIZE") == "1" else [])
        flags += sanitizer_flags
        with tempfile.TemporaryDirectory(prefix="pvrgpu-fragment-output-") as temporary:
            output = Path(os.environ.get("PVRGPU_NIR_TEST_OUT", temporary)).resolve()
            self.assertFalse(output == ROOT or ROOT in output.parents, "use external test artifacts")
            output.mkdir(parents=True, exist_ok=True)
            source = output / "fragment-output-pruning.c"
            source.write_text(FIXTURE.replace("/* PRODUCTION_HELPER */", self.helper))
            obj = output / "fragment-output-pruning.o"
            executable = output / "fragment-output-pruning"
            compile_command = [compiler, *flags, "-c", str(source), "-o", str(obj)]
            result = subprocess.run(compile_command, cwd=entry["directory"], capture_output=True, text=True)
            (output / "compile.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stderr)
            # Release Mesa archives omit nir_validate_shader. Recompile only
            # the validator and its core support into this private executable
            # with assertions enabled, as the existing NIR shrink test does.
            support_objects, support_commands = [], []
            for name in ("nir.c", "nir_validate.c"):
                support = next(item for item in database if Path(item["file"]).name == name)
                support_source = (Path(support["directory"]) / support["file"]).resolve()
                support_object = output / (name + ".o")
                support_command = [compiler, *flags, "-c", str(support_source), "-o", str(support_object)]
                result = subprocess.run(support_command, cwd=entry["directory"], capture_output=True, text=True)
                (output / ("compile-" + name + ".log")).write_text(result.stdout + result.stderr)
                self.assertEqual(result.returncode, 0, result.stderr)
                support_objects.append(str(support_object))
                support_commands.append(support_command)
            libraries = ["src/compiler/nir/libnir.a", "src/compiler/libcompiler.a",
                         "src/util/libmesa_util.a", "src/util/blake3/libblake3.a",
                         "src/c11/impl/libmesa_util_c11.a"]
            cxx = str(Path(compiler).with_name("clang++")) if "clang" in Path(compiler).name else "c++"
            dependency_flags = subprocess.check_output(
                ["pkg-config", "--libs", "libzstd", "zlib", "expat"], text=True)
            link_command = [cxx, str(obj), *support_objects, *libraries,
                            *shlex.split(dependency_flags), *sanitizer_flags,
                            "-lpthread", "-o", str(executable)]
            result = subprocess.run(link_command, cwd=build, capture_output=True, text=True)
            (output / "link.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stderr)
            environment = os.environ.copy()
            if sanitizer_flags:
                environment.update(ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
            result = subprocess.run([str(executable)], capture_output=True, text=True, env=environment)
            (output / "test.log").write_text(result.stdout + result.stderr)
            (output / "commands.json").write_text(json.dumps({"compile": compile_command,
                "support": support_commands, "link": link_command}, indent=2))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("checks PASS", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
