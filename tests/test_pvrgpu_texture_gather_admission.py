"""Bounded gather admission against real pinned Mesa NIR types and builders.

Opt in with PVRGPU_MESA_PVRGPU_BUILD_DIR; no shared Mesa rebuild or writes.
This tests the production admission helper, not texture execution or pixels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

try:
    from .test_pvrgpu_vertex_system_values import function_source
except ImportError:
    from test_pvrgpu_vertex_system_values import function_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/gallium/drivers/pvrgpu/pvrgpu_pco.c"
HELPER = "pvrgpu_native_fragment_gather"
RESOLVER = "pvrgpu_static_texture_slots"

FIXTURE = r'''
#include "nir_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* PRODUCTION_HELPER */
static unsigned checks;
static void check(bool value) {
   ++checks;
   if (!value) { fprintf(stderr, "gather admission check %u failed\n", checks); exit(1); }
}
static const nir_shader_compiler_options options = {0};
int main(void) {
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options, "gather-admission");
   nir_tex_instr *t = nir_tex_instr_create(b.shader, 1);
   t->op = nir_texop_tg4;
   t->sampler_dim = GLSL_SAMPLER_DIM_2D;
   t->dest_type = nir_type_float32;
   t->def.num_components = 4;
   t->def.bit_size = 32;
   t->coord_components = 2;
   t->src[0].src_type = nir_tex_src_coord;
   t->src[0].src = nir_src_for_ssa(nir_imm_vec2(&b, .25f, .75f));
   check(pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 1));
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_VERTEX, 1));
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_GEOMETRY, 1));
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_COMPUTE, 1));
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 0));
#define REJECT(field, value) do { \
   __typeof__(t->field + 0) saved = t->field; t->field = value; \
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 1)); \
   t->field = saved; \
} while (0)
   REJECT(op, nir_texop_tex);
   REJECT(sampler_dim, GLSL_SAMPLER_DIM_3D);
   REJECT(sampler_dim, GLSL_SAMPLER_DIM_CUBE);
   REJECT(is_array, true); REJECT(is_shadow, true); REJECT(is_sparse, true);
   REJECT(component, 1); REJECT(component, 2); REJECT(component, 3);
   REJECT(dest_type, nir_type_int32); REJECT(dest_type, nir_type_float16);
   REJECT(def.bit_size, 16); REJECT(def.num_components, 1);
   REJECT(coord_components, 3); REJECT(is_gather_implicit_lod, 1);
   REJECT(array_is_lowered_cube, 1); REJECT(texture_non_uniform, true);
   REJECT(sampler_non_uniform, true); REJECT(embedded_sampler, true);
   REJECT(texture_index, 1); REJECT(sampler_index, 1);
   REJECT(num_srcs, 0);
   for (unsigned tap = 0; tap < 4; ++tap)
      for (unsigned axis = 0; axis < 2; ++axis) {
         REJECT(tg4_offsets[tap][axis], t->tg4_offsets[tap][axis] + 1);
         REJECT(tg4_offsets[tap][axis], t->tg4_offsets[tap][axis] - 1);
      }
   nir_tex_src saved = t->src[0];
   for (unsigned src = 0; src < nir_num_tex_src_types; ++src) {
      if (src == nir_tex_src_coord) continue;
      t->src[0].src_type = src;
      check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 1));
   }
   t->src[0] = saved;
   t->src[0].src.ssa = NULL;
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 1));
   t->src[0].src = nir_src_for_ssa(nir_imm_float(&b, .5f));
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 1));
   t->src[0] = saved;
   nir_tex_src sources[3] = {saved, saved, saved};
   t->src = sources; t->num_srcs = 2;
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 1));
   nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT), "s");
   nir_deref_instr *deref = nir_build_deref_var(&b, sampler);
   sources[1].src_type = nir_tex_src_texture_deref;
   sources[1].src = nir_src_for_ssa(&deref->def);
   sources[2] = sources[1]; sources[2].src_type = nir_tex_src_sampler_deref;
   t->num_srcs = 3;
   check(pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 1));
   sampler->data.binding = 2;
   check(pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 3));
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 2));
   sampler->data.descriptor_set = 1;
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 3));
   sampler->data.descriptor_set = 0;
   t->num_srcs = 2;
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 3));
   t->num_srcs = 3; sources[2].src_type = nir_tex_src_texture_deref;
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 3));
   sources[1].src_type = sources[2].src_type = nir_tex_src_sampler_deref;
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 3));
   nir_deref_instr *cast = nir_build_deref_cast(&b, &deref->def, nir_var_uniform,
                                               sampler->type, 0);
   sources[1].src_type = nir_tex_src_texture_deref;
   sources[1].src = nir_src_for_ssa(&cast->def);
   check(!pvrgpu_native_fragment_gather(t, MESA_SHADER_FRAGMENT, 3));
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   printf("real NIR gather admission: %u checks PASS\n", checks);
   return 0;
}
'''


class TextureGatherAdmissionTests(unittest.TestCase):
    def test_validator_uses_bounded_helper_without_rewriting_gather(self):
        source = SOURCE.read_text()
        validator = function_source(source, "pvrgpu_validate_color_primitive_nir")
        self.assertIn(HELPER + "(", validator)
        self.assertIn("!size_query && !gather", validator)
        helper = function_source(source, HELPER)
        self.assertIn(RESOLVER + "(", helper)
        self.assertNotIn("nir_instr_remove", helper)
        self.assertNotIn("nir_src_rewrite", helper)

    def test_real_mesa_nir(self):
        configured = os.environ.get("PVRGPU_MESA_PVRGPU_BUILD_DIR")
        if not configured:
            self.skipTest("set PVRGPU_MESA_PVRGPU_BUILD_DIR for real NIR")
        build = Path(configured).resolve()
        self.assertFalse(build == ROOT or ROOT in build.parents)
        database = json.loads((build / "compile_commands.json").read_text())
        entry = next(e for e in database if Path(e["file"]).name == "nir_opt_shrink_vectors.c")
        command = entry.get("arguments") or shlex.split(entry["command"])
        flags, args = [], iter(command[1:])
        for arg in args:
            if arg in ("-o", "-MF", "-MQ", "-MT"):
                next(args)
            elif arg not in ("-c", "-MD", "-MMD", "-DNDEBUG", entry["file"]):
                flags.append(arg)
        flags += ["-O0", "-g", "-UNDEBUG"]
        sanitizer = (["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
                     if os.environ.get("PVRGPU_NIR_TEST_SANITIZE") == "1" else [])
        with tempfile.TemporaryDirectory(prefix="pvrgpu-gather-gate-") as temporary:
            output = Path(os.environ.get("PVRGPU_NIR_TEST_OUT", temporary)).resolve()
            self.assertFalse(output == ROOT or ROOT in output.parents)
            output.mkdir(parents=True, exist_ok=True)
            generated, obj, executable = (output / n for n in ("gate.c", "gate.o", "gate"))
            source = SOURCE.read_text()
            helpers = "\n\n".join(function_source(source, name)
                                   for name in (RESOLVER, HELPER))
            generated.write_text(FIXTURE.replace("/* PRODUCTION_HELPER */", helpers))
            compile_command = [command[0], *flags, *sanitizer, "-c", str(generated), "-o", str(obj)]
            libraries = ["src/compiler/nir/libnir.a", "src/compiler/libcompiler.a",
                         "src/util/libmesa_util.a", "src/util/blake3/libblake3.a",
                         "src/c11/impl/libmesa_util_c11.a"]
            libs = shlex.split(subprocess.check_output(
                ["pkg-config", "--libs", "libzstd", "zlib", "expat"], text=True))
            cxx = str(Path(command[0]).with_name("clang++"))
            link_command = [cxx, str(obj), *libraries, *libs, *sanitizer, "-lpthread", "-o", str(executable)]
            for label, cmd, cwd in (("compile", compile_command, entry["directory"]),
                                    ("link", link_command, build)):
                result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
                (output / (label + ".log")).write_text(result.stdout + result.stderr)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = os.environ.copy()
            if sanitizer:
                env.update(ASAN_OPTIONS="detect_leaks=0:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
            result = subprocess.run([str(executable)], capture_output=True, text=True, env=env, timeout=15)
            (output / "test.log").write_text(result.stdout + result.stderr)
            (output / "commands.json").write_text(json.dumps({"compile": compile_command, "link": link_command}, indent=2))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("checks PASS", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
