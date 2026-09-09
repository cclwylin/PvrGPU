"""Vertex-system-value packing/ABI regression, not a rendering test.

Compile the production allocation helper and its actual two call expressions
against minimal type shims. This checks its integer slot arithmetic without
building Mesa, changing a shader, or substituting rendered output.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER_ROOT = PROJECT_ROOT / "src" / "gallium" / "drivers" / "pvrgpu"


def function_source(source: str, name: str) -> str:
    signature = re.search(
        r"(?m)^(?:static\s+)?bool\s+" + re.escape(name)
        + r"\s*\([^;{}]*\)\s*\{", source
    )
    if signature is None:
        raise AssertionError(f"missing C function {name}")
    # Ignore braces in comments and string/character literals.
    tokens = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|'
                        r"'(?:\\.|[^'\\])*'|[{}]", re.DOTALL)
    depth = 0
    for token in tokens.finditer(source, signature.end() - 1):
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                return source[signature.start():token.end()]
    raise AssertionError(f"unterminated C function {name}")


class VertexSystemValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pco = (DRIVER_ROOT / "pvrgpu_pco.c").read_text(encoding="utf-8")
        cls.context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        header = (DRIVER_ROOT / "pvrgpu_pco.h").read_text(encoding="utf-8")
        match = re.search(r"#define PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES\s+(\d+)u?", header)
        if match is None:
            raise AssertionError("missing public attribute slot bound")
        cls.max_attributes = int(match.group(1))
        cls.helper = function_source(cls.pco, "pvrgpu_allocate_vertex_system_values")
        cls.calls = []
        for function, data in (
            ("pvrgpu_compile_extended_geometry_pipeline", "vd"),
            ("pvrgpu_pco_compile_color_triangle", "vertex_data"),
        ):
            body = function_source(cls.pco, function)
            calls = re.findall(r"pvrgpu_allocate_vertex_system_values\([^;{}]*?\)", body)
            if len(calls) != 1:
                raise AssertionError(f"expected one system-value allocation in {function}")
            cls.calls.append((function, data, calls[0]))

    def test_both_compiler_paths_use_original_nir_mask(self) -> None:
        for function, data, call in self.calls:
            with self.subTest(function=function):
                self.assertRegex(call, r"\(\s*vertex_nir\s*,\s*&" + data + r"\s*,")

    def test_packer_keeps_raw_ids_separate_from_rebased_indices(self) -> None:
        body = function_source(self.context, "pvrgpu_record_color_primitive_pco_draw_attempt")
        self.assertIn("const unsigned instance_id_attribute = attribute_count;", body)
        self.assertRegex(body, r"vertex_id_attribute\s*=\s*attribute_count\s*\+\s*"
                              r"\(reads_instance_id\s*\?\s*1u\s*:\s*0u\)")
        for name in ("instance_id_attribute", "vertex_id_attribute"):
            self.assertIn(f"attribute_formats[{name}] = PIPE_FORMAT_R32_UINT;", body)
            self.assertIn(f"attribute_components[{name}] = 1;", body)
        self.assertRegex(body, r"attribute_offsets\[instance_id_attribute\]\]\s*=\s*instance;")
        self.assertRegex(body, r"attribute_offsets\[vertex_id_attribute\]\]\s*=\s*v_idx;")
        self.assertRegex(body, r"info->index_size\s*!=\s*0\s*\?\s*v\s*\+\s*vertex_bias"
                              r"\s*:\s*draw->start\s*\+\s*v")
        self.assertIn("util_format_is_pure_integer(attribute_formats[attribute])", body)

    def test_production_allocator_all_masks_dce_subsets_and_capacity(self) -> None:
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler or shutil.which(compiler[0]) is None:
            self.skipTest("a C compiler is required for the extracted-helper test")
        wrappers = []
        for index, (_, data, call) in enumerate(self.calls):
            wrappers.append(f"""
static bool call_{index}(const nir_shader *vertex_nir, const nir_shader *vs,
                        unsigned attribute_count, pco_data *output) {{
    (void)vs;
    pco_data {data} = {{0}};
    char error[128] = {{0}};
    size_t error_size = sizeof(error);
    bool result = {call};
    *output = {data};
    return result;
}}
""")
        code = r"""
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef enum { SYSTEM_VALUE_INSTANCE_ID, SYSTEM_VALUE_VERTEX_ID } gl_system_value;
typedef struct { unsigned start, count; } pco_range;
typedef struct { struct { uint64_t system_values_read; } info; } nir_shader;
typedef struct { struct { unsigned vtxins; pco_range sys_vals[2]; } common; } pco_data;
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define BITSET_TEST(mask, value) (((mask) >> (value)) & 1U)
static bool pvrgpu_pco_fail(char *error, size_t size, const char *message) {
    snprintf(error, size, "%s", message);
    return false;
}
""" + f"\n#define PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES {self.max_attributes}u\n"
        code += self.helper + "\n" + "\n".join(wrappers) + r"""
int main(void) {
    bool (*calls[])(const nir_shader *, const nir_shader *, unsigned, pco_data *) = {
        call_0, call_1,
    };
    unsigned checks = 0;
    for (unsigned path = 0; path < ARRAY_SIZE(calls); ++path) {
        for (unsigned mask = 0; mask < 4; ++mask) {
            const nir_shader original = {.info = {.system_values_read = mask}};
            for (unsigned live = 0; live < 4; ++live) {
                if (live & ~mask) continue;
                const nir_shader optimized = {.info = {.system_values_read = live}};
                for (unsigned attributes = 0; attributes <= PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES; ++attributes) {
                    pco_data output = {0};
                    unsigned slots = attributes + !!(mask & 1U) + !!(mask & 2U);
                    bool fits = slots <= PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES;
                    bool actual = calls[path](&original, &optimized, attributes, &output);
                    assert(actual == fits);
                    if (fits) {
                        assert(output.common.vtxins == slots * 4);
                        assert(output.common.sys_vals[SYSTEM_VALUE_INSTANCE_ID].count == !!(mask & 1U));
                        assert(output.common.sys_vals[SYSTEM_VALUE_VERTEX_ID].count == !!(mask & 2U));
                        if (mask & 1U) assert(output.common.sys_vals[SYSTEM_VALUE_INSTANCE_ID].start == attributes * 4);
                        if (mask & 2U) assert(output.common.sys_vals[SYSTEM_VALUE_VERTEX_ID].start == (attributes + !!(mask & 1U)) * 4);
                    }
                    ++checks;
                }
            }
        }
    }
    printf("slot/DCE/capacity cases: %u PASS\n", checks);
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="pvrgpu-vertex-system-values-") as directory:
            source = Path(directory) / "allocator.c"
            executable = Path(directory) / "allocator"
            source.write_text(code, encoding="utf-8")
            built = subprocess.run(compiler + ["-std=c11", "-O0", "-UNDEBUG", "-Wall", "-Wextra", "-Werror",
                                               str(source), "-o", str(executable)],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
