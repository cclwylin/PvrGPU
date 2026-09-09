#!/usr/bin/env python3
"""Execute the real recorder entry/format guard with a minimal C state fixture.

This is admission-only evidence, not shader compilation, native submission or
rendering. It needs a C compiler but no Mesa checkout or driver installation.
The later compiler boundary is a checked sentinel, never a prepared renderer.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

DRIVER = Path(__file__).resolve().parents[1] / "src/gallium/drivers/pvrgpu"
SOURCE = DRIVER / "pvrgpu_context.c"
HELPERS_BEGIN = "static bool\npvrgpu_framebuffer_has_mixed_color_formats("
HELPERS_END = "static bool\npvrgpu_read_user_float2_vertex("
ENTRY_BEGIN = "static bool\npvrgpu_record_color_primitive_pco_draw_attempt("
ENTRY_END = "   const bool has_tessellation = ctx->tcs && ctx->tes;"
GUARD = """   if (!pvrgpu_framebuffer_color_transport_is_bounded(ctx)) {
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=entry reason=color_transport_bounds");
      return false;
   }
"""


def extract(source):
    for marker in (HELPERS_BEGIN, HELPERS_END, ENTRY_BEGIN):
        assert source.count(marker) == 1, "ambiguous or missing production function"
    helpers = HELPERS_BEGIN + source.split(HELPERS_BEGIN, 1)[1].split(HELPERS_END, 1)[0]
    rest = source.split(ENTRY_BEGIN, 1)[1]
    assert rest.count(ENTRY_END) == 1, "missing first post-entry operation"
    entry = ENTRY_BEGIN + rest.split(ENTRY_END, 1)[0]
    return helpers, entry


MOCK = r"""
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "pvrgpu_color_formats.h"
#define PVRGPU_MAX_RENDER_TARGETS 4u
enum pipe_format {RGBA8, RGB10, BGR10, R16};
struct pipe_resource {int unused;};
struct pipe_surface {struct pipe_resource *texture; enum pipe_format format;};
struct pipe_framebuffer_state {unsigned nr_cbufs; struct pipe_surface cbufs[8];};
struct pvrgpu_context {struct pipe_framebuffer_state framebuffer;};
struct pipe_draw_info {int unused;};
struct pipe_draw_start_count_bias {int unused;};
static const char *util_format_name(enum pipe_format format) {
  static const char *const names[]={"PIPE_FORMAT_R8G8B8A8_UNORM",
    "PIPE_FORMAT_R10G10B10A2_UNORM", "PIPE_FORMAT_B10G10R10A2_UNORM", "PIPE_FORMAT_R16_UNORM"};
  return names[format];
}
static const char *pvrgpu_command_output_path(void) {return "unit-only-not-written";}
static unsigned compiler_boundary, bounds_errors;
static void pvrgpu_counter_eventf(const char *event, const char *format, ...) {
  (void)event;
  if(strstr(format,"color_transport_bounds")) ++bounds_errors;
}
@HELPERS@
@ENTRY@
   (void)may_retry_payload;
   (void)terminal_failure;
   ++compiler_boundary;
   return true;
}
int main(int argc,char **argv) {
  if(argc!=3) return 2;
  if(strcmp(util_format_name(RGBA8),"PIPE_FORMAT_R8G8B8A8_UNORM")) return 2;
  unsigned count=(unsigned)strtoul(argv[1],NULL,10);
  unsigned mode=(unsigned)strtoul(argv[2],NULL,10);
  if(count>8 || mode>4) return 2;
  struct pipe_resource resource={0};
  struct pvrgpu_context ctx={0};
  ctx.framebuffer.nr_cbufs=count;
  for(unsigned i=0;i<8;++i) {
    ctx.framebuffer.cbufs[i].texture=&resource;
    ctx.framebuffer.cbufs[i].format=mode==4 ? R16 : RGBA8;
  }
  if(mode==1) {ctx.framebuffer.cbufs[1].format=RGB10;ctx.framebuffer.cbufs[2].format=BGR10;}
  if(mode==2) ctx.framebuffer.cbufs[1].format=R16;
  if(mode==3 && count) ctx.framebuffer.cbufs[count-1].texture=NULL;
  bool expected=count<=4 && !(mode==2 && count>=2) && !(mode==3 && count>0);
  const struct pvrgpu_context before=ctx;
  struct pipe_draw_info info={0};struct pipe_draw_start_count_bias draw={0};
  bool terminal=false;
  bool actual=pvrgpu_record_color_primitive_pco_draw_attempt(&ctx,&info,&draw,true,&terminal);
  if(actual!=expected || compiler_boundary!=(expected ? 1u : 0u) ||
     bounds_errors!=(expected ? 0u : 1u) || terminal || memcmp(&before,&ctx,sizeof(ctx))) {
    fprintf(stderr,"entry mismatch: count=%u mode=%u accepted=%u expected=%u boundary=%u errors=%u\n",
            count,mode,actual,expected,compiler_boundary,bounds_errors);
    return 1;
  }
  if(pvrgpu_framebuffer_color_transport_is_bounded(NULL)) return 1;
  printf("count=%u mode=%u entry PASS\n",count,mode);
}
"""


class MixedMrtAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang")
        if not cls.compiler:
            raise unittest.SkipTest("A C11 compiler is required for the actual-entry fixture")
        cls.temp = tempfile.TemporaryDirectory(prefix="pvrgpu-mixed-mrt-entry-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = SOURCE.read_text()
        cls.good = cls.compile("production-entry", cls.source)

    @classmethod
    def compile(cls, name, source):
        helpers, entry = extract(source)
        code = MOCK.replace("@HELPERS@", helpers).replace("@ENTRY@", entry)
        path = cls.root / (name + ".c")
        path.write_text(code)
        binary = cls.root / name
        result = subprocess.run([cls.compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                                 "-I", str(DRIVER), str(path), "-o", str(binary)],
                                text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return binary

    def run_case(self, binary, count, mode):
        return subprocess.run([str(binary), str(count), str(mode)], text=True, capture_output=True)

    def test_real_entry_accepts_only_bounded_transport_grid(self):
        for count in range(9):
            for mode in range(5):
                with self.subTest(count=count, mode=mode):
                    result = self.run_case(self.good, count, mode)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, f"count={count} mode={mode} entry PASS\n")

    def test_guard_precedes_first_resource_or_shader_operation(self):
        _, entry = extract(self.source)
        self.assertEqual(entry.count(GUARD), 1)
        self.assertNotIn("malloc(", entry)
        self.assertNotIn("calloc(", entry)
        self.assertNotIn("pvrgpu_flush_current_color_attachments(", entry)
        self.assertNotIn("pvrgpu_pco_compiler_create(", entry)

    def test_missing_fallback_entry_guard_is_detected(self):
        bad = self.source.replace(GUARD, "", 1)
        result = self.run_case(self.compile("missing-guard", bad), 8, 1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("accepted=1 expected=0", result.stderr)

    def test_guard_moved_after_first_shader_operation_is_detected(self):
        bad = self.source.replace(GUARD, "", 1)
        bad = bad.replace(ENTRY_END, ENTRY_END + "\n" + GUARD, 1)
        result = self.run_case(self.compile("late-guard", bad), 5, 1)
        self.assertNotEqual(result.returncode, 0)

    def test_allowing_eight_targets_is_detected(self):
        bad = self.source.replace("ctx->framebuffer.nr_cbufs > PVRGPU_MAX_RENDER_TARGETS",
                                  "ctx->framebuffer.nr_cbufs > 8", 1)
        result = self.run_case(self.compile("eight-targets", bad), 8, 1)
        self.assertNotEqual(result.returncode, 0)

    def test_quantizing_an_unsupported_mixed_format_is_detected(self):
        bad = self.source.replace("(mixed && !pvrgpu_is_explicit_color_format(util_format_name(surface->format)))",
                                  "(mixed && false)", 1)
        result = self.run_case(self.compile("unsupported-mixed", bad), 4, 2)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
