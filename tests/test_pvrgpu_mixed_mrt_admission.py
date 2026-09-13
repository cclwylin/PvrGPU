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
FORMAT_BEGIN = "const char *\npvrgpu_command_format_for_surface("
FORMAT_END = "static const char *\npvrgpu_command_format_for_framebuffer("
HELPERS_BEGIN = "static bool\npvrgpu_framebuffer_has_mixed_color_formats("
HELPERS_END = "static bool\npvrgpu_read_user_float2_vertex("
ENTRY_BEGIN = "static bool\npvrgpu_record_color_primitive_pco_draw_attempt("
ENTRY_END = "   const bool has_tessellation = ctx->tcs && ctx->tes;"
CLEAR_BEGIN = "   if (ctx->framebuffer.nr_cbufs &&\n       ctx->framebuffer.cbufs[0].texture) {"
CLEAR_END = "   command.raw_vertex_data = (const uint8_t *)interleaved;"
GUARD = """   if (!pvrgpu_framebuffer_color_transport_is_bounded(ctx)) {
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=entry reason=color_transport_bounds");
      return false;
   }
"""


def extract(source):
    for marker in (FORMAT_BEGIN, FORMAT_END, HELPERS_BEGIN, HELPERS_END,
                   ENTRY_BEGIN, CLEAR_BEGIN, CLEAR_END):
        assert source.count(marker) == 1, "ambiguous or missing production function"
    formats = FORMAT_BEGIN + source.split(FORMAT_BEGIN, 1)[1].split(FORMAT_END, 1)[0]
    helpers = HELPERS_BEGIN + source.split(HELPERS_BEGIN, 1)[1].split(HELPERS_END, 1)[0]
    rest = source.split(ENTRY_BEGIN, 1)[1]
    assert rest.count(ENTRY_END) == 1, "missing first post-entry operation"
    entry = ENTRY_BEGIN + rest.split(ENTRY_END, 1)[0]
    clear = CLEAR_BEGIN + source.split(CLEAR_BEGIN, 1)[1].split(CLEAR_END, 1)[0]
    return formats + helpers, entry, clear


MOCK = r"""
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pvrgpu_cmd.h"
#include "pvrgpu_color_formats.h"
#define PVRGPU_MAX_RENDER_TARGETS 4u
enum pipe_format {
  RGBA8, RGB10, BGR10, R16, RGBA8_SRGB, BGRA8_SRGB,
  PIPE_FORMAT_A8B8G8R8_SRGB,
  PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R8G8_UNORM,
  PIPE_FORMAT_R8G8B8X8_UNORM, PIPE_FORMAT_B8G8R8X8_UNORM,
  PIPE_FORMAT_R5G6B5_UNORM, PIPE_FORMAT_B5G6R5_UNORM,
  PIPE_FORMAT_R16G16_UNORM, PIPE_FORMAT_R16G16B16A16_UNORM,
  PIPE_FORMAT_R32G32B32A32_UNORM,
  PIPE_FORMAT_R8_SNORM, PIPE_FORMAT_R8G8_SNORM,
  PIPE_FORMAT_R8G8B8A8_SNORM, PIPE_FORMAT_R16_SNORM,
  PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R16G16B16A16_SNORM,
  PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R16_FLOAT,
  PIPE_FORMAT_R16G16_FLOAT, PIPE_FORMAT_R16G16B16A16_FLOAT,
  PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32G32_FLOAT,
  PIPE_FORMAT_R8_UINT, PIPE_FORMAT_R8G8_UINT,
  PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R16_UINT,
  PIPE_FORMAT_R16G16_UINT, PIPE_FORMAT_R16G16B16A16_UINT,
  PIPE_FORMAT_R32_UINT, PIPE_FORMAT_R32G32_UINT,
  PIPE_FORMAT_R32G32B32A32_UINT, PIPE_FORMAT_R10G10B10A2_UINT,
  PIPE_FORMAT_B10G10R10A2_UINT,
  PIPE_FORMAT_R8_SINT, PIPE_FORMAT_R8G8_SINT,
  PIPE_FORMAT_R8G8B8A8_SINT, PIPE_FORMAT_R16_SINT,
  PIPE_FORMAT_R16G16_SINT, PIPE_FORMAT_R16G16B16A16_SINT,
  PIPE_FORMAT_R32_SINT, PIPE_FORMAT_R32G32_SINT,
  PIPE_FORMAT_R32G32B32A32_SINT
};
#define PIPE_FORMAT_R8G8B8A8_SRGB RGBA8_SRGB
#define PIPE_FORMAT_B8G8R8A8_SRGB BGRA8_SRGB
#define PIPE_FORMAT_R10G10B10A2_UNORM RGB10
#define PIPE_FORMAT_B10G10R10A2_UNORM BGR10
#define PIPE_FORMAT_R16_UNORM R16
struct pipe_resource {int unused;};
struct pipe_surface {struct pipe_resource *texture; enum pipe_format format;};
struct pipe_framebuffer_state {unsigned nr_cbufs; struct pipe_surface cbufs[8];};
struct pvrgpu_context {
  struct pipe_framebuffer_state framebuffer;
  uint32_t color_clear_bits[4];
};
struct pipe_draw_info {int unused;};
struct pipe_draw_start_count_bias {int unused;};
static const char *util_format_name(enum pipe_format format) {
  switch (format) {
  case RGBA8: return "PIPE_FORMAT_R8G8B8A8_UNORM";
  case RGB10: return "PIPE_FORMAT_R10G10B10A2_UNORM";
  case BGR10: return "PIPE_FORMAT_B10G10R10A2_UNORM";
  case R16: return "PIPE_FORMAT_R16_UNORM";
  case PIPE_FORMAT_A8B8G8R8_SRGB: return "PIPE_FORMAT_A8B8G8R8_SRGB";
  default: return "PIPE_FORMAT_OTHER";
  }
}
static bool util_format_is_pure_uint(enum pipe_format format) {(void)format;return false;}
static bool util_format_is_pure_sint(enum pipe_format format) {(void)format;return false;}
static bool util_format_is_float(enum pipe_format format) {(void)format;return false;}
static unsigned util_format_get_nr_components(enum pipe_format format) {(void)format;return 4;}
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
static int check_command_clear(struct pvrgpu_context *ctx, bool expect_dummy) {
  struct pvrgpu_draw_pco_triangles_command command={0};
@CLEAR@
  const uint32_t canonical[4]={0,0,0,UINT32_C(0x3f800000)};
  const uint32_t *expected=expect_dummy ? canonical : ctx->color_clear_bits;
  return memcmp(command.clear_color_bits,expected,sizeof(canonical))==0;
}
int main(int argc,char **argv) {
  if(argc!=3) return 2;
  if(strcmp(util_format_name(RGBA8),"PIPE_FORMAT_R8G8B8A8_UNORM")) return 2;
  unsigned count=(unsigned)strtoul(argv[1],NULL,10);
  unsigned mode=(unsigned)strtoul(argv[2],NULL,10);
  if(count>8 || mode>6) return 2;
  struct pipe_resource resource={0};
  struct pvrgpu_context ctx={0};
  ctx.framebuffer.nr_cbufs=count;
  for(unsigned i=0;i<8;++i) {
    ctx.framebuffer.cbufs[i].texture=&resource;
    ctx.framebuffer.cbufs[i].format=mode==4 ? R16 :
      mode==6 ? PIPE_FORMAT_A8B8G8R8_SRGB : RGBA8;
  }
  if(mode==1) {ctx.framebuffer.cbufs[1].format=RGB10;ctx.framebuffer.cbufs[2].format=BGR10;}
  if(mode==2) ctx.framebuffer.cbufs[1].format=R16;
  if(mode==3 && count) ctx.framebuffer.cbufs[count-1].texture=NULL;
  if(mode==5 && count) ctx.framebuffer.cbufs[0].texture=NULL;
  for(unsigned component=0;component<4;++component)
    ctx.color_clear_bits[component]=UINT32_C(0x7fc00000)+component;
  bool expected_mixed=count>1 && (mode==1 || mode==2);
  if(pvrgpu_framebuffer_has_mixed_color_formats(&ctx)!=expected_mixed) return 2;
  /* mode 3 has a legal GL_NONE hole below nr_cbufs.  It must reach the
   * compiler boundary while counts above the fixed four-target ABI do not. */
  bool expected=count<=4;
  const struct pvrgpu_context before=ctx;
  struct pipe_draw_info info={0};struct pipe_draw_start_count_bias draw={0};
  bool terminal=false;
  bool actual=pvrgpu_record_color_primitive_pco_draw_attempt(&ctx,&info,&draw,true,&terminal);
  bool command_clear_ok=check_command_clear(
    &ctx,count==0 || mode==5 || (mode==3 && count==1));
  if(actual!=expected || compiler_boundary!=(expected ? 1u : 0u) ||
     bounds_errors!=(expected ? 0u : 1u) || !command_clear_ok || terminal ||
     memcmp(&before,&ctx,sizeof(ctx))) {
    fprintf(stderr,"entry mismatch: count=%u mode=%u accepted=%u expected=%u boundary=%u errors=%u clear=%u\n",
            count,mode,actual,expected,compiler_boundary,bounds_errors,command_clear_ok);
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
        helpers, entry, clear = extract(source)
        code = (MOCK.replace("@HELPERS@", helpers).replace("@ENTRY@", entry)
                .replace("@CLEAR@", clear))
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
            for mode in range(7):
                with self.subTest(count=count, mode=mode):
                    result = self.run_case(self.good, count, mode)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, f"count={count} mode={mode} entry PASS\n")

    def test_guard_precedes_first_resource_or_shader_operation(self):
        _, entry, _ = extract(self.source)
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

    def test_bypassing_canonical_transport_mapping_is_detected(self):
        bad = self.source.replace("pvrgpu_command_format_for_surface(surface->format)",
                                  "util_format_name(surface->format)", 1)
        result = self.run_case(self.compile("uncanonical-mixed", bad), 4, 6)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("accepted=0 expected=1", result.stderr)

    def test_leading_hole_canonicalizes_stale_nonfinite_integer_clear(self):
        result = self.run_case(self.good, 4, 5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("count=4 mode=5 entry PASS", result.stdout)

        bad = self.source.replace(
            "   } else {\n      command.clear_color_bits[3] = UINT32_C(0x3f800000);\n   }",
            "   } else {\n      memcpy(command.clear_color_bits, ctx->color_clear_bits, "
            "sizeof(command.clear_color_bits));\n   }", 1)
        rejected = self.run_case(self.compile("stale-leading-clear", bad), 4, 5)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("clear=0", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
