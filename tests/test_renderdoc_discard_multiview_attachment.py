#!/usr/bin/env python3
"""Apply the real RenderDoc patch and execute its query block with strict mock GL.

This verifies attachment selection, not a live renderer or multiview rendering.
No RenderDoc checkout, Mesa installation, capture, or prepared pixels is needed.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

PATCH = Path(__file__).resolve().parents[1] / "third_party" / "renderdoc-mesa-discard-multiview-attachment.patch"
SOURCE_NAME = "renderdoc/driver/gl/gl_debug.cpp"
BASELINE = "void query()\n{\n  int numviews = 1;\n\n  if(IsGLES && HasExt[OVR_multiview])\n  {\n    GL.glGetNamedFramebufferAttachmentParameterivEXT(\n        framebuffer, eGL_COLOR_ATTACHMENT0, eGL_FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR,\n        &numviews);\n  }\n\n  drv.glBindBufferBase(eGL_UNIFORM_BUFFER, 0, DebugData.discardPatternBuffer);\n}\n"
BEGIN = "  int numviews = 1;"
END = "  drv.glBindBufferBase"
MOCK = r"""
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
using GLenum = unsigned;
using GLint = int;
constexpr GLenum eGL_NONE=0, eGL_TEXTURE=0x1702, eGL_RENDERBUFFER=0x8d41;
constexpr GLenum eGL_COLOR_ATTACHMENT0=0x8ce0, eGL_DEPTH_ATTACHMENT=0x8d00, eGL_STENCIL_ATTACHMENT=0x8d20;
constexpr GLenum eGL_MAX_COLOR_ATTACHMENTS=0x8cdf, eGL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE=0x8cd0;
constexpr GLenum eGL_FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR=0x9630;
constexpr unsigned OVR_multiview=0;
bool IsGLES=true;
std::array<bool,1> HasExt{{true}};
constexpr unsigned framebuffer=37;
struct Api {
  int maximum=4, viewQueries=0, totalQueries=0;
  GLenum selected=eGL_NONE;
  std::map<GLenum,std::pair<GLenum,int>> attachments;
  void glGetIntegerv(GLenum pname, GLint *value) {
    ++totalQueries;
    if(pname != eGL_MAX_COLOR_ATTACHMENTS) throw std::runtime_error("unexpected integer query");
    *value=maximum;
  }
  void glGetNamedFramebufferAttachmentParameterivEXT(unsigned fbo, GLenum att, GLenum pname, GLint *value) {
    ++totalQueries;
    if(fbo != framebuffer) throw std::runtime_error("wrong framebuffer");
    if(!((att>=eGL_COLOR_ATTACHMENT0 && att<eGL_COLOR_ATTACHMENT0+unsigned(maximum)) ||
         att==eGL_DEPTH_ATTACHMENT || att==eGL_STENCIL_ATTACHMENT))
      throw std::runtime_error("invalid attachment enum");
    auto found=attachments.find(att);
    const auto object=found==attachments.end() ? std::make_pair(eGL_NONE,0) : found->second;
    if(pname==eGL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) { *value=int(object.first); return; }
    if(pname!=eGL_FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR) throw std::runtime_error("unexpected attachment query");
    if(object.first!=eGL_TEXTURE) throw std::runtime_error("texture query on absent/renderbuffer attachment");
    ++viewQueries; selected=att; *value=object.second;
  }
} GL;
int query() {
@BLOCK@
  return numviews;
}
int main(int argc, char **argv) {
  if(argc!=2) return 2;
  const int kind=std::atoi(argv[1]);
  int expected=0;
  GLenum selected=eGL_NONE;
  switch(kind) {
    case 0: selected=eGL_DEPTH_ATTACHMENT; break;
    case 1: selected=eGL_DEPTH_ATTACHMENT; expected=4; break;
    case 2: selected=eGL_STENCIL_ATTACHMENT; expected=2; break;
    case 3: GL.attachments[eGL_COLOR_ATTACHMENT0]={eGL_RENDERBUFFER,0}; selected=eGL_DEPTH_ATTACHMENT; break;
    case 4: selected=eGL_COLOR_ATTACHMENT0+3; expected=2; break;
    case 5: selected=eGL_COLOR_ATTACHMENT0; expected=4; GL.attachments[eGL_DEPTH_ATTACHMENT]={eGL_TEXTURE,4}; break;
    case 6: expected=1; break;
    case 7: expected=1; GL.attachments[eGL_COLOR_ATTACHMENT0]={eGL_RENDERBUFFER,0}; GL.attachments[eGL_DEPTH_ATTACHMENT]={eGL_RENDERBUFFER,0}; break;
    case 8: IsGLES=false; expected=1; break;
    case 9: HasExt[OVR_multiview]=false; expected=1; break;
    case 10: selected=eGL_COLOR_ATTACHMENT0; break;
    case 11: GL.maximum=8; selected=eGL_COLOR_ATTACHMENT0+7; expected=4; break;
    case 12: GL.maximum=1; selected=eGL_DEPTH_ATTACHMENT; expected=2; break;
    case 13: selected=eGL_STENCIL_ATTACHMENT; break;
    default: return 2;
  }
  if(selected!=eGL_NONE) GL.attachments[selected]={eGL_TEXTURE,expected};
  const auto before=GL.attachments;
  try {
    const int actual=query();
    if(actual!=expected || GL.attachments!=before || GL.selected!=selected ||
       GL.viewQueries!=(selected==eGL_NONE ? 0 : 1) ||
       ((!IsGLES || !HasExt[OVR_multiview]) && GL.totalQueries!=0)) {
      std::fprintf(stderr,"wrong query result/state: case=%d actual=%d expected=%d viewQueries=%d\n",kind,actual,expected,GL.viewQueries);
      return 1;
    }
  } catch(const std::exception &error) {
    std::fprintf(stderr,"%s\n",error.what()); return 1;
  }
  std::printf("case %d PASS\n",kind);
}
"""

def block(source):
    assert source.count(BEGIN) == source.count(END) == 1
    return BEGIN + source.split(BEGIN, 1)[1].split(END, 1)[0]

class DiscardMultiviewAttachment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
        if not cls.compiler:
            raise unittest.SkipTest("A C++17 compiler is required for actual patched-block mocks")
        cls.temp = tempfile.TemporaryDirectory(prefix="renderdoc-multiview-unit-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        source = cls.root / SOURCE_NAME
        source.parent.mkdir(parents=True)
        source.write_text(BASELINE)
        subprocess.run(["git", "init", "-q", str(cls.root)], check=True, capture_output=True)
        for args in (("--check",), ()):
            subprocess.run(["git", "-C", str(cls.root), "apply", *args, str(PATCH)], check=True, capture_output=True)
        cls.patched = source.read_text()
        cls.positive = cls.compile("patched", cls.patched)
        cls.old = cls.compile("old", BASELINE)

    @classmethod
    def compile(cls, name, source):
        path = cls.root / (name + ".cpp")
        path.write_text(MOCK.replace("@BLOCK@", block(source)))
        binary = cls.root / name
        result = subprocess.run([cls.compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                 str(path), "-o", str(binary)], text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return binary

    def execute(self, binary, case):
        return subprocess.run([str(binary), str(case)], text=True, capture_output=True)

    def test_actual_patched_block_all_attachment_shapes(self):
        for case in range(14):
            with self.subTest(case=case):
                result = self.execute(self.positive, case)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"case {case} PASS\n")

    def test_original_depth_only_query_is_rejected(self):
        result = self.execute(self.old, 0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("texture query on absent/renderbuffer attachment", result.stderr)

    def test_original_color_zero_control_is_preserved(self):
        for case in (5, 8, 9, 10):
            with self.subTest(case=case):
                self.assertEqual(self.execute(self.old, case).returncode, 0)

    def test_missing_object_type_guard_is_detected(self):
        bad = self.patched.replace("if(objectType == eGL_TEXTURE)", "if(true)")
        result = self.execute(self.compile("unguarded", bad), 0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("texture query on absent/renderbuffer attachment", result.stderr)

    def test_wrong_selected_attachment_is_detected(self):
        bad = self.patched.replace(
            "framebuffer, attachment, eGL_FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR",
            "framebuffer, eGL_COLOR_ATTACHMENT0, eGL_FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR")
        self.assertNotEqual(self.execute(self.compile("wrong-attachment", bad), 1).returncode, 0)

    def test_multiview_count_must_not_be_forced_to_one(self):
        bad = self.patched.replace("        break;", "        numviews = 1;\n        break;")
        result = self.execute(self.compile("forced-one", bad), 1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("actual=1 expected=4", result.stderr)

    def test_patch_changes_only_query_selection_context(self):
        self.assertEqual(self.patched.split(BEGIN)[0], BASELINE.split(BEGIN)[0])
        self.assertEqual(self.patched.split(END)[1], BASELINE.split(END)[1])
        self.assertNotIn("glGetError", block(self.patched))
        self.assertNotIn("glDisable", block(self.patched))

if __name__ == "__main__":
    unittest.main()
