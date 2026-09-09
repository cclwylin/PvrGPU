#!/usr/bin/env python3
"""Apply the real alias patch to a switch fixture; not renderer evidence."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile
import unittest


PATCH = (Path(__file__).resolve().parents[1] / "third_party" /
         "renderdoc-mesa-patch-parameter-alias.patch")
SOURCE_NAME = "renderdoc/driver/gl/gl_driver.cpp"
SWITCH_BEGIN = "  switch(chunk)\n  {\n"
SWITCH_END = "    default: return false;\n  }\n"

# The affected upstream dispatch context, with the original incorrect aliases.
# No external RenderDoc checkout, Mesa install, compiler, or capture is needed.
BASELINE = """bool WrappedOpenGL::ContextProcessChunk(ReadSerialiser &ser, GLChunk chunk)
{
  switch(chunk)
  {
    case GLChunk::glMinSampleShadingARB:
    case GLChunk::glMinSampleShadingOES:
    case GLChunk::glMinSampleShading: return Serialise_glMinSampleShading(ser, 0);
    case GLChunk::glRasterSamplesEXT: return Serialise_glRasterSamplesEXT(ser, 0, 0);
    case GLChunk::glPatchParameteri: return Serialise_glPatchParameteri(ser, eGL_NONE, 0);
    case GLChunk::glPatchParameterfv: return Serialise_glPatchParameterfv(ser, eGL_NONE, 0);
    case GLChunk::glLineWidth: return Serialise_glLineWidth(ser, 0);
    case GLChunk::glPointSize: return Serialise_glPointSize(ser, 0);
    case GLChunk::glPatchParameteriEXT:
    case GLChunk::glPatchParameteriOES:
    case GLChunk::glPointParameteri: return Serialise_glPointParameteri(ser, eGL_NONE, 0);
    case GLChunk::glPointParameteriv: return Serialise_glPointParameteriv(ser, eGL_NONE, 0);
    case GLChunk::glPointParameterfARB:
    case GLChunk::glPointParameterfEXT:
    case GLChunk::glPointParameterf: return Serialise_glPointParameterf(ser, eGL_NONE, 0);
    case GLChunk::glPointParameterfvARB:
    case GLChunk::glPointParameterfvEXT:
    case GLChunk::glPointParameterfv: return Serialise_glPointParameterfv(ser, eGL_NONE, 0);
    default: return false;
  }
}
"""


def dispatch_table(source: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Resolve grouped labels in this small switch, rejecting unknown syntax."""
    assert source.count(SWITCH_BEGIN) == source.count(SWITCH_END) == 1
    body = source.split(SWITCH_BEGIN, 1)[1].split(SWITCH_END, 1)[0]
    table = {}
    labels = []
    while body.strip():
        body = body.lstrip()
        case = re.match(r"case GLChunk::(\w+):", body)
        if case:
            name = case.group(1)
            assert name not in table and name not in labels, "duplicate case: " + name
            labels.append(name)
            body = body[case.end():]
            continue
        call = re.match(r"return (Serialise_\w+)\(([^()]*)\);", body)
        assert call is not None and labels, "unexpected switch statement: " + body
        dispatch = (call.group(1), tuple(arg.strip() for arg in call.group(2).split(",")))
        for name in labels:
            table[name] = dispatch
        labels.clear()
        body = body[call.end():]
    assert not labels, "unterminated case group"
    return table


def assert_parameter_dispatch(source: str) -> None:
    table = dispatch_table(source)
    for suffix in ("", "EXT", "OES"):
        name = "glPatchParameteri" + suffix
        assert table.get(name) == ("Serialise_glPatchParameteri", ("ser", "eGL_NONE", "0")), name
    for name in ("glPointParameteri", "glPointParameteriv", "glPointParameterf",
                 "glPointParameterfv", "glPatchParameterfv"):
        assert table.get(name) == ("Serialise_" + name, ("ser", "eGL_NONE", "0")), name


class PatchParameterAlias(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="renderdoc-alias-unit-")
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        source = root / SOURCE_NAME
        source.parent.mkdir(parents=True)
        source.write_text(BASELINE)
        subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
        for options in (("--check",), ()):
            result = subprocess.run(["git", "-C", str(root), "apply", *options, str(PATCH)],
                                    text=True, capture_output=True)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
        cls.patched = source.read_text()

    def test_real_patch_routes_core_ext_and_oes_to_patch_serializer(self):
        assert_parameter_dispatch(self.patched)

    def test_only_extension_alias_dispatch_changes(self):
        before = dispatch_table(BASELINE)
        after = dispatch_table(self.patched)
        self.assertEqual(set(before), set(after))
        self.assertEqual({name for name in before if before[name] != after[name]},
                         {"glPatchParameteriEXT", "glPatchParameteriOES"})

    def test_original_bad_mapping_is_rejected(self):
        for suffix in ("EXT", "OES"):
            self.assertEqual(dispatch_table(BASELINE)["glPatchParameteri" + suffix],
                             ("Serialise_glPointParameteri", ("ser", "eGL_NONE", "0")))
        with self.assertRaises(AssertionError):
            assert_parameter_dispatch(BASELINE)

    def test_each_patch_alias_misroute_is_rejected(self):
        for suffix in ("", "EXT", "OES"):
            with self.subTest(suffix=suffix):
                label = "case GLChunk::glPatchParameteri" + suffix + ":"
                if suffix:
                    bad = self.patched.replace(label, label +
                        " return Serialise_glPointParameteri(ser, eGL_NONE, 0);")
                else:
                    bad = self.patched.replace(label + " return Serialise_glPatchParameteri",
                                               label + " return Serialise_glPointParameteri")
                with self.assertRaises(AssertionError):
                    assert_parameter_dispatch(bad)

    def test_real_point_parameter_misroute_is_rejected(self):
        bad = self.patched.replace(
            "case GLChunk::glPointParameteri: return Serialise_glPointParameteri",
            "case GLChunk::glPointParameteri: return Serialise_glPatchParameteri")
        with self.assertRaises(AssertionError):
            assert_parameter_dispatch(bad)

    def test_missing_or_duplicate_extension_case_is_rejected(self):
        for suffix in ("EXT", "OES"):
            label = "    case GLChunk::glPatchParameteri" + suffix + ":\n"
            for replacement in ("", label * 2):
                with self.subTest(suffix=suffix, duplicate=bool(replacement)):
                    with self.assertRaises(AssertionError):
                        assert_parameter_dispatch(self.patched.replace(label, replacement))


if __name__ == "__main__":
    unittest.main()
