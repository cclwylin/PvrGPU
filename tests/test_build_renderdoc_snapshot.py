#!/usr/bin/env python3
"""Private builder protocol tests; mock compilers are NOT renderer evidence."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "script"
sys.path.insert(0, str(SCRIPT))
import build_renderdoc_snapshot as builder


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PrivateSnapshotBuilder(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="snapshot-builder-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source with spaces"
        self.base = self.root / "base with spaces"
        self.source.mkdir(); self.base.mkdir()
        self.old = {name: ("// baseline " + name + "\n").encode() for name in builder.CHANGED_EXISTING}
        self.new = {name: ("// snapshot " + name + "\n").encode()
                    for name in builder.CHANGED_EXISTING | builder.ADDED}
        self.old["vendor.txt"] = b"pinned existing vendor modification\n"
        self.old["empty.txt"] = b""
        for name, data in self.old.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Unit Test", "-c", "user.email=unit@example.invalid", "commit", "-qm", "fixture")
        head = self.git("rev-parse", "HEAD").strip()
        self.patch = self.root / "snapshot.patch"
        diffs = []
        for name in sorted(self.new):
            before = self.old.get(name, b"").decode().splitlines(keepends=True)
            after = self.new[name].decode().splitlines(keepends=True)
            diffs.append(f"diff --git a/{name} b/{name}\n")
            if name not in self.old:
                diffs.append("new file mode 100644\n")
            diffs.extend(difflib.unified_diff(before, after,
                fromfile="a/" + name if before else "/dev/null", tofile="b/" + name))
        self.patch.write_text("".join(diffs))
        self.lock_data = {"schema": builder.LOCK_SCHEMA, "source_git_head": head,
            "baseline_tracked_tree_sha256": builder.tree_digest({name: sha(data) for name, data in self.old.items()}),
            "patch_sha256": sha(self.patch.read_bytes()),
            "files": {name: {"base_sha256": sha(self.old[name]) if name in self.old else None,
                              "patched_sha256": sha(data)} for name, data in self.new.items()}}
        self.lock = self.root / "snapshot.lock.json"; self.write_lock()
        (self.base / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(self.source) + "\n")
        (self.base / "build.ninja").write_text("# mocked command inventory, never build\n")
        self.originals = {self.base / target: b"old object " + name.encode() for name, target in builder.TARGETS.items()}
        self.originals[self.base / "unchanged.o"] = b"reused unchanged object"
        self.originals[self.base / "lib/librenderdoc.dylib"] = b"installed library remains untouched"
        for path, data in self.originals.items():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        self.compiler = self.root / "mock compiler"
        self.compiler.write_text("#!" + sys.executable + "\n" +
            "import json, pathlib, sys\n"
            "args=sys.argv[1:]\n"
            "if args == ['--version']: print('mock compiler: protocol testing only'); sys.exit(0)\n"
            "if '-DFAIL_TEST_COMMAND' in args: sys.exit(23)\n"
            "out=pathlib.Path(args[args.index('-o')+1]); out.write_text(json.dumps(args))\n"
            "if '-MF' in args: pathlib.Path(args[args.index('-MF')+1]).write_text('private dependency file')\n")
        self.compiler.chmod(0o700)
        self.ninja = self.root / "mock ninja"
        self.ninja.write_text("#!" + sys.executable + "\n" +
            "import json, pathlib, sys\n"
            "a=sys.argv[1:]\n"
            "assert len(a)==5 and a[0]=='-C' and a[2:4]==['-t','commands'], 'must only query Ninja'\n"
            "print(json.loads((pathlib.Path(a[1])/'commands.json').read_text())[a[4]])\n")
        self.ninja.chmod(0o700)
        self.commands = {}
        for name, target in builder.TARGETS.items():
            command = [str(self.compiler), "-O3", "-DNDEBUG", "-arch", "x86_64", "-I" + str(self.source),
                       "-MD", "-MT", target, "-MF", target + ".d", "-o", target, "-c", str(self.source / name)]
            self.commands[target] = shlex.join(command)
        self.link_target = "lib/librenderdoc.dylib"
        self.link = [str(self.compiler), "-O3", "-dynamiclib", "-o", self.link_target,
                     "-install_name", "@rpath/librenderdoc.dylib", *builder.TARGETS.values(), "unchanged.o"]
        self.set_link()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    def write_lock(self):
        self.lock.write_text(json.dumps(self.lock_data))

    def set_link(self, suffix=""):
        # Non-target generator redirection is inventory only; it is never run.
        self.commands[self.link_target] = "printf unused > must-not-exist.c\n: && " + shlex.join(self.link) + " && :" + suffix
        self.write_commands()

    def write_commands(self):
        (self.base / "commands.json").write_text(json.dumps(self.commands))

    def run_build(self, expected=0, out=None):
        out = out or self.root / "private output"
        process = subprocess.run([sys.executable, str(SCRIPT / "build_renderdoc_snapshot.py"),
            str(self.source), str(self.base), str(out), "--lock", str(self.lock), "--patch", str(self.patch),
            "--ninja", str(self.ninja), "--jobs", "2"], text=True, capture_output=True)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        for path, data in self.originals.items():
            self.assertEqual(path.read_bytes(), data, str(path))
        self.assertFalse((self.base / "must-not-exist.c").exists())
        self.assertFalse(any(self.base.rglob("*.o.d")))
        return out, process

    def assert_built(self, out):
        receipt = json.loads((out / "build-receipt.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["source_unchanged"])
        self.assertTrue(receipt["shared_outputs_unchanged"])
        self.assertEqual((out / "source.patch").read_bytes(), self.patch.read_bytes())
        self.assertEqual((out / "source-lock.json").read_bytes(), self.lock.read_bytes())
        self.assertEqual(len(receipt["builder_sources"]), 3)
        for name, data in self.new.items():
            self.assertEqual((out / "source" / name).read_bytes(), data)
        for args in receipt["compile_commands"].values():
            self.assertIn("-O3", args); self.assertIn("-DNDEBUG", args)
            self.assertIn("-I" + str(out / "source"), args)
            self.assertNotIn(str(self.source), "\n".join(args))
            for flag in ("-o", "-MF", "-MT"):
                self.assertTrue(Path(args[args.index(flag) + 1]).is_relative_to(out / "objects"))
        link = receipt["link_command"]
        self.assertEqual(link[link.index("-install_name") + 1], str(out / "lib/librenderdoc-drawlist-snapshot.dylib"))
        self.assertEqual(len(receipt["object_identities"]), 5)
        self.assertEqual(receipt["library_identity"]["sha256"], sha(Path(receipt["library"]).read_bytes()))

    def test_exact_baseline_build_is_private(self):
        out, _ = self.run_build()
        self.assert_built(out)
        for name, data in self.old.items():
            self.assertEqual((self.source / name).read_bytes(), data)
        self.assertFalse(any((self.source / name).exists() for name in builder.ADDED))

    def test_patch_under_repository_subdirectory_has_complete_inventory(self):
        directory = self.source / ".git" / "patches"
        directory.mkdir()
        patch = directory / "snapshot.patch"; patch.write_bytes(self.patch.read_bytes())
        self.assertEqual(builder.patch_names(patch, "git"), set(self.new))

    def test_output_under_another_worktree_uses_private_git_root(self):
        unrelated = self.root / "unrelated-worktree"; unrelated.mkdir()
        subprocess.check_call(["git", "init", "-q", str(unrelated)])
        out, _ = self.run_build(out=unrelated / "private-build")
        self.assert_built(out)

    def test_already_patched_and_mixed_sources_normalize(self):
        for mode in ("mixed", "patched"):
            with self.subTest(mode=mode):
                for i, (name, data) in enumerate(sorted(self.new.items())):
                    if mode == "patched" or i % 2 == 0:
                        path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                out, _ = self.run_build(out=self.root / mode)
                self.assert_built(out)

    def test_unknown_vendor_edit_refuses(self):
        (self.source / "vendor.txt").write_text("unreviewed vendor changes\n")
        _, result = self.run_build(expected=1)
        self.assertIn("pinned Mesa/vendor baseline", result.stderr)

    def test_partial_patch_refuses(self):
        (self.source / next(iter(builder.CHANGED_EXISTING))).write_text("half applied\n")
        _, result = self.run_build(expected=1)
        self.assertIn("unrecognized source bytes", result.stderr)

    def test_unknown_untracked_file_refuses(self):
        (self.source / "unexpected.cpp").write_text("untracked\n")
        _, result = self.run_build(expected=1)
        self.assertIn("unknown untracked", result.stderr)

    def test_patch_hash_and_head_refuse(self):
        self.lock_data["patch_sha256"] = "0" * 64; self.write_lock()
        _, result = self.run_build(expected=1)
        self.assertIn("patch hash mismatch", result.stderr)
        self.lock_data["patch_sha256"] = sha(self.patch.read_bytes())
        self.lock_data["source_git_head"] = "0" * 40; self.write_lock()
        _, result = self.run_build(expected=1)
        self.assertIn("Git HEAD", result.stderr)

    def test_wrong_build_source_refuses(self):
        (self.base / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(self.root) + "\n")
        _, result = self.run_build(expected=1)
        self.assertIn("different source", result.stderr)

    def test_fresh_and_external_output_required(self):
        occupied = self.root / "occupied"; occupied.mkdir()
        for path in (occupied, self.base / "private", self.source / "private"):
            with self.subTest(path=path):
                self.run_build(expected=1, out=path)

    def test_source_and_output_symlinks_refuse(self):
        vendor = self.source / "vendor.txt"
        elsewhere = self.root / "vendor-copy"; elsewhere.write_bytes(vendor.read_bytes())
        vendor.unlink(); vendor.symlink_to(elsewhere)
        _, result = self.run_build(expected=1)
        self.assertIn("symlink", result.stderr.lower())
        vendor.unlink(); vendor.write_bytes(self.old["vendor.txt"])
        link = self.root / "output-link"; link.symlink_to(self.root / "uncreated")
        self.run_build(expected=1, out=link)

    def test_extra_link_step_and_missing_replacement_refuse(self):
        self.set_link(" && touch forbidden")
        _, result = self.run_build(expected=1)
        self.assertIn("unsupported prerequisite/post-link", result.stderr)
        self.link.remove(next(iter(builder.TARGETS.values()))); self.set_link()
        _, result = self.run_build(expected=1)
        self.assertIn("all five changed objects", result.stderr)

    def test_auxiliary_compiler_outputs_refuse(self):
        target = next(iter(builder.TARGETS.values()))
        self.commands[target] += " -ftime-trace"
        self.write_commands()
        _, result = self.run_build(expected=1)
        self.assertIn("auxiliary output flags", result.stderr)

    def test_response_file_refuses(self):
        self.link.append("@uninspected-args.rsp"); self.set_link()
        _, result = self.run_build(expected=1)
        self.assertIn("response files", result.stderr)

    def test_failed_compile_retains_failure_receipt(self):
        target = next(iter(builder.TARGETS.values()))
        self.commands[target] += " -DFAIL_TEST_COMMAND"; self.write_commands()
        out, _ = self.run_build(expected=1)
        receipt = json.loads((out / "build-receipt.json").read_text())
        self.assertEqual(receipt["status"], "FAIL")
        self.assertTrue(receipt["source_unchanged"] and receipt["shared_outputs_unchanged"])
        self.assertNotIn("library_identity", receipt)

    def test_relative_path_contract(self):
        for value in ("../escape", "/absolute", "a/../b", "a//b", "a\nb", "a\0b", "./a"):
            with self.subTest(value=value), self.assertRaises(builder.ReplayError):
                builder.relative_name(value)


if __name__ == "__main__":
    unittest.main()
