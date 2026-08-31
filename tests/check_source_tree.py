#!/usr/bin/env python3
"""Reject generated and compiled artifacts inside the source tree.

This is the cross-platform, Python-standard-library replacement for the
former check-source-tree.sh guard. The public check function is importable,
and --self-test runs temporary-fixture unit tests without touching the source
tree.
"""

from __future__ import annotations

import argparse
import os
import stat
import struct
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


GENERATED_PATHS = (
    ".venv",
    "build",
    "out",
    "tmp",
    "tools/__pycache__",
    "tests/__pycache__",
)
COMPILED_SUFFIXES = {".o", ".a", ".so", ".dylib"}
MACH_O_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}
MAX_PE_HEADER_OFFSET = 16 * 1024 * 1024


@dataclass(frozen=True)
class CheckResult:
    errors: tuple[str, ...]


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _walk_source_tree(root: Path) -> tuple[list[Path], list[str]]:
    entries: list[Path] = []
    walk_errors: list[str] = []

    def on_error(error: OSError) -> None:
        walk_errors.append(f"Cannot scan source tree: {error}")

    for directory, directory_names, file_names in os.walk(
        root, topdown=True, onerror=on_error, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        parent = Path(directory)
        entries.extend(parent / name for name in directory_names)
        entries.extend(parent / name for name in file_names)

    entries.sort(key=lambda path: _display_path(path, root))
    return entries, walk_errors


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _has_executable_permissions(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False

    if os.name == "nt":
        # Windows does not expose POSIX execute bits consistently. A native PE
        # file is executable by format; other formats retain the mode check.
        if path.suffix.lower() in {".exe", ".com"}:
            return True
        return bool(mode & stat.S_IXUSR)

    # Match the former find -perm -111 expression: all three execute bits.
    return mode & 0o111 == 0o111


def _binary_kind(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
            if header.startswith(b"\x7fELF"):
                return "ELF"
            if header[:4] in MACH_O_MAGICS:
                return "Mach-O"
            if not header.startswith(b"MZ") or len(header) < 64:
                return None

            pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
            if pe_offset > MAX_PE_HEADER_OFFSET:
                return None
            stream.seek(pe_offset)
            pe_header = stream.read(26)
    except (OSError, struct.error):
        return None

    if not pe_header.startswith(b"PE\x00\x00") or len(pe_header) < 26:
        return None
    optional_magic = struct.unpack_from("<H", pe_header, 24)[0]
    if optional_magic in {0x010B, 0x020B}:
        return "PE32"
    return None


def check_source_tree(root: Path) -> CheckResult:
    root = root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        return CheckResult((f"Source tree does not exist: {root}",))

    for relative_path in GENERATED_PATHS:
        if (root / relative_path).exists():
            errors.append(f"Generated path found in source tree: {relative_path}")

    entries, walk_errors = _walk_source_tree(root)
    errors.extend(walk_errors)

    if any(path.name.endswith(".pyc") for path in entries):
        errors.append("Python bytecode found in source tree.")
    if any(path.name == ".DS_Store" for path in entries):
        errors.append("Finder metadata found in source tree.")

    for path in entries:
        if (
            _is_regular_file(path)
            and _has_executable_permissions(path)
            and _binary_kind(path) is not None
        ):
            errors.append(
                "Compiled executable found in source tree: "
                f"{_display_path(path, root)}"
            )

    has_compiled_artifact = any(
        (_is_regular_file(path) and path.suffix in COMPILED_SUFFIXES)
        or (_is_real_directory(path) and path.name.endswith(".dSYM"))
        for path in entries
    )
    if has_compiled_artifact:
        errors.append(
            "Compiled object/library/debug bundle found in source tree."
        )

    return CheckResult(tuple(errors))


class SourceTreeGuardSelfTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="pvrgpu-source-tree-check-"
        )
        self.root = Path(self._temporary.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "example.cpp").write_text(
            "int example() { return 0; }\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def assert_reports(self, expected: str) -> None:
        result = check_source_tree(self.root)
        self.assertTrue(
            any(expected in error for error in result.errors),
            f"{expected!r} not found in {result.errors!r}",
        )

    def test_clean_tree_passes(self) -> None:
        script = self.root / "tools" / "helper.py"
        script.parent.mkdir()
        script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        script.chmod(0o755)
        self.assertEqual(check_source_tree(self.root).errors, ())

    def test_generated_path_is_rejected(self) -> None:
        (self.root / "build").mkdir()
        self.assert_reports("Generated path found in source tree: build")

    def test_python_bytecode_is_rejected(self) -> None:
        (self.root / "src" / "example.pyc").write_bytes(b"bytecode")
        self.assert_reports("Python bytecode found in source tree.")

    def test_finder_metadata_is_rejected(self) -> None:
        (self.root / "src" / ".DS_Store").write_bytes(b"metadata")
        self.assert_reports("Finder metadata found in source tree.")

    def test_elf_executable_is_rejected(self) -> None:
        executable = self.root / "src" / "program"
        executable.write_bytes(b"\x7fELF" + bytes(60))
        executable.chmod(0o755)
        self.assert_reports("Compiled executable found in source tree: src/program")

    def test_mach_o_executable_is_rejected(self) -> None:
        executable = self.root / "src" / "program"
        executable.write_bytes(b"\xcf\xfa\xed\xfe" + bytes(60))
        executable.chmod(0o755)
        self.assert_reports("Compiled executable found in source tree: src/program")

    def test_pe_executable_is_rejected(self) -> None:
        data = bytearray(256)
        data[0:2] = b"MZ"
        struct.pack_into("<I", data, 0x3C, 0x80)
        data[0x80:0x84] = b"PE\x00\x00"
        struct.pack_into("<H", data, 0x80 + 24, 0x020B)
        executable = self.root / "src" / "program.exe"
        executable.write_bytes(data)
        executable.chmod(0o755)
        self.assert_reports(
            "Compiled executable found in source tree: src/program.exe"
        )

    def test_compiled_object_is_rejected(self) -> None:
        (self.root / "src" / "example.o").write_bytes(b"object")
        self.assert_reports(
            "Compiled object/library/debug bundle found in source tree."
        )

    def test_debug_bundle_is_rejected(self) -> None:
        (self.root / "src" / "example.dSYM").mkdir()
        self.assert_reports(
            "Compiled object/library/debug bundle found in source tree."
        )


def run_self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        SourceTreeGuardSelfTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    print("Source tree guard self-test: PASS")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"repository root (default: {default_root})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run temporary-fixture unit tests",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        return run_self_test()

    result = check_source_tree(args.root)
    if result.errors:
        for error in result.errors:
            print(error, file=sys.stderr)
        return 1

    print("Source tree guard PASS: generated files are outside iCloud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
