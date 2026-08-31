#!/usr/bin/env python3
"""Enforce the one-SystemC-module/one-header/one-source layout rule.

The checker deliberately uses only the Python standard library so it can run
before the C++ build.  It is a source-layout check, not a C++ parser; the
regular expressions cover the explicit ``sc_core::sc_module`` inheritance and
``SC_MODULE`` forms used by this project.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
SOURCE_SUFFIXES = {".cxx", ".cc", ".cpp"}

_COMMENT_OR_LITERAL_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)
_CLASS_DEFINITION_RE = re.compile(
    r"\b(?:class|struct)\s+(?P<name>[A-Za-z_]\w*)\b"
    r"(?P<tail>[^;{}]*)\{",
    re.DOTALL,
)
_SC_MODULE_BASE_RE = re.compile(
    r"(?:^|,)\s*(?:(?:public|protected|private)\s+)?"
    r"(?:(?:::)?sc_core\s*::\s*)?sc_module\b"
)
_SC_MODULE_MACRO_RE = re.compile(
    r"\bSC_MODULE\s*\(\s*(?P<name>[A-Za-z_]\w*)\s*\)"
)
_INCLUDE_RE = re.compile(
    r"^[ \t]*#[ \t]*include[ \t]*[<\"](?P<path>[^>\"]+)[>\"]",
    re.MULTILINE,
)
_SC_MAIN_RE = re.compile(r"\bsc_main\s*\(")
_CAMEL_CASE_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


@dataclass(frozen=True)
class ModuleDeclaration:
    name: str
    line: int


@dataclass(frozen=True)
class CheckResult:
    errors: tuple[str, ...]
    module_count: int


def _mask_token(token: str) -> str:
    """Replace a token with spaces while retaining line positions."""

    return "".join("\n" if character == "\n" else " " for character in token)


def _mask_comments_and_literals(source: str) -> str:
    return _COMMENT_OR_LITERAL_RE.sub(lambda match: _mask_token(match.group()), source)


def _strip_comments(source: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group()
        if token.startswith("//") or token.startswith("/*"):
            return _mask_token(token)
        return token

    return _COMMENT_OR_LITERAL_RE.sub(replace, source)


def _line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _find_modules(source: str) -> list[ModuleDeclaration]:
    masked = _mask_comments_and_literals(source)
    declarations: list[ModuleDeclaration] = []

    for match in _CLASS_DEFINITION_RE.finditer(masked):
        tail = match.group("tail")
        colon = tail.find(":")
        if colon < 0 or not _SC_MODULE_BASE_RE.search(tail[colon + 1 :]):
            continue
        declarations.append(
            ModuleDeclaration(match.group("name"), _line_number(masked, match.start()))
        )

    for match in _SC_MODULE_MACRO_RE.finditer(masked):
        declarations.append(
            ModuleDeclaration(match.group("name"), _line_number(masked, match.start()))
        )

    return sorted(declarations, key=lambda declaration: declaration.line)


def _camel_to_snake(name: str) -> str:
    # Keep a run of capital letters together: UscCluster -> usc_cluster and
    # PVRCore -> pvr_core.
    first_pass = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    second_pass = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)
    return second_pass.lower()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _cmake_mentions_source(cmake_source: str, relative_source: str) -> bool:
    # Strip comments so a stale filename in a comment cannot satisfy the rule.
    without_comments = re.sub(r"(?m)#[^\n]*$", "", cmake_source)
    boundary = r"[A-Za-z0-9_./+-]"
    return (
        re.search(
            rf"(?<!{boundary}){re.escape(relative_source)}(?!{boundary})",
            without_comments,
        )
        is not None
    )


def check_layout(
    root: Path, model_dirs: Sequence[Path], cmake_path: Path
) -> CheckResult:
    root = root.resolve()
    resolved_model_dirs = [model_dir.resolve() for model_dir in model_dirs]
    cmake_path = cmake_path.resolve()
    errors: list[str] = []

    for model_dir in resolved_model_dirs:
        if not model_dir.is_dir():
            errors.append(f"model directory does not exist: {model_dir}")
    if errors:
        return CheckResult(tuple(errors), 0)
    if not cmake_path.is_file():
        return CheckResult((f"CMakeLists file does not exist: {cmake_path}",), 0)

    headers = sorted(
        path
        for model_dir in resolved_model_dirs
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in HEADER_SUFFIXES
    )
    sources = sorted(
        path
        for model_dir in resolved_model_dirs
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    cmake_source = cmake_path.read_text(encoding="utf-8")
    module_names: dict[str, Path] = {}
    module_count = 0

    for header in headers:
        source = header.read_text(encoding="utf-8")
        modules = _find_modules(source)
        display_header = _display_path(header, root)

        if len(modules) > 1:
            names = ", ".join(module.name for module in modules)
            errors.append(
                f"{display_header}: contains {len(modules)} SystemC modules "
                f"({names}); a module header must contain exactly one"
            )

        for module in modules:
            module_count += 1
            location = f"{display_header}:{module.line}"
            previous = module_names.get(module.name)
            if previous is not None:
                errors.append(
                    f"{location}: duplicate SystemC module {module.name}; first "
                    f"declared in {_display_path(previous, root)}"
                )
            else:
                module_names[module.name] = header

            if not _CAMEL_CASE_RE.fullmatch(module.name):
                errors.append(
                    f"{location}: module name {module.name!r} is not CamelCase"
                )

            expected_stem = _camel_to_snake(module.name)
            if header.suffix != ".h":
                errors.append(
                    f"{location}: module header must use .h, not {header.suffix}"
                )
            if header.stem != expected_stem:
                errors.append(
                    f"{location}: {module.name} must be declared in "
                    f"{expected_stem}.h, not {header.name}"
                )

            implementation = header.with_name(f"{header.stem}.cpp")
            display_implementation = _display_path(implementation, root)
            if not implementation.is_file():
                errors.append(
                    f"{location}: matching implementation is missing: "
                    f"{display_implementation}"
                )
                continue

            implementation_source = implementation.read_text(encoding="utf-8")
            includes = {
                Path(match.group("path")).name
                for match in _INCLUDE_RE.finditer(
                    _strip_comments(implementation_source)
                )
            }
            if header.name not in includes:
                errors.append(
                    f"{display_implementation}: does not include its own header "
                    f"{header.name}"
                )

            relative_implementation = _display_path(implementation, root)
            if not _cmake_mentions_source(cmake_source, relative_implementation):
                errors.append(
                    f"{display_implementation}: module implementation is not "
                    f"listed in {_display_path(cmake_path, root)}"
                )

    for implementation in sources:
        source = implementation.read_text(encoding="utf-8")
        modules = _find_modules(source)
        if not modules:
            continue

        display_implementation = _display_path(implementation, root)
        names = ", ".join(module.name for module in modules)
        errors.append(
            f"{display_implementation}: declares SystemC module(s) {names}; "
            "module declarations belong in their matching .h file"
        )
        if _SC_MAIN_RE.search(_mask_comments_and_literals(source)):
            errors.append(
                f"{display_implementation}: sc_main source must not define a "
                "SystemC module"
            )

    return CheckResult(tuple(errors), module_count)


def _write_fixture(root: Path, *, valid: bool = True) -> None:
    model_dir = root / "model_stub"
    model_dir.mkdir(parents=True)
    (model_dir / "foo_bar.h").write_text(
        "#include <systemc>\n"
        "class FooBar final : public sc_core::sc_module {\n"
        " public:\n"
        "  explicit FooBar(sc_core::sc_module_name name);\n"
        "};\n",
        encoding="utf-8",
    )
    (model_dir / "foo_bar.cpp").write_text(
        '#include "foo_bar.h"\n'
        "FooBar::FooBar(sc_core::sc_module_name name) : sc_module(name) {}\n",
        encoding="utf-8",
    )
    (model_dir / "types.h").write_text("struct Payload {};\n", encoding="utf-8")
    (model_dir / "sc_main.cpp").write_text(
        "int sc_main(int, char**) { return 0; }\n", encoding="utf-8"
    )
    cmake_entry = "model_stub/foo_bar.cpp" if valid else ""
    (root / "CMakeLists.txt").write_text(
        f"add_executable(model {cmake_entry} model_stub/sc_main.cpp)\n",
        encoding="utf-8",
    )


def run_self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="pvrgpu-layout-check-") as temp:
        root = Path(temp)
        _write_fixture(root)
        result = check_layout(
            root, [root / "model_stub"], root / "CMakeLists.txt"
        )
        if result.errors or result.module_count != 1:
            failures.append(f"valid fixture failed: {result.errors}")

    cases = {
        "missing CMake source": (
            lambda root: _write_fixture(root, valid=False),
            "not listed",
        ),
        "multiple modules in header": (
            lambda root: (
                _write_fixture(root),
                (root / "model_stub" / "foo_bar.h").write_text(
                    (root / "model_stub" / "foo_bar.h").read_text(encoding="utf-8")
                    + "class Extra : public sc_core::sc_module {};\n",
                    encoding="utf-8",
                ),
            ),
            "contains 2 SystemC modules",
        ),
        "module declared in sc_main": (
            lambda root: (
                _write_fixture(root),
                (root / "model_stub" / "sc_main.cpp").write_text(
                    "class Bad : public sc_core::sc_module {};\n"
                    "int sc_main(int, char**) { return 0; }\n",
                    encoding="utf-8",
                ),
            ),
            "sc_main source must not define",
        ),
        "missing own-header include": (
            lambda root: (
                _write_fixture(root),
                (root / "model_stub" / "foo_bar.cpp").write_text(
                    "FooBar::FooBar(sc_core::sc_module_name name)"
                    " : sc_module(name) {}\n",
                    encoding="utf-8",
                ),
            ),
            "does not include its own header",
        ),
    }

    for name, (prepare, expected_error) in cases.items():
        with tempfile.TemporaryDirectory(prefix="pvrgpu-layout-check-") as temp:
            root = Path(temp)
            prepare(root)
            result = check_layout(
                root, [root / "model_stub"], root / "CMakeLists.txt"
            )
            if not any(expected_error in error for error in result.errors):
                failures.append(
                    f"{name} did not report {expected_error!r}: {result.errors}"
                )

    if failures:
        for failure in failures:
            print(f"SELF-TEST FAIL: {failure}", file=sys.stderr)
        return 1

    print("SystemC module layout checker self-test: PASS")
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
        "--model-dir",
        type=Path,
        action="append",
        help=(
            "model source directory; repeat for multiple roots "
            "(default: <root>/model_stub and <root>/src/systemc)"
        ),
    )
    parser.add_argument(
        "--cmake",
        type=Path,
        help="CMakeLists file (default: <root>/CMakeLists.txt)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the checker's built-in temporary-fixture tests",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        return run_self_test()

    root = args.root.resolve()
    model_dirs = args.model_dir or [root / "model_stub", root / "src/systemc"]
    cmake_path = (args.cmake or root / "CMakeLists.txt").resolve()
    result = check_layout(root, model_dirs, cmake_path)

    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"SystemC module layout check: FAIL "
            f"({len(result.errors)} error(s), {result.module_count} module(s))",
            file=sys.stderr,
        )
        return 1

    print(
        f"SystemC module layout check: PASS "
        f"({result.module_count} module(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
