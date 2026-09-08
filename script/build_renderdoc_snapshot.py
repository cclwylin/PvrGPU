#!/usr/bin/env python3
"""Build five locked RenderDoc snapshot TUs in a fresh private source overlay.

SOURCE BUILD_BASE OUTDIR must identify an exact locked Mesa/vendor source tree,
an existing compatible Ninja build, and a new directory outside either tree and
outside this repository. Ninja is queried with '-t commands' only. Compiler
flags, architecture and optimization are inherited verbatim; no shared source,
object, dependency file or library is rebuilt/overwritten.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from run_drawlist_replay import (ReplayError, atomic_json, digest_string,
                                file_identity, read_json, require, safe_path)

REPO = Path(__file__).resolve().parents[1]
LOCK_SCHEMA = "pvrgpu.renderdoc-snapshot-source-lock.v1"
DEFAULT_LOCK = REPO / "third_party/renderdoc-drawlist-snapshot-v1.lock.json"
DEFAULT_PATCH = REPO / "third_party/renderdoc-drawlist-snapshot.patch"
TARGETS = {
    **{f"renderdoc/driver/gl/{name}.cpp": f"renderdoc/driver/gl/CMakeFiles/rdoc_gl.dir/{name}.cpp.o"
       for name in ("gl_driver", "gl_replay", "gl_resources", "gl_initstate")},
    "renderdoc/replay/replay_controller.cpp": "renderdoc/CMakeFiles/rdoc.dir/replay/replay_controller.cpp.o",
}
CHANGED_EXISTING = set(TARGETS) | {
    "renderdoc/driver/gl/gl_driver.h", "renderdoc/driver/gl/gl_replay.h",
    "renderdoc/driver/gl/gl_manager.h", "renderdoc/replay/replay_controller.h",
}
ADDED = {"renderdoc/replay/replay_snapshot.h", "renderdoc/replay/replay_snapshot_api.inl",
         "renderdoc/driver/gl/gl_replay_snapshot.inl", "renderdoc/driver/gl/gl_snapshot_codec_safety.inl"}


def command_output(args: list[str], cwd: Path | None = None) -> str:
    process = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    require(process.returncode == 0, f"command failed: {shlex.join(args)}\n{process.stderr}")
    return process.stdout


def relative_name(value: str) -> str:
    path = Path(value)
    require(value and not path.is_absolute() and ".." not in path.parts and
            path.as_posix() == value and "\0" not in value and "\n" not in value,
            f"unsafe relative source path: {value!r}")
    return value


def source_names(source: Path, git: str) -> tuple[list[str], set[str]]:
    tracked = command_output([git, "-C", str(source), "ls-files", "-z"]).split("\0")
    others = command_output([git, "-C", str(source), "ls-files", "--others", "--exclude-standard", "-z"]).split("\0")
    return sorted(relative_name(p) for p in tracked if p), {relative_name(p) for p in others if p}


def tree_digest(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8") + b"\0" + hashes[name].encode("ascii") + b"\n")
    return digest.hexdigest()


def source_hash(path: Path) -> str:
    # Empty tracked files are valid source. Runtime/artifact identities require
    # nonempty bytes, so source hashing intentionally has a separate helper.
    safe_path(path)
    require(path.is_file(), f"missing/nonregular tracked source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_lock(path: Path, patch: Path) -> dict:
    lock = read_json(path)
    require(lock.get("schema") == LOCK_SCHEMA, "unsupported snapshot source-lock schema")
    head = lock.get("source_git_head")
    require(isinstance(head, str) and len(head) == 40 and all(c in "0123456789abcdef" for c in head),
            "invalid locked Git HEAD")
    digest_string(lock.get("baseline_tracked_tree_sha256"), "baseline source tree SHA-256")
    require(file_identity(patch)["sha256"] == lock.get("patch_sha256"), "snapshot patch hash mismatch")
    files = lock.get("files")
    require(isinstance(files, dict) and set(files) == CHANGED_EXISTING | ADDED,
            "lock must describe exactly the nine existing and four new snapshot files")
    for name, entry in files.items():
        relative_name(name)
        require(isinstance(entry, dict), f"invalid lock entry: {name}")
        digest_string(entry.get("patched_sha256"), f"patched {name}")
        if name in ADDED:
            require(entry.get("base_sha256") is None, f"added file has baseline bytes: {name}")
        else:
            digest_string(entry.get("base_sha256"), f"baseline {name}")
    return lock


def inspect_source(source: Path, lock: dict, git: str) -> tuple[list[str], dict[str, str]]:
    require(command_output([git, "-C", str(source), "rev-parse", "HEAD"]).strip() == lock["source_git_head"],
            "RenderDoc source Git HEAD does not match lock")
    names, others = source_names(source, git)
    require(CHANGED_EXISTING <= set(names), "baseline tracked source is incomplete")
    require(not (set(names) & ADDED), "snapshot additions unexpectedly tracked in baseline tree")
    require(others <= ADDED, "unknown untracked RenderDoc source files: " + ", ".join(sorted(others - ADDED)))
    hashes = {name: source_hash(source / name) for name in names}
    normalized = dict(hashes)
    for name in CHANGED_EXISTING:
        entry = lock["files"][name]
        require(hashes[name] in (entry["base_sha256"], entry["patched_sha256"]), f"unrecognized source bytes: {name}")
        normalized[name] = entry["base_sha256"]
    for name in ADDED:
        path = source / name
        if path.exists() or path.is_symlink():
            hashes[name] = source_hash(path)
            require(hashes[name] == lock["files"][name]["patched_sha256"], f"unrecognized added source bytes: {name}")
    require(tree_digest(normalized) == lock["baseline_tracked_tree_sha256"],
            "normalized tracked source differs from pinned Mesa/vendor baseline")
    return names, hashes


def patch_names(patch: Path, git: str) -> set[str]:
    # Git filters patches relative to the current subdirectory when cwd is
    # inside a repository. Inventory from the filesystem root, not third_party,
    # so a perfectly valid external-source patch cannot appear empty.
    text = command_output([git, "apply", "--numstat", "-z", str(patch)], cwd=Path(patch.anchor))
    names = set()
    for entry in text.split("\0"):
        if not entry:
            continue
        fields = entry.split("\t", 2)
        require(len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit(),
                "snapshot patch must contain plain source text, not binary/renames")
        names.add(relative_name(fields[2]))
    return names


def prepare_overlay(source: Path, overlay: Path, names: list[str], hashes: dict[str, str],
                    lock: dict, patch: Path, git: str) -> None:
    require(patch_names(patch, git) == set(lock["files"]), "patch file set differs from lock")
    overlay.mkdir()
    # Ensure Git never discovers an unrelated ancestor worktree and applies a
    # subdirectory path filter. This metadata is private and never committed.
    command_output([git, "init", "--quiet", str(overlay)])
    for name in sorted(hashes):
        destination = overlay / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination, follow_symlinks=False)
        require(source_hash(destination) == hashes[name], f"source changed during overlay copy: {name}")
    # Both clean-baseline and already-patched source trees are supported. Mixed
    # exact files also normalize independently; arbitrary partial edits refuse.
    for name in sorted(lock["files"]):
        entry = lock["files"][name]
        if hashes.get(name) == entry["patched_sha256"]:
            command_output([git, "apply", "--reverse", "--include=" + name, str(patch)], cwd=overlay)
    normalized = {name: source_hash(overlay / name) for name in names}
    require(tree_digest(normalized) == lock["baseline_tracked_tree_sha256"], "private reverse patch did not restore baseline")
    require(not any((overlay / name).exists() for name in ADDED), "private reverse patch retained added files")
    command_output([git, "apply", "--check", str(patch)], cwd=overlay)
    command_output([git, "apply", str(patch)], cwd=overlay)
    for name, entry in lock["files"].items():
        require(source_hash(overlay / name) == entry["patched_sha256"], f"patched overlay hash mismatch: {name}")


def shell_segments(line: str, *, strict: bool = True) -> list[list[str]]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    parts: list[list[str]] = [[]]
    for token in lexer:
        if token == "&&":
            require(parts[-1], "empty Ninja shell segment")
            parts.append([])
        else:
            require(not strict or token not in (";", "|", "||", "&", ">", ">>", "<", "<<", "&>"),
                    "unsupported Ninja shell redirection/control operator")
            parts[-1].append(token)
    require(parts[-1], "empty Ninja command")
    return parts


def option(args: list[str], flag: str) -> int:
    require(args.count(flag) == 1 and args.index(flag) + 1 < len(args), f"expected exactly one {flag} option")
    return args.index(flag) + 1


def query_command(ninja: str, base: Path, target: str) -> tuple[list[str], str]:
    output = command_output([ninja, "-C", str(base), "-t", "commands", target])
    found = []
    for line in output.splitlines():
        if not line.strip():
            continue
        # '-t commands' also prints prerequisites. Never execute or reinterpret
        # those generator commands, which may legitimately use shell redirects.
        parts = shell_segments(line, strict=False)
        for args in parts:
            if args.count("-o") != 1 or args.index("-o") + 1 >= len(args):
                continue
            at = option(args, "-o")
            if (base / args[at]).resolve() != (base / target).resolve():
                continue
            require(all(part is args or part == [":"] for part in parts),
                    "target Ninja command has unsupported prerequisite/post-link shell steps")
            shell_segments(line)
            require(not any(arg.startswith("@") and not (i > 0 and args[i - 1] == "-install_name")
                            for i, arg in enumerate(args)), "Ninja response files require explicit support")
            require(not any(arg.startswith(("-save-temps", "-MJ", "--serialize-diagnostics", "-serialize-diagnostics",
                                           "-fprofile-generate", "-fprofile-instr-generate", "-ftime-trace"))
                            for arg in args), "compiler command has unsupported auxiliary output flags")
            found.append(args)
    require(len(found) == 1, f"could not uniquely derive direct compiler command for {target}")
    return found[0], output


def relocate_source_argument(arg: str, source: Path, overlay: Path, base: Path) -> str:
    # Current locked commands use absolute source includes/translation units.
    # Handle relative -I paths as well, without changing any non-path flag.
    if str(source) in arg:
        return arg.replace(str(source), str(overlay))
    prefix, value = ("-I", arg[2:]) if arg.startswith("-I") else ("", arg)
    if value and not value.startswith("-"):
        resolved = (base / value).resolve()
        if resolved == source or source in resolved.parents:
            return prefix + str(overlay / resolved.relative_to(source))
    return arg


def executable_identity(value: str) -> dict:
    resolved = shutil.which(value) if not Path(value).is_absolute() else value
    require(resolved is not None, f"executable not found: {value}")
    path = safe_path(resolved, symlinks=True)
    require(os.access(path, os.X_OK), f"not executable: {path}")
    return file_identity(path)


def protected_inputs(args: list[str], base: Path) -> set[Path]:
    result = set()
    for arg in args[1:]:
        values = arg.split(",")[1:] if arg.startswith("-Wl,") else [arg]
        for value in values:
            if not value or value.startswith("-") or value.startswith("@rpath/"):
                continue
            path = base / value
            if path.is_file():
                result.add(path.resolve())
    return result


def build(args: argparse.Namespace) -> Path:
    source, base, out = (safe_path(value) for value in (args.source, args.build_base, args.outdir))
    require(source.is_dir() and base.is_dir(), "source and Ninja build must exist")
    require(not out.exists() and out.parent.is_dir(), "output must be fresh with an existing parent")
    for protected in (source, base, REPO):
        require(out != protected and out not in protected.parents and protected not in out.parents,
                "private build output must be outside source/build/repository trees")
    require(args.jobs in (1, 2), "private snapshot build supports one or two compiler workers")
    lock_path, patch = safe_path(args.lock), safe_path(args.patch)
    lock_identity, patch_identity = file_identity(lock_path), file_identity(patch)
    lock = load_lock(lock_path, patch)
    require(file_identity(lock_path) == lock_identity and file_identity(patch) == patch_identity,
            "snapshot patch/lock changed during preflight")
    git = executable_identity(args.git)
    ninja = executable_identity(args.ninja)
    cache = base / "CMakeCache.txt"
    values = [line.split("=", 1)[1] for line in cache.read_text().splitlines()
              if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
    require(len(values) == 1 and safe_path(values[0]) == source, "Ninja build was configured for a different source")
    names, source_before = inspect_source(source, lock, git["path"])
    libraries = [p for p in (base / "lib/librenderdoc.dylib", base / "lib/librenderdoc.so") if p.is_file()]
    require(len(libraries) == 1, "expected one existing base RenderDoc shared library")
    original_library = libraries[0]
    library_target = original_library.relative_to(base).as_posix()
    commands, raw_queries = {}, {}
    for name, target in TARGETS.items():
        commands[name], raw_queries[target] = query_command(ninja["path"], base, target)
        original = commands[name]
        require((base / original[option(original, "-c")]).resolve() == source / name,
                f"Ninja object compiles the wrong source: {target}")
        option(original, "-MF"); option(original, "-MT")
    link, raw_queries[library_target] = query_command(ninja["path"], base, library_target)
    compiler_ids = {value["path"]: value for value in
                    (executable_identity(command[0]) for command in [*commands.values(), link])}
    # All reused objects/archives, explicit link inputs, Ninja metadata and the
    # original outputs are hashed before and after. Compiler/ninja tools never
    # execute a shared build or write the shared depfiles.
    protected = {cache, base / "build.ninja", original_library}
    for name in ("CMakeFiles/rules.ninja", ".ninja_log", ".ninja_deps"):
        if (base / name).is_file():
            protected.add(base / name)
    for command in [*commands.values(), link]:
        protected.update(protected_inputs(command, base))
    protected.update(base / target for target in TARGETS.values())
    before = {str(path): source_hash(path) for path in sorted(protected)}
    out.mkdir(mode=0o700)
    for name in ("objects", "lib", "logs"):
        (out / name).mkdir(mode=0o700)
    # Keep the exact inputs beside the receipt. The repository's next source
    # generation must not erase the provenance of a completed private build.
    private_lock, private_patch = out / "source-lock.json", out / "source.patch"
    for original, destination, identity in ((lock_path, private_lock, lock_identity),
                                            (patch, private_patch, patch_identity)):
        shutil.copy2(original, destination, follow_symlinks=False)
        require(source_hash(destination) == identity["sha256"], "patch/lock changed during private copy")
    overlay = out / "source"
    prepare_overlay(source, overlay, names, source_before, lock, private_patch, git["path"])
    library = out / "lib" / ("librenderdoc-drawlist-snapshot" + original_library.suffix)
    private_commands = {}
    replacements = {}
    for name, original in commands.items():
        command = [relocate_source_argument(arg, source, overlay, base) for arg in original]
        target = out / "objects" / Path(TARGETS[name]).name
        command[option(command, "-o")] = str(target)
        command[option(command, "-MF")] = str(target) + ".d"
        command[option(command, "-MT")] = str(target)
        require(str(source) not in "\n".join(command), "compiler still references mutable source tree")
        private_commands[name] = command
        replacements[(base / TARGETS[name]).resolve()] = target
    private_link = [relocate_source_argument(arg, source, overlay, base) for arg in link]
    private_link[option(private_link, "-o")] = str(library)
    if original_library.suffix == ".dylib":
        private_link[option(private_link, "-install_name")] = str(library)
    else:
        sonames = [i for i, value in enumerate(private_link) if value.startswith("-Wl,-soname,")]
        require(len(sonames) == 1, "unsupported shared-library SONAME command")
        private_link[sonames[0]] = "-Wl,-soname," + library.name
    replaced = set()
    for i, arg in enumerate(private_link):
        resolved = (base / arg).resolve()
        if resolved in replacements:
            private_link[i] = str(replacements[resolved]); replaced.add(resolved)
    require(replaced == set(replacements), "shared library does not directly link all five changed objects")
    receipt = {"schema": "pvrgpu.renderdoc-snapshot-build.v1", "status": "started",
        "source": str(source), "source_git_head": lock["source_git_head"], "base_build": str(base),
        "overlay": str(overlay), "baseline_tracked_tree_sha256": lock["baseline_tracked_tree_sha256"],
        "source_input_tree_sha256": tree_digest(source_before), "source_file_sha256": source_before,
        "lock": lock_identity, "patch": patch_identity,
        "private_lock": file_identity(private_lock), "private_patch": file_identity(private_patch),
        "builder_sources": [file_identity(REPO / name) for name in
                            ("script/build_renderdoc_snapshot.py", "script/build_renderdoc_snapshot.sh",
                             "script/run_drawlist_replay.py")],
        "git": git, "ninja": ninja, "compilers": list(compiler_ids.values()),
        "compiler_versions": {path: command_output([path, "--version"]) for path in compiler_ids},
        "original_compile_commands": commands, "compile_commands": private_commands,
        "original_link_command": link, "link_command": private_link,
        "ninja_command_queries": raw_queries, "protected_input_sha256": before,
        "jobs": args.jobs, "library": str(library)}
    atomic_json(out / "build-invocation.json", receipt)

    def execute_one(name: str, command: list[str]) -> None:
        print("Building private " + name, flush=True)
        with (out / "logs" / (name + ".stdout.log")).open("xb") as stdout, (out / "logs" / (name + ".stderr.log")).open("xb") as stderr:
            result = subprocess.run(command, cwd=base, stdout=stdout, stderr=stderr)
        require(result.returncode == 0, f"private command failed: {name}; inspect {out / 'logs'}")

    failure = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(execute_one, Path(name).stem, command) for name, command in private_commands.items()]
            for future in futures:
                future.result()
        execute_one("link", private_link)
    except Exception as error:
        failure = str(error)
    # Even compilation failure retains the no-shared-mutation evidence.
    _, source_after = inspect_source(source, lock, git["path"])
    after = {str(path): source_hash(path) for path in sorted(protected)}
    require(source_after == source_before, "source changed while private snapshot build was running")
    require(after == before, "shared build input/output changed while private build was running")
    for value in [git, ninja, *compiler_ids.values()]:
        require(file_identity(Path(value["path"])) == value, "build tool changed during compilation")
    require(file_identity(lock_path) == receipt["lock"] and file_identity(patch) == receipt["patch"],
            "snapshot patch/lock changed during compilation")
    for identity in [receipt["private_lock"], receipt["private_patch"], *receipt["builder_sources"]]:
        require(file_identity(Path(identity["path"])) == identity, "private provenance/builder source changed during compilation")
    receipt.update(status="FAIL" if failure else "PASS", source_unchanged=True, shared_outputs_unchanged=True)
    if failure:
        receipt["error"] = failure
    else:
        receipt["library_identity"] = file_identity(library)
        receipt["object_identities"] = [file_identity(path) for path in replacements.values()]
    atomic_json(out / "build-receipt.json", receipt)
    require(failure is None, failure or "")
    return library


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", metavar="SOURCE")
    parser.add_argument("build_base", metavar="BUILD_BASE")
    parser.add_argument("outdir", metavar="OUTDIR")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--patch", default=str(DEFAULT_PATCH))
    parser.add_argument("--jobs", type=int, default=2, choices=(1, 2))
    parser.add_argument("--ninja", default="ninja")
    parser.add_argument("--git", default="git")
    try:
        library = build(parser.parse_args())
    except (ReplayError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"build_renderdoc_snapshot: {error}", file=sys.stderr)
        return 1
    print(library)
    return 0


if __name__ == "__main__":
    sys.exit(main())
