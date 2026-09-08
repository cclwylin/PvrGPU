#!/usr/bin/env python3
"""Check formal player fail-closed path/environment guards without replaying GL.

Usage: python3 tests/check_renderdoc_player_guards.py PLAYER EXISTING_CAPTURE.rdc
The supplied capture is only read/canonicalized; no input is rewritten.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    player, capture = (Path(value).resolve(strict=True) for value in sys.argv[1:])
    checks = 0
    with tempfile.TemporaryDirectory(prefix="formal-player-guards-") as temporary:
        root = Path(temporary)

        def run(name, arguments, reason, extra_env=None, existing=None):
            nonlocal checks
            case = root / name
            case.mkdir()
            png, receipt = case / "final.png", case / "receipt.json"
            env = os.environ.copy()
            for key in tuple(env):
                if key.startswith("PVRGPU_SYSTEMC_") or key.startswith("PVRGPU_DRIVER_"):
                    del env[key]
            env.update(GALLIUM_DRIVER="llvmpipe", PVRGPU_RDC_FINAL_OUTPUT_RECEIPT=str(receipt))
            if extra_env:
                env.update(extra_env(case))
            preserved = {}
            if existing:
                for path in existing(case):
                    path.write_bytes(b"preserve existing user artifact\n")
                    preserved[path] = path.read_bytes()
            result = subprocess.run([str(player), *arguments(case)], env=env,
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode != 0, (name, result.stdout, result.stderr)
            assert reason in result.stderr, (name, reason, result.stderr)
            assert not png.exists() or png in preserved, name
            assert not receipt.exists() or receipt in preserved, name
            for path, contents in preserved.items():
                assert path.read_bytes() == contents, (name, "mutated", path)
            checks += 1

        run("usage", lambda _: [], "Usage:")
        run("missing-capture", lambda d: [str(d / "missing.rdc"), str(d / "final.png")], "Formal replay failed:")
        run("input-png-alias", lambda _: [str(capture), str(capture)], "Input/output paths must be distinct")
        run("input-receipt-alias", lambda d: [str(capture), str(d / "final.png")],
            "Input/output paths must be distinct", lambda _: {"PVRGPU_RDC_FINAL_OUTPUT_RECEIPT": str(capture)})
        run("output-alias", lambda d: [str(capture), str(d / "receipt.json")], "Input/output paths must be distinct")
        run("input-trace-alias", lambda d: [str(capture), str(d / "final.png"), str(capture)],
            "Input/output paths must be distinct")
        run("png-trace-alias", lambda d: [str(capture), str(d / "final.png"), str(d / "final.png")],
            "Input/output paths must be distinct")
        run("receipt-trace-alias", lambda d: [str(capture), str(d / "final.png"), str(d / "receipt.json")],
            "Input/output paths must be distinct")
        run("existing-png", lambda d: [str(capture), str(d / "final.png")], "Refusing existing final PNG/receipt",
            existing=lambda d: [d / "final.png"])
        run("existing-receipt", lambda d: [str(capture), str(d / "final.png")], "Refusing existing final PNG/receipt",
            existing=lambda d: [d / "receipt.json"])
        run("existing-trace", lambda d: [str(capture), str(d / "final.png"), str(d / "trace.md")],
            "Refusing existing final PNG/receipt", existing=lambda d: [d / "trace.md"])
        def aliased_output(d):
            (d / "alias").symlink_to(d, target_is_directory=True)
            return {"PVRGPU_RDC_FINAL_OUTPUT_RECEIPT": str(d / "alias" / "final.png")}
        run("symlink-parent-alias", lambda d: [str(capture), str(d / "final.png")],
            "Input/output paths must be distinct", aliased_output)
        run("missing-native-receipt", lambda d: [str(capture), str(d / "final.png")],
            "requires a final-output receipt", lambda _: {"GALLIUM_DRIVER": "pvrgpu", "PVRGPU_RDC_FINAL_OUTPUT_RECEIPT": ""})
        run("missing-native-runtime", lambda d: [str(capture), str(d / "final.png")],
            "requires native API library", lambda _: {"GALLIUM_DRIVER": "pvrgpu"})
        def native_environment(d):
            bridge = d / "unused-runtime.dylib"
            bridge.touch()  # Guard exits before loading this deliberate non-runtime fixture.
            return {"GALLIUM_DRIVER": "pvrgpu", "PVRGPU_SYSTEMC_API_LIB": str(bridge),
                    "PVRGPU_DRIVER_COUNTER_OUT": str(d / "driver-counter.txt"),
                    "PVRGPU_SYSTEMC_JSONL_OUT": str(d / "model.jsonl")}
        run("existing-initial-audit", lambda d: [str(capture), str(d / "final.png")],
            "Refusing existing initial-copy driver audit", native_environment,
            existing=lambda d: [d / "receipt.json.initial-copy-driver-counter.txt"])
        run("initial-audit-png-alias", lambda d: [str(capture), str(d / "receipt.json.initial-copy-driver-counter.txt")],
            "Initial-copy audit paths must be distinct", native_environment)
        def counter_alias(d):
            env = native_environment(d)
            env["PVRGPU_DRIVER_COUNTER_OUT"] = str(d / "receipt.json.initial-copy-driver-counter.txt")
            return env
        run("initial-audit-native-counter-alias", lambda d: [str(capture), str(d / "final.png")],
            "Initial-copy audit paths must be distinct", counter_alias)
    print(f"Formal player fail-closed guards PASS: {checks} cases")


if __name__ == "__main__":
    main()
