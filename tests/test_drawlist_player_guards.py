#!/usr/bin/env python3
"""Run pre-GL path/argument rejection tests against an explicitly built player."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("player", type=Path)
    args = parser.parse_args()
    player = args.player.resolve(strict=True)
    # Do not let inherited native configuration/preload state affect these
    # pre-GL tests. The selected binary still uses its exact linked RenderDoc.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
    env["GALLIUM_DRIVER"] = "llvmpipe"
    tests = 0
    with tempfile.TemporaryDirectory(prefix="drawlist-player-guards-") as directory:
        root = Path(directory).resolve()
        capture = root / "input.rdc"
        capture.write_bytes(b"not a capture: parsing guards must run before OpenCapture\n")
        original = capture.read_bytes()
        output, receipt, state = root / "state.bin", root / "receipt.json", root / "saved.bin"
        state.write_bytes(b"saved state fixture")
        normal = [str(capture), "--stop-after-draw", "147", "--state-out", str(output),
                  "--receipt-out", str(receipt)]

        def run(argv, expected, *, environment=None, success=False):
            nonlocal tests
            result = subprocess.run([str(player), *argv], env=environment or env,
                                    text=True, capture_output=True, timeout=15)
            text = result.stdout + result.stderr
            assert (result.returncode == 0) == success, (argv, result.returncode, text)
            assert expected in text, (argv, expected, text)
            assert capture.read_bytes() == original
            assert not output.exists() and not receipt.exists()
            tests += 1

        run(["--help"], "zero-based", success=True)
        run([], "Missing capture")
        run([str(capture)], "are required")
        run(normal + ["--unknown", "1"], "Unknown option")
        run(normal + ["--color-out"], "Missing option value")
        run(normal + ["--stop-after-draw", "0"], "Duplicate draw")
        run([str(capture), "--stop-after-draw", "-1"], "unsigned decimal")
        run([str(capture), "--stop-after-draw", "4294967296"], "exceeds replay")
        run(normal + ["--expected-resume-event", "6482"], "Resume requires")
        run(normal + ["--state-in", str(state)], "Resume requires")
        run(normal + ["--state-in", str(state), "--expected-resume-event", "0"], "Resume requires")
        run([str(capture), "--stop-after-draw", "0", "--state-out", str(capture),
             "--receipt-out", str(receipt)], "Output already exists")
        run([str(capture), "--stop-after-draw", "0", "--state-out", str(output),
             "--receipt-out", str(output)], "paths must be distinct")
        run(normal + ["--color-out", str(output)], "paths must be distinct")
        link = root / "linked.rdc"
        link.symlink_to(capture)
        run([str(link), *normal[1:]], "Symlink is not allowed")
        run([str(root / ".." / root.name / capture.name), *normal[1:]], "Parent traversal")
        # All fresh output aliases must be rejected before loading the extension.
        native = dict(env, GALLIUM_DRIVER="pvrgpu", PVRGPU_SYSTEMC_API_LIB=str(state),
                      PVRGPU_DRIVER_COUNTER_OUT=str(receipt) + ".restore-driver.txt",
                      PVRGPU_SYSTEMC_JSONL_OUT=str(root / "model.jsonl"),
                      PVRGPU_DRIVER_COMMAND_OUT=str(root / "commands.txt"))
        run(normal, "paths must be distinct", environment=native)
        native["PVRGPU_DRIVER_COUNTER_OUT"] = str(root / "driver.txt")
        native["PVRGPU_DRIVER_COMMAND_OUT"] = str(output)
        run(normal, "paths must be distinct", environment=native)
    print(f"DrawList player pre-GL guards: {tests} PASS")


if __name__ == "__main__":
    main()
