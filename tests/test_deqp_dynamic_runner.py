from pathlib import Path
import os
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "script/run_deqp_dynamic.sh"


class DynamicRunnerTests(unittest.TestCase):
    def test_process_exit_survives_optional_loader_filter(self) -> None:
        source = RUNNER.read_text()
        start = source.index('    case_start_ms="$(now_ms)"')
        end = source.index('    case_end_ms="$(now_ms)"', start)
        # Execute the actual runner pipeline, not a second implementation.
        pipeline = source[start:end]
        with tempfile.TemporaryDirectory(prefix="pvrgpu-runner-test-") as output:
            for verify in (0, 1):
                for code in (0, 1, 17, 124, 137):
                    for loader_only in (0, 1):
                        with self.subTest(verify=verify, code=code, loader_only=loader_only):
                            shell = r'''
set -uo pipefail
now_ms() { printf '0'; }
run_one_case() {
    printf 'dyld[123]: loaded library\n'
    if (( ! loader_only )); then printf 'test output\n'; fi
    return "${test_exit}"
}
''' + pipeline + '\nprintf "captured_exit=%s\\n" "${exit_code}"\n'
                            env = dict(os.environ, opt_verify_link=str(verify),
                                       test_exit=str(code), loader_only=str(loader_only),
                                       log_path=str(Path(output) / "run.log"))
                            result = subprocess.run(["bash", "-c", shell], env=env,
                                                    capture_output=True, text=True, check=True)
                            self.assertIn(f"captured_exit={code}\n", result.stdout)
                            self.assertIn("dyld[123]", (Path(output) / "run.log").read_text())

    def test_default_memory_mode_matches_graphics_and_preserves_override(self) -> None:
        line = next(line for line in RUNNER.read_text().splitlines()
                    if line.startswith("export PVRGPU_MODEL_MEMORY_MODE="))
        for value, expected in ((None, "cache"), ("", "cache"), ("cache", "cache"),
                                ("direct", "direct"), ("bypass", "bypass")):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("PVRGPU_MODEL_MEMORY_MODE", None)
                if value is not None:
                    env["PVRGPU_MODEL_MEMORY_MODE"] = value
                result = subprocess.run(["bash", "-c", line +
                                         '\nprintf "%s" "$PVRGPU_MODEL_MEMORY_MODE"'],
                                        env=env, capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout, expected)

    def test_default_texture_lod_mode_is_exact_and_preserves_override(self) -> None:
        line = next(line for line in RUNNER.read_text().splitlines()
                    if line.startswith("export PVRGPU_TEXTURE_LOD_MODE="))
        for value, expected in ((None, "exact"), ("", "exact"),
                                ("exact", "exact"),
                                ("llvmpipe", "llvmpipe")):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("PVRGPU_TEXTURE_LOD_MODE", None)
                if value is not None:
                    env["PVRGPU_TEXTURE_LOD_MODE"] = value
                result = subprocess.run(
                    ["bash", "-c", line +
                     '\nprintf "%s" "$PVRGPU_TEXTURE_LOD_MODE"'],
                    env=env, capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
